import torch
import torch.nn as nn
import torch.nn.functional as F

import math, time
from dataclasses import dataclass
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class ModelConfig:
    d_model = 128
    seq_len = 32
    batch_size = 2
    n_blocks = 6
    q_heads = 8
    kv_heads = 4
    ffn_multiplier = None
    ffn_muliple_of = 32
    vocab_size = 64 # set later
    eps = 1e-6

## ROPE
def precompute_freqs_cis(head_dim, seq_len, theta: float = 10000.0):
    """
    precompute rope freqs for all possible value of m-theta to multiple with 
    q and k 
    """
    assert head_dim % 2 == 0, "head_dim must be divisible by 2"

    theta_numerator = torch.arange(0, head_dim, 2).float()
    freqs = 1.0 / (theta ** (theta_numerator / head_dim )).to(DEVICE)
    m = torch.arange(seq_len, device=DEVICE)
    freqs = torch.outer(m, freqs).float()

    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
    ## convert to pair of numbers for complex number representation
    q_cmplx = torch.view_as_complex( q.float().reshape(*q.shape[:-1], -1, 2 ) )
    k_cmplx = torch.view_as_complex( k.float().reshape(*k.shape[:-1], -1, 2 ) )
    ## add fake batch and heads
    freqs = freqs.unsqueeze(0).unsqueeze(2)                                   # (seq_len, head_dim/ 2) -> (1, seq_len, 1, head_dim/ 2)
    ## view as real and reshape to original
    q_rotated = torch.view_as_real(q_cmplx * freqs).reshape(*q.shape)
    k_rotated = torch.view_as_real(k_cmplx * freqs).reshape(*k.shape)
    return q_rotated, k_rotated

class GQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.q_heads = config.q_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.d_model // self.q_heads
        self.n_groups = self.q_heads // self.kv_heads

        self.q_proj =  nn.Linear(config.d_model, self.q_heads * self.head_dim , bias=False)
        self.k_proj =  nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias=False)
        self.v_proj =  nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        ## max seq_len for rot emb/ kv cache
        max_pos = 2 * self.seq_len
        if max_pos < 2048: max_pos = 2048
        freqs_cis = precompute_freqs_cis(self.head_dim, max_pos)
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        
        self.register_buffer('trill', torch.tril(torch.ones((config.seq_len, config.seq_len))), persistent=False)
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, step=None):
        B, T, C = x.shape
        # q(x)       -> B T C -> B T C
        # k(x), v(x) -> B T C -> B T (C// self.n_groups)
        # split C -> n_heads * heads_dim
        q, k, v = (
            self.q_proj(x).reshape( B, T, self.q_heads, self.head_dim ),
            self.k_proj(x).reshape( B, T, self.kv_heads, self.head_dim ),
            self.v_proj(x).reshape( B, T, self.kv_heads, self.head_dim ),
        )
        # during training whole seq length is needed
        if step is None:
            q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])
        else:
            q, k = apply_rotary_emb(q, k, self.freqs_cis[step:step+T])

        ## use kv cache
        if step is not None:
            if self.k_cache is None:
                self.k_cache = torch.zeros(B, self.seq_len, self.kv_heads, self.head_dim, device=x.device)
                self.v_cache = torch.zeros(B, self.seq_len, self.kv_heads, self.head_dim, device=x.device)

            self.k_cache[:, step:step+T] = k
            self.v_cache[:, step:step+T] = v
            k = self.k_cache[:, :step+T]
            v = self.v_cache[:, :step+T]

        # since C of k and v is less, repeat in heads dim.
        k = k.repeat_interleave(self.n_groups, dim=2)
        v = v.repeat_interleave(self.n_groups, dim=2)

        ## swap head and T
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        ## causal attn
        attn = ( q @ k.transpose(-2,-1) ) * (self.head_dim ** -0.5)
        if step is None:
            attn = attn.masked_fill(self.trill[:T, :T] == 0, float("-inf"))
        else:
            mask = self.trill[:attn.size(-2), :attn.size(-1)].to(x.device)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        # swap H, T and then back to B T C
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class RMSNorm(nn.Module):
    """
    Refer formula; shape in - shape out
    1. calculate rms_norm 
    2. x * rms_norm * gain
    """
    def __init__(self, config):
        super().__init__()
        self.eps = config.eps
        self.gain = nn.Parameter(torch.ones(config.d_model)) # gain param like gamma in layerNorm
    
    def _norm(self, x):
        """
        RMS norm across emb dim/ feature layer
        """
        rms = torch.rsqrt( torch.mean(x**2, dim=-1, keepdim=True) + self.eps ) # 1/sqrt
        x_norm = x * rms
        return x_norm

    def forward(self, x):
        x_norm = self._norm(x.float()).type_as(x) # from any mixed precision to high precision float
        return self.gain * x_norm

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hddn_dim = 8 * config.d_model/ 3
        if config.ffn_multiplier is not None:
            hddn_dim = int( config.ffn_multiplier * hddn_dim )
        ## round off 
        hddn_dim = config.ffn_muliple_of * (( hddn_dim + config.ffn_muliple_of -1 ) // config.ffn_muliple_of )
        hddn_dim = int(hddn_dim)
        ## lin layers
        self.w1 = nn.Linear(config.d_model, hddn_dim, bias=False )
        self.w2 = nn.Linear(hddn_dim, config.d_model, bias=False )
        self.w3 = nn.Linear(config.d_model, hddn_dim, bias=False )

    def forward(self, x):
        swish = F.silu(self.w1(x))          # B T hddn 
        return self.w2( self.w3(x) * swish) # B T hddn -> B T d_model

class Block(nn.Module):
    def __init__(self, config):
        """
        refer block for flow
        """
        super().__init__()
        self.attn_norm = RMSNorm(config=config)
        self.attn = GQA(config=config)
        self.ff_norm = RMSNorm(config=config)
        self.ff = FeedForward(config=config)
    
    def forward(self, x, step=None):
        """
        receives (B T C)
        returns (B T C)
        """
        ## refer diagram for the logic
        x = x + self.attn( self.attn_norm(x), step=step )
        x = x + self.ff( self.ff_norm(x) )
        return x

class LLAMAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.Sequential(* [Block(config=config) for _ in range(config.n_blocks) ] )
        self.out_norm = RMSNorm(config)
        self.out_proj = nn.Linear( config.d_model , config.vocab_size )

    def forward(self, X, step=None, targets=None):
        """
        B: Batch; T: time_step/ seq_len; C: Channel/ emb_dim/ d_model
        """
        B, T = X.shape

        x = self.tok_emb(X)            # (B T C)
        for block in self.blocks:      # (B T C)
            x = block(x, step=step)
        x = self.out_norm(x)           # (B T C)
        logits = self.out_proj(x)      # (B T vc)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)    # (B', C)
            targets = targets.view(B * T)   # (B', 1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, use_kv_cache = True):
        if use_kv_cache:
            for block in self.blocks:
                block.attn.k_cache = None
                block.attn.v_cache = None

        for step in range(max_new_tokens):
            ## prompt processing phase
            ## during inital step or when kv cache disabled, we need to process whole context
            if (step == 0) or (not use_kv_cache):
                context = idx[:, -self.seq_len:]
            ## token generation phase
            ## send only the last generated token and reuse from kv-cache
            else:
                context = idx[:, -1:]

            # kv caching index
            pos = 0 if (step == 0) else idx.shape[1] - 1
            if use_kv_cache:
                logits, _ = self(context, step = pos)
            else:
                logits, _ = self(context)
            logits = logits[:, -1, :] # take the newly generated token
            probs = F.softmax(logits, dim=-1)
            idx_new = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_new), dim = 1)
        return idx

if __name__ == "__main__":
    ## load data and tokenize
    data_path = "../data/hp_book1.txt"
    txt = open(data_path).read()
    chars = sorted(list(set(txt)))
    stoi = {c:i for c, i in enumerate(chars)}
    itos = {i:c for c, i in enumerate(chars)}
    tokenizer = stoi

    # train-val 
    encode = lambda x: [tokenizer[i] for i in x]
    decode = lambda x: ''.join( [itos[i] for i in x] )
    dataset = torch.tensor(encode(txt), dtype=torch.long)

    config = ModelConfig()
    model = LLAMAModel(config)