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
    n_layers = 6
    n_heads = 8
    n_kv_heads = None
    vocab_size = 64 # set later
    eps = 1e-6

class GQA(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x

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
        hddn_dim = config.muliple_of * (( hddn_dim + config.muliple_of -1 ) // config.muliple_of )
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
    
    def forward(self, x):
        """
        receives (B T C)
        returns (B T C)
        """
        ## refer diagram for the logic
        x = x + self.attn( self.attn_norm(x) )
        x = x + self.ff_norm( self.ff(x) )
        return x

class LLAMAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.Sequential(* [Block(config=config) for _ in range(config.n_layers) ] )
        self.out_norm = RMSNorm(config)
        self.out_proj = nn.Linear( config.d_model , config.vocab_size )

    def forward(self, X, targets=None):
        """
        B: Batch; T: time_step/ seq_len; C: Channel/ emb_dim/ d_model
        """
        B, T = X.shape

        x = self.tok_emb(X)         # (B T C)
        x = self.layers(x)          # (B T C)
        x = self.out_norm(x)        # (B T C)
        logits = self.out_proj(x)   # (B T vc)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)    # (B', C)
            targets = targets.view(B * T)   # (B', 1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

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