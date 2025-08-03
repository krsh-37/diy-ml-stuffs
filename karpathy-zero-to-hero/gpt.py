import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import tamil
import codecs
from tamil import utf8
import warnings
warnings.filterwarnings('ignore')

_ = pd.read_json("./data/lyrics_2017.json", lines=True)['பாடல்வரிகள்']
_ = pd.concat(
    [_, 
        pd.read_json("./data/lyrics_2018.json", lines=True)['பாடல்வரிகள்'] ],
    ignore_index=True
)
_ = pd.concat(
    [_, 
        pd.read_json("./data/lyrics_2019.json", lines=True)['பாடல்வரிகள்'] ],
    ignore_index=True
)

text = "\n\n".join(_.to_list())


# create i_to_s and s_to_i mapping dict 
chars = sorted(list(set(text)))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}
len(itos)

## encode - decode
attn_drop = heads_drop = ff_drop = 0
encode = lambda x: [stoi[i] for i in x]
decode = lambda x: ''.join([itos[i] for i in x])

decode( encode(text[:11]) )

# hyperparameters
n_heads = 4
emb_dim = 128
n_block = 6
batch_size = 32
seq_len = 64
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

chars = sorted(list(set(text)))
vocab_size = len(chars)

## data prep
data = torch.tensor(encode(text), dtype=torch.long)

train = data[:int( len(text)*0.9 ) ] 
val = data[int( len(text)*0.9 ): ]

def get_batch_data(mode):
    data = train if mode == 'train' else val
    idx = torch.randint( len(data) - seq_len, (batch_size,) )
    
    x = torch.stack( [ data[i:i+seq_len] for i in idx] )
    y = torch.stack( [ data[i+1:i+seq_len+1] for i in idx] )
    x, y = x.to(device), y.to(device)
    return x,y 

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_data(split)
#            print(X, Y)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size, bias = False):
        super().__init__()
        self.q_proj = nn.Linear(emb_dim, head_size, bias=bias)
        self.k_proj = nn.Linear(emb_dim, head_size, bias=bias)
        self.v_proj = nn.Linear(emb_dim, head_size, bias=bias)
        self.head_size = head_size

        self.attn_drop = nn.Dropout(attn_drop)
        self.register_buffer('trill', torch.tril(torch.ones((seq_len, seq_len))) )
    
    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = ( q @ k.transpose(-2, -1) ) * (self.head_size ** 0.5)
        attn = attn.masked_fill( self.trill[:T, :T] == 0, float("-inf") )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        return out

class MHA(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.heads_interaction = nn.Linear(head_size * n_heads, emb_dim)
        self.heads_drop = nn.Dropout(heads_drop)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.heads_drop( self.heads_interaction(out) )
        return out

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.proj_in = nn.Linear(emb_dim, 4 * emb_dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(4 * emb_dim, emb_dim)
        self.ff_drop = nn.Dropout(ff_drop)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.act(x)
        x = self.proj_out(x)
        return self.ff_drop(x)

class Block(nn.Module):
    def __init__(self, emb_dim, head_size):
        super().__init__()
        head_size = emb_dim // n_heads
        self.sa = MHA(n_heads, head_size)
        self.ff = FeedForward(emb_dim)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class CausalGPT(nn.Module):
    def __init__(self,):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(seq_len, emb_dim)
        self.blocks = nn.Sequential(* [Block(emb_dim, n_heads) for _ in range(n_block) ] )
        self.out_norm = nn.LayerNorm(emb_dim) # final layer norm
        self.out_proj = nn.Linear(emb_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb( torch.arange(T, device=device) )
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.out_norm(x)
        logits = self.out_proj(x)

        loss = None
        if not targets is None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # We are not modifying in forward
            probs = F.softmax(logits, dim=-1)
            idx_new = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_new), dim = 1)
        return idx

def gen_rand():
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

m = CausalGPT()
x, y = get_batch_data('train')

opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for i in range(max_iters):
    x, y = get_batch_data('train')
    
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        gen_rand()
    
    logits, loss = m(x, y)
    
    opt.zero_grad(set_to_none=True)
    
    loss.backward()
    opt.step()


def save_model(m):
    # Save model
    torch.save({
        'model_state_dict': m.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'emb_dim': emb_dim,
            'n_heads': n_heads,
            'n_block': n_block,
            'seq_len': seq_len
        }
    }, 'causal_gpt_checkpoint.pt')

save_model(m)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


def load_chk():
    # Load checkpoint
    checkpoint = torch.load('causal_gpt_checkpoint.pt', map_location=device)

    # Recreate the model with saved config
    config = checkpoint['config']
    m = CausalGPT()  # assumes global vars like vocab_size etc. are set
    m.load_state_dict(checkpoint['model_state_dict'])
    m.to(device)
    m.eval()  # Set to eval mode if you're doing inference

    # (Optional) Load optimizer if resuming training
    opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
load_chk()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))