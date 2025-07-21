import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
#%matplotlib inline

# !pip install open-tamil
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
block_size = 8
encode = lambda x: [stoi[i] for i in x]
decode = lambda x: ''.join([itos[i] for i in x])

decode( encode(text[:11]) )

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 30000
eval_interval = 30000
learning_rate = 1e-2
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
    idx = torch.randint( len(data) - block_size, (batch_size,) )
    
    x = torch.stack( [ data[i:i+block_size] for i in idx] )
    y = torch.stack( [ data[i+1:i+block_size+1] for i in idx] )
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

class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits, loss = self.emb_table(idx), None

        if not targets is None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] # We are not modifying in forward
            probs = F.softmax(logits, dim=-1)
            idx_new = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_new), dim = 1)
        return idx
    
m = Bigram(vocab_size)
x, y = get_batch_data('train')

opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for i in range(max_iters):
    x, y = get_batch_data('train')
    
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    logits, loss = m(x, y)
    
    opt.zero_grad(set_to_none=True)
    
    loss.backward()
    opt.step


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))