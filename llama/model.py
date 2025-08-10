import torch
import torch.nn as nn
import torch.nn.functional as F

import math, time
from dataclasses import dataclass
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class ModelConfig:
    pass

class LLAMAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(self.vocab_size, self.emb_dim)

    def forward(self, X, targets=None):
        B, T = X.shape

        x = self.tok_emb(x)

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