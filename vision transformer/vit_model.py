import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class PositionalEmbedding(nn.Module):
    """
    receives (B n_patches emb_dim)
    1. create pos emb learnable param based on pooling type. 
        1.1. initialize cls token if pooling type is cls. 
        1.2 for cls we need to add extra 1 more pos. else n_patches
    2. add pos emb to x and return
    """
    def __init__(self, emb_dim = 768, n_patches = 196, pooling_type = "cls"):
        super().__init__()
        breakpoint()
        n_tokens = (n_patches+1) if pooling_type == "cls" else n_patches
        self.pooling_type = pooling_type

        if pooling_type == "cls":
            self.cls_token = nn.Parameter( torch.zeros( 1, 1, emb_dim ) )         # (B 1 emb_dim)
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_emb = nn.Parameter( torch.zeros(1, n_tokens, emb_dim) )          # (B T emb_dim)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
    
    def forward(self, x):
        if self.pooling_type == "cls":
            # expand for all the batches, since broadcasting wont happen across batch in torch.cat
            cls_token = self.cls_token.expand( x.shape[0], -1, -1 )
            x = torch.cat((cls_token, x), dim=1 )
        return x + self.pos_emb
    
class PatchEmbedding(nn.Module):
    """
    receives (B C H W)
    - conv (C H W) input into n_patches of emb_dim
    """
    def __init__(self, emb_dim=768, patch_size=16,
                 in_channels=3, img_size=224, bias = True):
        super().__init__()
        self.emb_dim, self.patch_size, self.in_channels, self.img_size = (emb_dim, patch_size,
                 in_channels, img_size)
        self.n_patches = ( img_size // patch_size ) ** 2
        self.proj = nn.Conv2d( in_channels=in_channels, out_channels=emb_dim,
                                  kernel_size=patch_size, stride = patch_size, bias=bias )
    
    def forward(self, x):
        x = self.proj(x)                # (B C H W ) -> (B, emd_dim, sqrt(n_patches), sqrt(n_patches))
        x = x.flatten(2)                # abve to (B emd_dim n_patches)
        x = x.transpose(1, 2)           # to (B n_patches emb_dim)
        return x

class FeedForward(nn.Module):
    """
    input: (B T C); regular ff
    """
    def __init__(self, emb_dim=768, mlp_ratio = 4, ff_drop = 0):
        super().__init__()
        self.proj_in = nn.Linear(emb_dim, mlp_ratio * emb_dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(mlp_ratio * emb_dim, emb_dim)
        self.ff_drop = nn.Dropout(ff_drop)

    def forward(self, x):
        x = self.act( self.proj_in(x) )
        x = self.ff_drop(x)
        x = self.proj_out(x)
        return self.ff_drop(x)

class Block(nn.Module):
    def __init__(self, emb_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim)
    
    def forward(self, x):
        norm_x = self.attn_norm(x)
        x = x + self.attn( norm_x, norm_x, norm_x )[0]
        x = x + self.ff( self.ff_norm(x) )
        return x

class VisionTransformer(nn.Module):
    def __init__(self, emb_dim=768,
                 patch_size=16, in_channels=3, img_size=224, 
                 pooling_type="cls", n_blocks=12, n_classes=30
                 ):
        super().__init__()
        self.n_classes = n_classes
        self.pooling_type = pooling_type
        self.patch_emb = PatchEmbedding(
                 emb_dim, patch_size,
                 in_channels, img_size
                 )
        self.apply_pos_emb = PositionalEmbedding(emb_dim, self.patch_emb.n_patches, self.pooling_type)
        self.out_norm = nn.LayerNorm(emb_dim)
        self.out_proj = nn.Linear(emb_dim, self.n_classes)
        self.blocks = nn.Sequential(* [ Block() for _ in range(n_blocks) ] )
    
    def forward(self, x, targets = None):
        B = x.shape[0]
        x = self.apply_pos_emb( self.patch_emb(x) )            # patch emb and then add pos emb
        x = self.blocks(x)
        x = self.out_norm(x)

        if self.pooling_type == "cls":
            agg_score = x[:, 0]                         # Take the cls token for out_proj
        elif self.pooling_type == "avg":
            agg_score = x.mean(dim=1)                   # take avg across time step dim

        logits = self.out_proj(agg_score)
        loss = None
        if targets is not None:
            pass
        
        return logits, loss

if __name__ == "__main__":
    model = VisionTransformer()
    x = torch.randn( size=(2, 3, 224, 224) ) # (B, image channels, height, width)
    model(x)