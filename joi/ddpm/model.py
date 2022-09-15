import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def exists(x):
    return x is not None


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
    
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        
        return x
    
    
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv(x)
        
        return x
    
    
class TimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = -1 * math.log(10000) / (half_dim - 1)
        self.embeddings = torch.exp(torch.arange(half_dim) * emb)

    def forward(self, time):
        embeddings = self.embeddings.to(time.device)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
            
        return embeddings
        
        
class ConvNextBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim=None, mult=2):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, in_dim)) if exists(time_emb_dim) else None
        )
        
        self.ds_conv = nn.Conv2d(in_dim, in_dim, 7, padding=3, groups=in_dim)
        
        self.net = nn.Sequential(
            nn.GroupNorm(1, in_dim),
            nn.Conv2d(in_dim, out_dim * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_dim * mult),
            nn.Conv2d(out_dim * mult, out_dim, 3, padding=1),
        )
        
        self.res_conv = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
            
        h = self.net(h)
        
        return h + self.res_conv(x)
    
    
class Attention(nn.Module):
    def __init__(self, in_dim, n_heads=4, dim_head=32):
        super().__init__()
        self.scale = math.sqrt(dim_head)
        self.n_heads = n_heads
        hidden_dim = n_heads * dim_head
        self.to_qkv = nn.Conv2d(in_dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, in_dim, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [rearrange(x, "b (h c) x y -> b h c (x y)", h=self.n_heads) for x in (q, k, v)]
            
        scores = torch.einsum("b h c i, b h c j -> b h i j", q, k) / self.scale
        attn_p = F.softmax(scores, dim=-1)
            
        out = torch.einsum("b h i j, b h c j-> b h i c", attn_p, v)
        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=h, y=w)
            
        return self.to_out(out)
            
            
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
            
    def forward(self, x):
        x = self.norm(x)
            
        return self.fn(x)
        
        
class Unet(nn.Module):
    def __init__(
        self,
        dim=64,
        in_channels=3,
        init_dim=96,
        out_channels=None,
        dim_mults=(1, 2, 4, 8),
        n_heads=4, 
        dim_head=32,
        convnext_mult=2,
        ):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, init_dim, 7, padding=3)
            
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
            
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            )
            
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
            
        for num, (in_dim, out_dim) in enumerate(in_out):
            is_last = num == (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(in_dim, out_dim, time_emb_dim=time_dim, mult=convnext_mult),
                        ConvNextBlock(out_dim, out_dim, time_emb_dim=time_dim, mult=convnext_mult),
                        Residual(PreNorm(out_dim, Attention(out_dim, n_heads, dim_head))),
                        Downsample(out_dim) if not is_last else nn.Identity(),
                    ]
                )
            )
            
        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim, mult=convnext_mult)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim, mult=convnext_mult)
        
        for num, (in_dim, out_dim) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(out_dim * 2, in_dim, time_emb_dim=time_dim, mult=convnext_mult),
                        ConvNextBlock(in_dim, in_dim, time_emb_dim=time_dim, mult=convnext_mult),
                        Residual(PreNorm(in_dim, Attention(in_dim))),
                        Upsample(in_dim),
                    ]
                )
            )
            
        out_dim = out_channels or in_channels
        self.last_conv = nn.Sequential(
            ConvNextBlock(dim, dim, mult=convnext_mult),
            nn.Conv2d(dim, out_dim, 1),
        )
        
    def forward(self, x, time):
        x = self.init_conv(x)  
        t = self.time_mlp(time) 
        h = []
            
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            
        return self.last_conv(x)
                