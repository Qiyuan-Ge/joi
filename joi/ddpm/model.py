import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
        
    return module
    
    
class TimestepEmbedding(nn.Module):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param in_dim: the dimension of the input.
    :param out_dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    def __init__(self, in_dim, out_dim, max_period=10000):
        super().__init__()
        assert in_dim % 2 == 0
        half = in_dim // 2
        self.freq = torch.exp(-math.log(max_period) * torch.arange(half) / half).unsqueeze(0)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, timesteps):
        args = timesteps[:, None] * self.freq.type_as(timesteps)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embedding = self.proj(embedding)
        
        return embedding


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """_summary_

        Args:
            x: input
            emb: timestep embedding

        """        


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        
    def forward(self, qkv):
        B, D, L = qkv.shape
        q, k, v = qkv.chunk(3, dim=1)
        ch = D // (3 * self.n_heads)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", 
            (q*scale).reshape(B * self.n_heads, ch, L), 
            (k*scale).reshape(B * self.n_heads, ch, L))
        weight = F.softmax(weight, dim=-1)
        z = torch.einsum("bts,bcs->bct", weight, v.reshape(B * self.n_heads, ch, L))
        
        return z.reshape(B, -1, L)
        

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.positional_embedding = nn.Parameter(torch.randn(embed_dim, spacial_dim ** 2 + 1) / math.sqrt(embed_dim))
        self.qkv_proj = nn.Conv1d(embed_dim, 3 * embed_dim, 1)
        self.attention = QKVAttention(num_heads)
        self.proj = nn.Conv1d(embed_dim, output_dim or embed_dim, 1)
        
    def forward(self, x):
        B, C, *_ = x.shape
        x = x.reshape(B, C, -1) # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1) # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype) # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.proj(x)
        
        return x[:, :, 0]
    
    
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
    

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, updown=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1)
        )
        self.updown = updown
        if updown == 'up':
            self.h_upd = Upsample(channels)
            self.x_upd = Upsample(channels)
        elif updown == 'down':
            self.h_upd = Downsample(channels)
            self.x_upd = Downsample(channels)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()
            
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)
            
    def forward(self, x, emb):
        if self.updown is not None:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            h = in_conv(h)
            x = self.x_upd(x)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h
        
    
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))  
         
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)
        qkv = self.qkv(self.group_norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        
        return (x + h).reshape(B, C, H, W)
            
    
class UNet(nn.Module):
    def __init__(
        self, 
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 3, 4),
        num_classes=None,
        num_heads=4,
        ):
        super().__init__()
        self.model_channels = model_channels
        self.num_classes = num_classes
        time_embed_dim = model_channels * 4
        self.timestep_embedding = TimestepEmbedding(model_channels, time_embed_dim)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = 0
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=int(mult * model_channels))
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, updown='down')))
                ch = out_ch
                input_block_chans.append(ch)
                ds += 1
                
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads),
            ResBlock(ch, time_embed_dim, dropout),
            )
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=int(model_channels * mult))]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, updown='up'))
                    ds -= 1
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1))
        )
        
    def forward(self, x, timesteps, y=None):
        """

        Args:
            x: (B, C, H, W)
            timesteps: 1-D batch of timesteps
            y: (B,) labels, if class-conditional

        Returns:
            output: (B, C, H, W)
        """        
        hs = []
        emb = self.timestep_embedding(timesteps)
        
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            
        return self.out(h)
    
