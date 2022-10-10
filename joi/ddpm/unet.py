import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from abc import abstractmethod


def exists(x):
    return x is not None


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time): 
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings
      

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, cond=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, cond=None, text_mask=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, cond, text_mask)
            else:
                x = layer(x)
                
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.dim
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.dim
        
        return self.conv(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param dim:     the number of input channels.
    :param time_cond_dim: the number of timestep embedding channels.
    :param dropout:      the rate of dropout.
    :param out_dim: if specified, the number of out channels.
    """

    def __init__(
        self,
        dim,
        time_cond_dim,
        dropout,
        out_dim=None,
        use_cross_attention=True
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim
        self.use_cross_atten = use_cross_attention
        
        self.layer_1 = nn.Sequential(
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv2d(dim, self.out_dim, 3, padding=1),
        )
        
        if use_cross_attention:
            self.cross_atten = CrossAttention(self.out_dim, time_cond_dim)
        
        self.layer_t = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_cond_dim, 2 * self.out_dim),
        )
        
        self.layer_2 = nn.Sequential(
            nn.GroupNorm(32, self.out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1)),
        )

        if self.out_dim == dim:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(dim, self.out_dim, 3, padding=1)

    def forward(self, x, emb, cond=None, text_mask=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.layer_1(x)
        if self.use_cross_atten:
            H, W = h.shape[-2:]
            h = rearrange(h, 'b c h w -> b (h w) c')
            h = self.cross_atten(h, cond, text_mask) + h
            h = rearrange(h, 'b (h w) c -> b c h w', h=H, w=W)
        
        emb_out = self.layer_t(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        # use scale_shift_norm
        out_norm, out_rest = self.layer_2[0], self.layer_2[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
            
        return self.skip_connection(x) + h


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, num_heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = math.sqrt(dim_head)
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.proj = nn.Linear(inner_dim, dim, bias=False)
        
    def forward(self, x, context, mask=None):
        x = self.norm(x)
        context = self.norm_context(context)
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads) for x in (q, k, v)]
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) / self.scale
        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> b 1 1 j')
            sim.masked_fill_(~mask, max_neg_value)
        attn_p = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn_p, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.proj(out)
    
    
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        dim_head = dim // num_heads
        self.scale = math.sqrt(dim_head)
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x):
        _, _, h, w = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = [rearrange(x, "b (h c) x y -> b h c (x y)", h=self.num_heads) for x in (q, k, v)]
        scores = torch.einsum("b h c i, b h c j -> b h i j", q, k) / self.scale
        attn_p = F.softmax(scores, dim=-1)
            
        out = torch.einsum("b h i j, b h c j-> b h i c", attn_p, v)
        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=h, y=w)
        out = self.to_out(out)
            
        return x + out


class Unet(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_dim:           channels in the input Tensor.
    :param d_model:          base channel count for the model.
    :param out_dim:          channels in the output Tensor.
    :param num_res_blocks:   number of residual blocks per downsample.
    :param layer_attention:  a collection of downsample rates at which attention will take place.
    :param dropout:          the dropout probability.
    :param dim_mult:         channel multiplier for each level of the UNet.
    :param condition:        unconditinal, class-conditional or text-conditional.
    :param num_classes:      must be speciefied, if this model is class-conditional.
    :param num_heads:        the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_dim,
        d_model,
        out_dim=None,
        num_res_blocks=2,
        dim_mult=(1, 2, 4, 8),
        layer_attention=(False, False, False, True),
        layer_cross_attention=(False, False, False, True),
        dropout=0,
        condition=None,
        text_model_name='t5-base',
        num_classes=None,
        num_heads=4,
        num_heads_upsample=-1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if condition in [None, 'class', 'text']:
            self.condition = condition
        else:
            raise ValueError(f'unknown objective {condition}. condition must be None, class or text')
        self.timestep_embedding = SinusoidalPositionEmbeddings(d_model)

        time_cond_dim = d_model * 4
        self.time_emb = nn.Sequential(
            nn.Linear(d_model, time_cond_dim),
            nn.SiLU(),
            nn.Linear(time_cond_dim, time_cond_dim),
        )

        if exists(condition):
            if condition == 'class':
                self.cond_emb = nn.Embedding(num_classes, time_cond_dim)
            elif condition == 'text':
                dim_t5 = {'t5-small':512, 't5-base':768}
                cond_text_dim = dim_t5[text_model_name]
                self.cond_emb = nn.Sequential(
                    nn.LayerNorm(cond_text_dim),
                    nn.Linear(cond_text_dim, time_cond_dim),
                    nn.SiLU(),
                    nn.Linear(time_cond_dim, time_cond_dim),
                )
            else:
                raise ValueError(f'unknown objective {condition}')

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(nn.Conv2d(in_dim, d_model, 3, padding=1))
            ]
        )
        input_block_chans = [d_model]
        ch = d_model
        for level, mult in enumerate(dim_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, 
                             time_cond_dim, 
                             dropout, 
                             out_dim=mult * d_model, 
                             use_cross_attention=layer_cross_attention[level],   
                    )
                ]
                ch = mult * d_model
                if layer_attention[level]:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(dim_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch))
                )
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_cond_dim, dropout, use_cross_attention=True),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, time_cond_dim, dropout, use_cross_attention=True),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(dim_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_cond_dim,
                        dropout,
                        out_dim=d_model * mult,
                        use_cross_attention=layer_cross_attention[level],
                    )
                ]
                ch = d_model * mult
                if layer_attention[level]:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads_upsample)
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(d_model, self.out_dim, 3, padding=1)),
        )

    def forward(self, x, timesteps, cond=None, text_mask=None):
        """
        Apply the model to an input batch.

        :param x:         an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cond:      an [N] Tensor of labels, if class-conditional. an [N x D] Tensor of texts, if text-conditional.
        :return:          an [N x C x ...] Tensor of outputs.
        """

        hs = []
        time = self.time_emb(self.timestep_embedding(timesteps))

        if exists(cond):
            cond = self.cond_emb(cond)
            time = time + cond
            if len(cond.shape) == 2:
                cond = cond.unsqueeze(1)

        h = x
        for module in self.input_blocks:
            h = module(h, time, cond, text_mask)
            hs.append(h)
        h = self.middle_block(h, time, cond, text_mask)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, time, cond, text_mask)
        
        return self.out(h)

    
class SuperResUnet(Unet):
    """
    p(x_{t-1}|x_t, z_0)
    Unet model performs super-resolution.
    condition on a low resolution image z_0.
    """    
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)
        
    def forward(self, x, timesteps, low_res, cond=None):
        _, _, new_h, new_w = x.shape
        upsampled = F.interpolate(low_res, (new_h, new_w), mode="bilinear")
        x = torch.cat([x, upsampled], dim=1)
        
        return super().forward(x, timesteps, cond)
    
    
class InpaintUnet(Unet):
    """
    A Unet which can perform inpainting.
    condition on a inpaint_image and inpaint_mask.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2 + 1, *args, **kwargs)

    def forward(self, x, timesteps, inpaint_image, inpaint_mask, cond=None):
        x = torch.cat([x, inpaint_image * inpaint_mask, inpaint_mask], dim=1)
    
        return super().forward(x, timesteps, cond)
           
