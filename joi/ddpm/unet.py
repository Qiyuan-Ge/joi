import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from abc import abstractmethod
from .t5 import create_encoder, create_mask


def exists(x):
    return x is not None


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimeEmbedding(nn.Module):
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


class TextEmbedding(nn.Module):
    def __init__(self, dim, model_name='t5-base', pretrained=True):
        super().__init__()
        d_model = {'t5-small':512, 't5-base':768}
        self.t5 = create_encoder(model_name, pretrained)
        self.proj = nn.Linear(d_model[model_name], dim)
        
    def forward(self, token_ids):
        self.t5.eval()
        with torch.no_grad():
            output = self.t5(input_ids=token_ids, attention_mask=create_mask(token_ids))
            encoded_text = output.last_hidden_state.detach()
            encoded_text = encoded_text[token_ids==1] # EOS_id = 1
        
        return self.proj(encoded_text)
        

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        
        return self.conv(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
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
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        # use scale_shift_norm
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
            
        return self.skip_connection(x) + h
    
    
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        num_head_channels = channels // num_heads
        self.scale = math.sqrt(num_head_channels)
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1)
        
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

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param condition: unconditinal, class-conditional or text-conditional.
    :param num_classes: must be speciefied, if this model is class-conditional.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels=None,
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        condition=None,
        text_model_name='t5-base',
        text_model_pretrained=True,
        num_classes=None,
        num_heads=4,
        num_heads_upsample=-1,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels or in_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.condition = condition
        self.text_model_name = text_model_name
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.timestep_embedding = TimeEmbedding(model_channels)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if exists(condition):
            if condition == 'class':
                self.cond_emb = nn.Embedding(num_classes, time_embed_dim)
            elif condition == 'text':
                self.cond_emb = TextEmbedding(time_embed_dim, text_model_name, text_model_pretrained)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, 
                             time_embed_dim, 
                             dropout, 
                             out_channels=mult * model_channels)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, time_embed_dim, dropout),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads_upsample)
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, self.out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(self.timestep_embedding(timesteps))

        if exists(y):
            emb = emb + self.cond_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        
        return self.out(h)

    
class SuperResUnet(Unet):
    """
    p(x_{t-1}|x_t, z_0)
    Unet model performs super-resolution.
    condition on a low resolution image z_0.
    """    
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)
        
    def forward(self, x, timesteps, low_res, y=None):
        _, _, new_h, new_w = x.shape
        upsampled = F.interpolate(low_res, (new_h, new_w), mode="bilinear")
        x = torch.cat([x, upsampled], dim=1)
        
        return super().forward(x, timesteps, y)
    
    
class InpaintUnet(Unet):
    """
    A Unet which can perform inpainting.
    condition on a inpaint_image and inpaint_mask.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2 + 1, *args, **kwargs)

    def forward(self, x, timesteps, inpaint_image, inpaint_mask, y=None):
        x = torch.cat([x, inpaint_image * inpaint_mask, inpaint_mask], dim=1)
    
        return super().forward(x, timesteps, y)
           
