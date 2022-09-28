from .model import Unet, SuperResModel
from .gaussian_diffusion import GaussianDiffusion
from .train_util import DiffusionTrainer

__all__ = ['create_model', 'create_gaussian_diffusion', 'create_model_and_diffusion', 'DiffusionTrainer']

def create_model(img_size=64, in_channels=3, num_res_blocks=2, out_channels=None, num_classes=None, dropout=0):
    if img_size == 32:
        return Unet(in_channels,
                    model_channels=128,
                    out_channels=out_channels,
                    num_res_blocks=num_res_blocks,
                    attention_resolutions=(32, 16, 8),
                    dropout=dropout,
                    channel_mult=(1, 2, 2, 2),
                    num_classes=num_classes,
                    num_heads=4,
                    num_heads_upsample=-1,
                    )
    elif img_size == 64:
        return Unet(in_channels,
                    model_channels=128,
                    out_channels=out_channels,
                    num_res_blocks=num_res_blocks,
                    attention_resolutions=(32, 16, 8),
                    dropout=dropout,
                    channel_mult=(1, 2, 3, 4),
                    num_classes=num_classes,
                    num_heads=8,
                    num_heads_upsample=-1,
                    )
    elif img_size == 128:
        return Unet(in_channels,
                    model_channels=128,
                    out_channels=out_channels,
                    num_res_blocks=num_res_blocks,
                    attention_resolutions=(32, 16, 8),
                    dropout=dropout,
                    channel_mult=(1, 1, 2, 3, 4),
                    num_classes=num_classes,
                    num_heads=8,
                    num_heads_upsample=-1,
                    )
    elif img_size == 256:
        return Unet(in_channels,
                    model_channels=256,
                    out_channels=out_channels,
                    num_res_blocks=num_res_blocks,
                    attention_resolutions=(32, 16, 8),
                    dropout=dropout,
                    channel_mult=(1, 1, 2, 2, 4, 4),
                    num_classes=num_classes,
                    num_heads=8,
                    num_heads_upsample=-1,
                    )
    else:
        raise ValueError(f"unsupported image size: {img_size}")
        
          
def create_gaussian_diffusion(model, 
                              timesteps=1000, 
                              beta_schedule='cosine', 
                              loss_type="l2"):
    
    return GaussianDiffusion(model, timesteps, beta_schedule, loss_type)


def create_model_and_diffusion(img_size, 
                               in_channels=3,
                               num_res_blocks=2,
                               out_channels=None, 
                               timesteps=1000, 
                               beta_schedule='cosine', 
                               loss_type="l1", 
                               num_classes=None,
                               dropout=0,
                               ):
    model = create_model(img_size, in_channels, num_res_blocks, out_channels, num_classes, dropout)
    diffusion = create_gaussian_diffusion(model, timesteps, beta_schedule, loss_type)
    
    return model, diffusion


def create_sr_model(resolution="64->256", in_channels=3, num_res_blocks=3, out_channels=None, num_classes=None, dropout=0): 
    if resolution == "64->128":
        return SuperResModel(in_channels,
                             model_channels=128,
                             out_channels=out_channels,
                             num_res_blocks=num_res_blocks,
                             attention_resolutions=(16,),
                             dropout=dropout,
                             channel_mult=(1, 2, 4, 8, 8),
                             num_classes=num_classes,
                             num_heads=8,
                             num_heads_upsample=-1,
                            )
    elif resolution == "64->256":
        return SuperResModel(in_channels,
                             model_channels=128,
                             out_channels=out_channels,
                             num_res_blocks=num_res_blocks,
                             attention_resolutions=(16,),
                             dropout=dropout,
                             channel_mult=(1, 2, 4, 4, 8, 8),
                             num_classes=num_classes,
                             num_heads=8,
                             num_heads_upsample=-1,
                            )
