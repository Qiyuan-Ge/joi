from .model import Unet
from .gaussian_diffusion import GaussianDiffusion
from .train_util import DDPM_Trainer

__all__ = ['create_model', 'create_gaussian_diffusion', 'create_model_and_diffusion', 'DDPM_Trainer']

def create_model(img_size=64, in_channels=3, out_channels=None):
    if img_size <= 32:
        return Unet(dim=64,
                    in_channels=in_channels,
                    init_dim=96,
                    out_channels=out_channels,
                    dim_mults=(1, 2, 4, 8),
                    n_heads=4, 
                    dim_head=32,
                    convnext_mult=2)
    elif img_size <= 64:
        return Unet(dim=64,
                    in_channels=in_channels,
                    init_dim=96,
                    out_channels=out_channels,
                    dim_mults=(1, 2, 4, 8),
                    n_heads=4, 
                    dim_head=32,
                    convnext_mult=2)
    elif img_size <= 128:
        return Unet(dim=64,
                    in_channels=in_channels,
                    init_dim=96,
                    out_channels=out_channels,
                    dim_mults=(1, 2, 4, 8),
                    n_heads=4, 
                    dim_head=32,
                    convnext_mult=2)
    elif img_size <= 256:
        return Unet(dim=128,
                    in_channels=in_channels,
                    init_dim=128,
                    out_channels=out_channels,
                    dim_mults=(1, 2, 4, 8),
                    n_heads=4, 
                    dim_head=32,
                    convnext_mult=2)
    elif img_size <= 512:
        return Unet(dim=128,
                    in_channels=in_channels,
                    init_dim=128,
                    out_channels=out_channels,
                    dim_mults=(1, 2, 4, 8),
                    n_heads=4, 
                    dim_head=32,
                    convnext_mult=2)
    else:
        raise ValueError(f"unsupported image size: {img_size}")
        
          
def create_gaussian_diffusion(model, timesteps=1000, beta_schedule='cosine', loss_type="l2"):
    diffusion = GaussianDiffusion(model, timesteps, beta_schedule, loss_type)
    
    return diffusion


def create_model_and_diffusion(img_size=64, in_channels=3, out_channels=None, timesteps=1000, beta_schedule='cosine', loss_type="l2"):
    model = create_model(img_size, in_channels, out_channels)
    diffusion = create_gaussian_diffusion(model, timesteps, beta_schedule, loss_type)
    
    return model, diffusion
