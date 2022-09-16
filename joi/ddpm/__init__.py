from .model import Unet
from .gaussian_diffusion import GaussianDiffusion
from .train_util import DiffusionTrainer

__all__ = ['create_model', 'create_gaussian_diffusion', 'create_model_and_diffusion', 'DiffusionTrainer']

def create_model(img_size=64, in_channels=3, out_channels=None, num_classes=None):
    if img_size <= 32:
        return Unet(dim=64,
                    in_channels=in_channels,
                    init_dim=32,
                    out_channels=out_channels,
                    dim_mults=(1, 2, 3, 4),
                    n_heads=4, 
                    dim_head=32,
                    convnext_mult=2,
                    num_classes=num_classes,
                    )
    elif img_size <= 64:
        return Unet(dim=128,
                    in_channels=in_channels,
                    init_dim=96,
                    out_channels=out_channels,
                    dim_mults=(1, 2, 3, 4),
                    n_heads=4, 
                    dim_head=32,
                    convnext_mult=2,
                    num_classes=num_classes,
                    )
    elif img_size <= 128:
        return Unet(dim=128,
                    in_channels=in_channels,
                    init_dim=96,
                    out_channels=out_channels,
                    dim_mults=(1, 2, 3, 4),
                    n_heads=4, 
                    dim_head=32,
                    convnext_mult=2,
                    num_classes=num_classes,
                    )
    elif img_size <= 256:
        return Unet(dim=128,
                    in_channels=in_channels,
                    init_dim=128,
                    out_channels=out_channels,
                    dim_mults=(1, 1, 2, 3, 4),
                    n_heads=4, 
                    dim_head=32,
                    convnext_mult=2,
                    num_classes=num_classes,
                    )
    elif img_size <= 512:
        return Unet(dim=128,
                    in_channels=in_channels,
                    init_dim=128,
                    out_channels=out_channels,
                    dim_mults=(1, 1, 2, 3, 4),
                    n_heads=4, 
                    dim_head=32,
                    convnext_mult=2,
                    num_classes=num_classes,
                    )
    else:
        raise ValueError(f"unsupported image size: {img_size}")
        
          
def create_gaussian_diffusion(model, timesteps=1000, beta_schedule='cosine', loss_type="l2"):
    diffusion = GaussianDiffusion(model, timesteps, beta_schedule, loss_type)
    
    return diffusion


def create_model_and_diffusion(img_size=64, in_channels=3, out_channels=None, timesteps=1000, beta_schedule='cosine', loss_type="l2", num_classes=None):
    model = create_model(img_size, in_channels, out_channels, num_classes)
    diffusion = create_gaussian_diffusion(model, timesteps, beta_schedule, loss_type)
    
    return model, diffusion
