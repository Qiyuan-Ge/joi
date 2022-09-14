from .model import UNet
from .gaussian_diffusion import GaussianDiffusion
from .train_util import DDPM_Trainer

__all__ = ['create_model', 'create_gaussian_diffusion', 'create_model_and_diffusion', 'DDPM_Trainer']

def create_model(img_size=64, num_channels=3, num_classes=None, dropout=0):
    if img_size <= 32:
        return UNet(in_channels=num_channels, 
                    model_channels=128, 
                    out_channels=num_channels, 
                    num_res_blocks=2, 
                    attention_resolutions=(2,), 
                    dropout=dropout, 
                    channel_mult=(1, 2, 4), 
                    num_classes=num_classes, 
                    num_heads=2)
    elif img_size <= 64:
        return UNet(in_channels=num_channels, 
                    model_channels=194, 
                    out_channels=num_channels, 
                    num_res_blocks=2, 
                    attention_resolutions=(3,), 
                    dropout=dropout, 
                    channel_mult=(1, 2, 3, 4), 
                    num_classes=num_classes, 
                    num_heads=4)
    elif img_size <= 128:
        return UNet(in_channels=num_channels, 
                    model_channels=256, 
                    out_channels=num_channels, 
                    num_res_blocks=2, 
                    attention_resolutions=(3, 4), 
                    dropout=dropout, 
                    channel_mult=(1, 1, 2, 3, 4), 
                    num_classes=num_classes, 
                    num_heads=4)
    elif img_size <= 256:
        return UNet(in_channels=num_channels, 
                    model_channels=256, 
                    out_channels=num_channels, 
                    num_res_blocks=2, 
                    attention_resolutions=(3, 4, 5), 
                    dropout=dropout, 
                    channel_mult=(1, 1, 2, 2, 4, 4), 
                    num_classes=num_classes, 
                    num_heads=4)
    elif img_size <= 512:
        return UNet(in_channels=num_channels, 
                    model_channels=256, 
                    out_channels=num_channels, 
                    num_res_blocks=2, 
                    attention_resolutions=(4, 5, 6), 
                    dropout=dropout, 
                    channel_mult=(0.5, 1, 1, 2, 2, 4, 4), 
                    num_classes=num_classes, 
                    num_heads=4)
    else:
        raise ValueError(f"unsupported image size: {img_size}")
        
          
def create_gaussian_diffusion(model, timesteps=1000, beta_schedule='cosine', loss_type="l2"):
    diffusion = GaussianDiffusion(model, timesteps, beta_schedule, loss_type)
    
    return diffusion


def create_model_and_diffusion(img_size=64, num_channels=3, num_classes=None, dropout=0, timesteps=1000, beta_schedule='cosine', loss_type="l2"):
    model = create_model(img_size, num_channels, num_classes, dropout)
    diffusion = create_gaussian_diffusion(model, timesteps, beta_schedule, loss_type)
    
    return model, diffusion