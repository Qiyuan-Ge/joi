from .unet import Unet, SuperResUnet
from .pipeline import Painter
from .train_util import Trainer
from .gaussian_diffusion import GaussianDiffusion


__all__ = ['create_model', 'create_sr_model', 'create_gaussian_diffusion', 'create_model_and_diffusion', 'Trainer', 'Painter']

def create_model(img_size=64, in_dim=3, num_res_blocks=2, out_dim=None, condition=None, text_model_name='t5-base', num_classes=None, dropout=0):
    if img_size == 32:
        return Unet(in_dim,
                    d_model=128,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dim_mult=(1, 2, 2, 2),
                    layer_attention=(False, False, False, True),
                    layer_cross_attention=(False, False, False, True),
                    dropout=dropout,
                    condition=condition,
                    text_model_name=text_model_name,
                    num_classes=num_classes,
                    num_heads=4,
                    num_heads_upsample=-1,
                    )
    elif img_size == 64:
        return Unet(in_dim,
                    d_model=128,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dim_mult=(1, 2, 3, 4),
                    layer_attention=(False, False, False, True),
                    layer_cross_attention=(False, False, False, True),
                    dropout=dropout,
                    condition=condition,
                    text_model_name=text_model_name,
                    num_classes=num_classes,
                    num_heads=8,
                    num_heads_upsample=-1,
                    )
    elif img_size == 128:
        return Unet(in_dim,
                    d_model=128,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dim_mult=(1, 1, 2, 3, 4),
                    layer_attention=(False, False, False, True, True),
                    layer_cross_attention=(False, False, False, True, True),
                    dropout=dropout,
                    condition=condition,
                    text_model_name=text_model_name,
                    num_classes=num_classes,
                    num_heads=8,
                    num_heads_upsample=-1,
                    )
    elif img_size == 256:
        return Unet(in_dim,
                    d_model=256,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dim_mult=(1, 1, 2, 2, 4, 4),
                    layer_attention=(False, False, False, False, True, True),
                    layer_cross_attention=(False, False, False, False, True, True),
                    dropout=dropout,
                    condition=condition,
                    text_model_name=text_model_name,
                    num_classes=num_classes,
                    num_heads=8,
                    num_heads_upsample=-1,
                    )
    else:
        raise ValueError(f"unsupported image size: {img_size}")
        
          
def create_gaussian_diffusion(model, timesteps=1000, beta_schedule='cosine', loss_type="l2", p2_loss_weight_gamma=1.0, p2_loss_weight_k=1):
    return GaussianDiffusion(model, timesteps, beta_schedule, loss_type, p2_loss_weight_gamma, p2_loss_weight_k)


def create_model_and_diffusion(img_size, in_dim=3, num_res_blocks=2, out_dim=None, timesteps=1000, beta_schedule='cosine', loss_type="l1", condition=None, text_model_name='t5-base', num_classes=None, dropout=0, p2_loss_weight_gamma=1.0, p2_loss_weight_k=1):
    model = create_model(img_size, in_dim, num_res_blocks, out_dim, condition, text_model_name, num_classes, dropout)
    diffusion = create_gaussian_diffusion(model, timesteps, beta_schedule, loss_type, p2_loss_weight_gamma, p2_loss_weight_k)
    
    return model, diffusion


def create_sr_model(resolution="64->256", in_channels=3, num_res_blocks=3, out_channels=None, condition=None, text_model_name='t5-base', num_classes=None, dropout=0): 
    if resolution == "64->128":
        return SuperResUnet(in_channels,
                            model_channels=128,
                            out_channels=out_channels,
                            num_res_blocks=num_res_blocks,
                            channel_mult=(1, 2, 4, 8, 8),
                            layer_attention=(False, False, False, True, True),
                            dropout=dropout,
                            condition=condition,
                            text_model_name=text_model_name,
                            num_classes=num_classes,
                            num_heads=8,
                            num_heads_upsample=-1,
                            )
    elif resolution == "64->256":
        return SuperResUnet(in_channels,
                            model_channels=128,
                            out_channels=out_channels,
                            num_res_blocks=num_res_blocks,
                            channel_mult=(1, 2, 4, 4, 8, 8),
                            layer_attention=(False, False, False, False, True, True),
                            dropout=dropout,
                            condition=condition,
                            text_model_name=text_model_name,
                            num_classes=num_classes,
                            num_heads=8,
                            num_heads_upsample=-1,
                            )
