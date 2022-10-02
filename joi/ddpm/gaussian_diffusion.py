import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return torch.clip(betas, 0.0001, 0.9999)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_schedule='cosine', loss_type="l2"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
            
        # define beta schedule
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
            
        # define alphas 
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        
        # calculations for q(x_t|x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        
        # calculations for posterior q(x_{t-1}|x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    # q(x_t|x_0)        
    def q_sample(self, x_start, t, noise=None): 
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def loss_func(self, x, y):
        if self.loss_type == 'l1':
            loss = F.l1_loss(x, y)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x, y)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x, y)
        else:
            NotImplementedError()
            
        return loss
    
    # L_t^simple    
    def p_losses(self, x_start, t, noise=None, y=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, y)

        return self.loss_func(noise, predicted_noise)
    
    def forward(self, x_start, t, noise=None, y=None):
        return self.p_losses(x_start, t, noise, y)
    
    # q(x_{t-1}|x_t, x_0)
    @torch.no_grad()
    def p_sample(self, x, t, t_index, y=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t, y) / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    # x_t -> x_{t-1} -> ... -> x_0
    @torch.no_grad()
    def p_sample_loop(self, shape, y):
        b = shape[0]
        device = next(self.model.parameters()).device
        # start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i, y)
            imgs.append(img.cpu())
            
        return imgs
    
    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3, labels=None):
        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size), y=labels)
