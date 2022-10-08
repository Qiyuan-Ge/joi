import os
import torch
from torchvision.utils import save_image
import joi.ddpm as ddpm
from .t5 import encode_text


class Painter:
    def __init__(self, model, task='text2image', text_model_name='t5-small', timesteps=1000, beta_schedule='cosine', device='cuda', img_size=64):
        self.device = device
        self.img_size = img_size
        self.num_channel = model.in_channels
        self.diffusion = ddpm.create_gaussian_diffusion(model, timesteps, beta_schedule)
        self.diffusion.to(device)
        self.task = task
        self.text_model_name = text_model_name
    
    def __call__(self, condition=None, num_samples=1, saved_path=None):
        return self.paint(condition, num_samples, saved_path)
    
    def paint(self, condition, num_samples, saved_path):
        condition = [condition] * num_samples
        if self.task == 'text2image':
            condition = encode_text(condition, name=self.text_model_name)
        elif self.task == 'class2image':
            condition = torch.tensor(condition).long()
        else:
            RaiseValueError(f'unknown object {self.task}')
        imgs = self.diffusion.sample(self.img_size, num_samples, self.num_channel, condition.to(self.device))[-1]
        imgs = (imgs.clamp(-1, 1) + 1) * 0.5
        n_row = int(num_samples ** 0.5)
        save_image(imgs, saved_path, nrow=n_row)
        
