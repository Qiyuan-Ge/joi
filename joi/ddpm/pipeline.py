
class Painter:
    def __init__(self, model, timesteps=1000, beta_schedule='cosine', device='cuda', image_size=64, num_channel=3):
        self.img_size = image_size
        self.num_channel = num_channel
        self.diffusion = joi.ddpm.create_gaussian_diffusion(model, timesteps, beta_schedule)
        self.diffusion.to(device)
    
    def __call__(self, condition=None, num_samples=1):
        return self.paint(condition, num_samples)
    
    def paint(self, condition=None, num_samples=1):
        imgs = self.diffusion.sample(self.img_size, num_samples, self.num_channel, condition)[-1]
        