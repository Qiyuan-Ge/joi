import os
import torch
from torchvision.utils import save_image

class DDPM_Trainer:
    def __init__(self, diffusion, lr, weight_decay, dataloader, sample_interval=None, device=None, result_folder=None):
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = lr
        self.weight_decay = weight_decay
        self.diffusion = diffusion
        self.timesteps = diffusion.timesteps
        self.optimizer = torch.optim.AdamW(self.diffusion.denoise_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.dataloader = dataloader
        self.sample_interval = sample_interval
        self.result_folder = result_folder
        self.diffusion.to(self.device)
        
    def reverse_transform(self, img):
        return (img + 1) * 0.5
    
    def sample_and_save(self, img_size, batch_size, channels, milestone):
        all_images = self.diffusion.sample(img_size, batch_size, channels)
        n_cols = 10
        strides = self.timesteps // n_cols
        all_images = torch.cat(all_images[::strides], dim=0)
        all_images = self.reverse_transform(all_images)
        image_path = os.path.join(self.result_folder, f"sample-{milestone}.png")
        save_image(all_images.data, image_path, nrow=batch_size)
        
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for step, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                batch = batch[0]
                batch_size, ch, img_size, _ = batch.shape
                batch = batch.to(self.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                loss = self.diffusion.p_losses(batch, t)

                loss.backward()
                self.optimizer.step()
                
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                    % (epoch, num_epochs, step, len(self.dataloader), loss)
                    )
    
                # save generated images
                milestone = epoch * len(self.dataloader) + step
                if milestone != 0 and milestone % self.sample_interval == 0:
                    self.sample_and_save(img_size, batch_size=10, channels=ch, milestone=milestone)
                    
        print("Train finished!")
                    

            
            