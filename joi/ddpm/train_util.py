import os
import torch
from torchvision.utils import save_image

class DiffusionTrainer:
    def __init__(self, diffusion, timesteps, lr, weight_decay, dataloader, sample_interval=None, device=None, result_folder=None, num_classes=None):
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = lr
        self.weight_decay = weight_decay
        self.diffusion = diffusion
        self.timesteps = timesteps
        self.optimizer = torch.optim.AdamW(self.diffusion.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.dataloader = dataloader
        self.sample_interval = sample_interval
        self.result_folder = result_folder
        self.num_classes = num_classes
        self.diffusion.to(self.device)
        
    def reverse_transform(self, img):
        return torch.clamp((img + 1) * 0.5, 0, 1)
    
    def sample_and_save(self, img_size, batch_size, channels, milestone):
        if self.num_classes is not None:
            labels = torch.randint(0, self.num_classes, size=(batch_size,))
            all_images = self.diffusion.sample(img_size, batch_size, channels, labels)
        else:
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
                imgs, labels = batch
                batch_size, ch, img_size, img_size = imgs.shape
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                if self.num_classes is not None:
                    loss = self.diffusion.p_losses(imgs, t, y=labels)
                else:
                    loss = self.diffusion.p_losses(imgs, t)
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
                    

            
            