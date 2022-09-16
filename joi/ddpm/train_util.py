import os
import numpy as np
import torch
from torchvision.utils import save_image

def reverse_transform(img):
    return (img + 1) * 0.5

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
    
    def sample_and_save(self, img_size, channels, img_name):
        n_row, n_col = 10, 6
        if self.num_classes is not None:
            if self.num_classes <= n_row:
                n_row = self.num_classes
                labels = torch.tensor([num for num in range(n_row) for _ in range(n_col)]).long()
                gen_images = self.diffusion.sample(img_size, n_row*n_col, channels, labels)[-1]
            else:
                random_labels = np.random.choice(np.arange(self.num_classes), size=n_row, replace=False)
                labels = torch.tensor([num for num in random_labels for _ in range(n_col)]).long()
                gen_images = self.diffusion.sample(img_size, n_row*n_col, channels, labels)[-1]
        else:
            gen_images = self.diffusion.sample(img_size, n_row*n_col, channels)[-1]
        gen_images = reverse_transform(gen_images)
        image_path = os.path.join(self.result_folder, f"sample-{img_name}.png")
        save_image(gen_images, image_path, nrow=n_row) 
        
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
                    self.sample_and_save(img_size, channels=ch, img_name=milestone)
                    
        print("Train finished!")
                    

            
