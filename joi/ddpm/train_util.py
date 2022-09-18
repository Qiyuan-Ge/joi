import os
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image


def reverse_transform(img):
    return (img + 1) * 0.5


def rate(step, warmup):
    if step == 0:
        step = 1
    return 10000 * (512 ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


class DiffusionTrainer:
    def __init__(self, 
                 diffusion, 
                 timesteps, 
                 lr, 
                 weight_decay, 
                 dataloader, 
                 warm_up_steps=8000, 
                 sample_interval=None, 
                 device=None, 
                 result_folder=None, 
                 num_classes=None,
                ):
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.diffusion = diffusion
        self.timesteps = timesteps
        self.optimizer = torch.optim.AdamW(self.diffusion.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_scheduler = LambdaLR(
            optimizer=self.optimizer, lr_lambda=lambda step: rate(step, warm_up_steps)
        )
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
                label_lst = np.arange(n_row)
            else:
                label_lst = np.random.choice(np.arange(self.num_classes), size=n_row, replace=False)
            labels = torch.tensor([num for _ in range(n_col) for num in label_lst]).long().to(self.device)
            gen_images = self.diffusion.sample(img_size, n_row*n_col, channels, labels)[-1]
        else:
            gen_images = self.diffusion.sample(img_size, n_row*n_col, channels)[-1]
        gen_images = torch.clamp(reverse_transform(gen_images), 0, 1)
        image_path = os.path.join(self.result_folder, f"sample-{img_name}.png")
        save_image(gen_images, image_path, nrow=n_row) 
        
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for step, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                imgs, labels = batch
                batch_size, ch, img_size, img_size = imgs.shape
                imgs = imgs.to(self.device)
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                if self.num_classes is not None:
                    labels = labels.to(self.device)
                    loss = self.diffusion.p_losses(imgs, t, y=labels)
                else:
                    loss = self.diffusion.p_losses(imgs, t)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                
                print(
                    "[Epoch %d|%d] [Batch %d|%d] [loss: %f] [lr: %f]"
                    % (epoch, num_epochs, step, len(self.dataloader), loss, self.optimizer.state_dict()['param_groups'][0]['lr'])
                    )
    
                # save generated images
                milestone = epoch * len(self.dataloader) + step
                if milestone != 0 and milestone % self.sample_interval == 0:
                    self.sample_and_save(img_size, channels=ch, img_name=milestone)
                    
        print("Train finished!")
                    
