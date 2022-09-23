import os
import numpy as np
import torch
from torchvision.utils import save_image
from joi.util import EMA


def reverse_transform(img):
    return (img + 1) * 0.5


class DiffusionTrainer:
    def __init__(self, 
                 diffusion, 
                 timesteps, 
                 lr, 
                 weight_decay, 
                 dataloader,
                 lr_decay=False,
                 sample_interval=None, 
                 device=None, 
                 result_folder=None, 
                 num_classes=None,
                 ema_decay=0.99,
                ):
        self.lr = lr
        self.steps = 0
        self.total_steps = None
        self.lr_decay = lr_decay
        self.device = device
        self.diffusion = diffusion
        self.timesteps = timesteps
        self.optimizer = torch.optim.AdamW(self.diffusion.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.dataloader = dataloader
        self.result_folder = result_folder
        self.sample_interval = sample_interval
        self.num_classes = num_classes
        self.ema = EMA(self.diffusion.model, ema_decay)
        self.diffusion.to(self.device)
        
    def _lr_update(self):
        lr = self.lr * (1 - 0.9 * self.steps / self.total_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
    def _ema_update(self, model):
        print("saving model state ...")
        self.ema.update(model)
        model_path = os.path.join(self.result_folder, "model_ema.pt")
        torch.save(self.ema.model_ema.state_dict(), model_path)
    
    def sample_and_save(self, img_size, channels, img_name):
        if img_size <= 64:
            n_row, n_col = 10, 10
        else:
            n_row, n_col = 4, 4
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
        self.total_steps = len(self.dataloader) * num_epochs
        for epoch in range(num_epochs):
            for step, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                imgs, labels = batch
                batch_size, ch, img_size, img_size = imgs.shape
                imgs = imgs.to(self.device)
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                if self.num_classes is not None:
                    labels = labels.to(self.device)
                    loss = self.diffusion(imgs, t, y=labels)
                else:
                    loss = self.diffusion(imgs, t)
                loss.backward()
                self.optimizer.step()
                if self.lr_decay:
                    self._lr_update()
                self.steps = epoch * len(self.dataloader) + step
                
                print(
                    "[Epoch %d|%d] [Batch %d|%d] [loss: %f] [lr: %f]"
                    % (epoch, num_epochs, step, len(self.dataloader), loss, self.optimizer.param_groups[0]['lr'])
                    )
    
                # save generated images
                if self.steps != 0 and self.steps % self.sample_interval == 0:
                    self.sample_and_save(img_size, channels=ch, img_name=self.steps)
                    
            self._ema_update(self.diffusion.model)
                    
        print("Train finished!")
        self.steps = 0
                    
