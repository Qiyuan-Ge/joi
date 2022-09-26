import os
import numpy as np
import torch
from torchvision.utils import save_image
from joi.util import EMA, Log


def reverse_transform(img, clip=True):
    if clip:
        img = img.clamp(-1, 1)
        
    return (img + 1) * 0.5


class DiffusionTrainer:
    def __init__(self, 
                 diffusion, 
                 timesteps, 
                 lr, 
                 weight_decay, 
                 dataloader,
                 lr_decay=0.9,
                 sample_interval=None, 
                 device=None, 
                 result_folder=None, 
                 num_classes=None,
                 ema_decay=0.95,
                ):
        self.lr = lr
        self.lr_decay = lr_decay
        self.device = device
        self.diffusion = diffusion
        self.timesteps = timesteps
        self.optimizer = torch.optim.AdamW(self.diffusion.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.result_folder = result_folder
        self.sample_interval = sample_interval
        self.ema = EMA(self.diffusion.model, ema_decay)
        self.diffusion.to(self.device)
        
    def _lr_update(self, steps):
        lr = self.lr * (1 - self.lr_decay * steps / self.total_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
    def _ema_update(self, model):
        print("saving model state ...")
        self.ema.update(model)
        model_path = os.path.join(self.result_folder, "model_ema.pt")
        torch.save(self.ema.model_ema.state_dict(), model_path)
    
    def sample_and_save(self, img_size, channels, img_name):
        n_row, n_col = 10, 10
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
        gen_images = reverse_transform(gen_images, clip=True)
        image_path = os.path.join(self.result_folder, f"sample-{img_name}.png")
        save_image(gen_images, image_path, nrow=n_row) 
        
    def train(self, num_epochs):
        self.total_steps = len(self.dataloader) * num_epochs
        for epoch in range(num_epochs):
            log = Log()
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
                
                log.add({'total_loss':float(loss)*batch_size, 'n_sample':batch_size})
                log.update({'loss':float(loss), 'lr':self.optimizer.param_groups[0]['lr']})
                print(
                    "[Epoch %d|%d] [Batch %d|%d] [Loss %f|%f] [Lr %f]"
                    % (epoch, num_epochs, step, len(self.dataloader), log['loss'], log['total_loss']/log['n_sample'], log['lr'])
                    )
                
                steps = epoch * len(self.dataloader) + step
                if self.lr_decay is not None:
                    self._lr_update(steps)

                # save generated images
                if steps != 0 and steps % self.sample_interval == 0:
                    self.sample_and_save(img_size, channels=ch, img_name=steps)
                    
            self._ema_update(self.diffusion.model)
                    
        print("Train finished!")
                    
