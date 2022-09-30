import os
import torch
from torchvision.utils import save_image
from joi.utils import EMA, Log


def exists(x):
    return x is not None


def reverse_transform(img, clip=True):
    if clip:
        img = img.clamp(-1, 1)
        
    return (img + 1) * 0.5


class Trainer:
    def __init__(self, 
                 diffusion, 
                 timesteps, 
                 lr, 
                 wd, 
                 dataloader,
                 lr_decay=0.9,
                 device=None, 
                 condition=None,
                 ema_decay=0.95,
                 result_folder=None,
                 sample_interval=None,  
                ):
        self.lr = lr
        self.lr_decay = lr_decay
        self.device = device
        self.diffusion = diffusion
        self.timesteps = timesteps
        self.optimizer = torch.optim.AdamW(self.diffusion.model.parameters(), lr=lr, weight_decay=wd)
        self.dataloader = dataloader
        self.condition = condition
        self.image_folder = os.path.join(result_folder, 'image')
        self.model_folder = os.path.join(result_folder, 'model')
        self.sample_interval = sample_interval
        self.ema = EMA(self.diffusion.model, ema_decay)
        self.diffusion.to(self.device)
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)
        
    def _lr_update(self):
        lr = self.lr * (1 - self.lr_decay * self.steps / self.total_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
    def _ema_update(self, model):
        print("saving model state ...")
        self.ema.update(model)
        model_path = os.path.join(self.model_folder, "model_ema.pt")
        torch.save(self.ema.model_ema.state_dict(), model_path)
    
    def sample_and_save(self, img_size, channels, img_name):
        (n_row, n_col) = (10, 10) if img_size < 64 else (5, 5)
        gen_images = self.diffusion.sample(img_size, n_row*n_col, channels)[-1]
        gen_images = reverse_transform(gen_images, clip=True)
        image_path = os.path.join(self.image_folder, f"sample-{img_name}.png")
        save_image(gen_images, image_path, nrow=n_row)
        
    def forward(self, batch):
        imgs, cond = batch
        batch_size, ch, img_size, img_size = imgs.shape
        imgs = imgs.to(self.device)
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        if exists(self.condition):
            cond = cond.to(self.device)
            loss = self.diffusion(imgs, t, y=cond)
        else:
            loss = self.diffusion(imgs, t)
        
        return loss, batch_size, ch, img_size
         
    def train(self, num_epochs):
        self.steps = 0
        self.total_steps = len(self.dataloader) * num_epochs
        for epoch in range(num_epochs):
            log = Log()
            for step, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                loss, batch_size, ch, img_size = self.forward(batch)
                loss.backward()
                self.optimizer.step()
                
                log.add({'total_loss':float(loss)*batch_size, 'n_sample':batch_size})
                log.update({'loss':float(loss), 'lr':self.optimizer.param_groups[0]['lr']})
                print(
                    "[Epoch %d|%d] [Batch %d|%d] [Loss %f|%f] [Lr %f]"
                    % (epoch, num_epochs, step, len(self.dataloader), log['loss'], log['total_loss']/log['n_sample'], log['lr'])
                    )
                
                self.steps += 1
                if self.lr_decay is not None:
                    self._lr_update()

                # save generated images
                if self.steps != 0 and self.steps % self.sample_interval == 0:
                    self.sample_and_save(img_size, channels=ch, img_name=self.steps)
                    
            self._ema_update(self.diffusion.model)
                    
        print("Train finished!")
                    