import os
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
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
                 condition=None,
                 ema_decay=0.9999,
                 max_grad_norm=1.0,
                 result_folder=None,
                 sample_interval=None,  
                ):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device.type
        self.lr = lr
        self.lr_decay = lr_decay
        self.timesteps = timesteps
        self.diffusion = diffusion
        self.ema_decay = ema_decay
        if exists(self.ema_decay):
            self.ema = EMA(self.diffusion.model, ema_decay)
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=wd)
        self.diffusion, self.optimizer, self.dataloader = self.accelerator.prepare(self.diffusion, self.optimizer, dataloader)
        self.condition = condition
        if self.condition not in [None, 'text', 'class']:
            raise ValueError("condition must be None, text or class. ")
        self.image_dir = os.path.join(result_folder, 'image')
        self.model_dir = os.path.join(result_folder, 'model')
        self.sample_interval = sample_interval or len(self.dataloader)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _lr_update(self):
        lr = self.lr * (1 - self.lr_decay * self.steps / self.total_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
    def _ema_update(self, diffusion):
        self.accelerator.wait_for_everyone()
        unwrapped_diffusion = self.accelerator.unwrap_model(diffusion)
        self.ema.update(unwrapped_diffusion.model)
        model_path = os.path.join(self.model_dir, "model_ema.pt")
        self.accelerator.save(self.ema.model.state_dict(), model_path)
    
    def sample_and_save(self, img_size, channels, img_name):
        (n_row, n_col) = (10, 10) if img_size < 64 else (5, 5)
        unwrap_diffusion = self.accelerator.unwrap_model(self.diffusion)
        if exists(self.condition):
            n_row = min(n_row, self.curr_cond.shape[0])
            if self.condition == 'class':
                conds = self.curr_cond[:n_row].repeat(n_col)
            elif self.condition == 'text':
                conds = self.curr_cond[:n_row].repeat(n_col, 1)
            gen_images = unwrap_diffusion.sample(img_size, n_row*n_col, channels, conds)[-1]       
        else:
            gen_images = unwrap_diffusion.sample(img_size, n_row*n_col, channels)[-1]
        gen_images = reverse_transform(gen_images, clip=True)
        image_path = os.path.join(self.image_dir, f"sample-{img_name}.png")
        save_image(gen_images, image_path, nrow=n_row)
         
    def train(self, num_epochs):
        self.steps = 0
        self.total_steps = len(self.dataloader) * num_epochs
        for epoch in range(num_epochs):
            log = Log()
            with tqdm(self.dataloader, dynamic_ncols=True) as tqdm_dataloader:
                for batch in tqdm_dataloader:
                    self.optimizer.zero_grad()
                    
                    imgs, cond = batch
                    bs, ch, img_size, img_size = imgs.shape
                    t = torch.randint(0, self.timesteps, (bs,), device=self.device).long()
                    if exists(self.condition):
                        loss = self.diffusion(imgs, t, y=cond)
                    else:
                        loss = self.diffusion(imgs, t)
                    
                    self.accelerator.backward(loss)
                    if exists(self.max_grad_norm):  
                        self.accelerator.clip_grad_norm_(self.diffusion.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    if exists(self.ema_decay):
                        self._ema_update(self.diffusion)
                    
                    log.add({'total_loss':loss.item() * bs, 'n_sample':bs})
                    log.update({'loss':loss.item(), 'lr':self.optimizer.param_groups[0]['lr']})
                    tqdm_dataloader.set_postfix(
                        ordered_dict={
                            "Epoch"      : epoch,
                            "Loss"       : log['loss'],
                            "Mean Loss"  : log['total_loss']/log['n_sample'],
                            "LR"         : log['lr'],
                            }
                        )

                    self.steps += 1
                    if exists(self.lr_decay):
                        self._lr_update()
                    if self.steps != 0 and self.steps % self.sample_interval == 0:
                        self.curr_cond = cond
                        self.sample_and_save(img_size, channels=ch, img_name=self.steps)
                         
        print("Train finished!")
                    
