import torch
from copy import deepcopy

class EMA:
    """Exponential Moving Average

    Args:
        model:
        decay:
        device:
    """    
    def __init__(self, model, decay=0.99, device='cpu'):
        self.model_ema = deepcopy(model)
        self.decay = decay
        self.device = device
        self.model_ema.to(device)
        
    @torch.no_grad()
    def update(self, model):
        for p_ema, p_model in zip(self.model_ema.state_dict().values(), model.state_dict().values()):
            p_model = p_model.to(self.device)
            p_ema.copy_(self.func(p_ema, p_model))
            
    def func(self, p1, p2):
        return self.decay * p1 + (1 - self.decay) * p2
    
    