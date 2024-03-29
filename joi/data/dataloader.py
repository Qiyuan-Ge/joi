import torch
from functools import partial
from torch.utils.data import DataLoader
from joi.ddpm.t5 import tokenize, encode_text, DEFAULT_T5_NAME

    
class collate_fn:
    def __init__(self, text_model_name=DEFAULT_T5_NAME):
        self.text_model_name = text_model_name
        self.encode_text = partial(encode_text, name=text_model_name)
    
    def __call__(self, batch):
        img_batch = []
        txt_batch = []
        for img, txt in batch:
            img_batch.append(img.unsqueeze(0))
            txt_batch.append(txt)
        img_batch = torch.cat(img_batch, dim=0)
        encoded_text, attn_mask = self.encode_text(txt_batch)

        return img_batch, encoded_text, attn_mask

    
def Txt2ImgDataloader(dataset, batch_size, shuffle=True, text_model_name=DEFAULT_T5_NAME, num_workers=0, pin_memory=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn(text_model_name), pin_memory=pin_memory)
