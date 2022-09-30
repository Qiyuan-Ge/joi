import torch
from torch.utils.data import DataLoader
from joi.ddpm.t5 import tokenize


def collate_fn_t5(batch):
    img_batch = []
    txt_batch = []
    for img, txt in batch:
        img_batch.append(img)
        txt_batch.append(txt)
    img_batch = torch.tensor(img_batch)
    input_ids = tokenize(txt_batch)
    
    return img_batch, input_ids
    
         
def Txt2ImgDataloader(dataset, batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn_t5):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
