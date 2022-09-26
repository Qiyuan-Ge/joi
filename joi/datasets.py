import os
import torch
from PIL import Image
import torchvision.datasets as datasets

__all__ = ['MNIST', 'CIFAR10', 'CelebA']

def MNIST(root, download, transform):
    return datasets.MNIST(root=root, train=True, download=download, transform=transform)


def CIFAR10(root, download, transform):
    return datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    

class CelebA:
    def __init__(self, img_dir, ann_dir, type='identity', transform=None):
        """CelebA Dataset http://personal.ie.cuhk.edu.hk/~lz013/projects/CelebA.html

        Args:
            img_dir (str): image folder path
            ann_dir (str): anno folder path
            type (str, optional): 'identity' or 'attr'. Defaults to 'identity'.
            transform (torchvision.transforms, optional): torchvision.transforms. Defaults to None.
        """        
        self.img_dir = img_dir
        self.imgs = os.listdir(img_dir)
        if type == 'identity':
            self.img2id = {}
            with open(os.path.join(ann_dir, 'identity_CelebA.txt'), 'r') as f:
                for line in f.readlines():
                    name, id = line.strip().split(' ')
                    self.img2id[name] = int(id)
        self.transform = transform
        
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        ann = self.img2id[img_name]
        if self.transform is not None:
            img = self.transform(img)
        
        ann = torch.tensor(ann)
            
        return img, ann
    
    def __len__(self):
        return len(self.imgs)