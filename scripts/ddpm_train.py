import pathlib
import argparse
import torch
import torchvision.transforms as T
import joi.ddpm as ddpm
from joi.data import datasets, Txt2ImgDataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--bs", type=int, default=32, help="batch size: size of the batches")
    parser.add_argument("--timesteps", type=int, default=1000, help="timesteps, default: 1000") 
    parser.add_argument("--lr", type=float, default=1e-4, help="adamw: learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="adamw: weight decay")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="number of residual blocks")
    parser.add_argument("--out_channels", type=int, default=None, help="number of output channels")
    parser.add_argument("--condition", type=str, default=None, help="unconditional or condition on text, class")
    parser.add_argument("--text_model_name", type=str, default='t5-small', help="t5-small, t5-base, default: t5-small")
    parser.add_argument("--num_classes", type=int, default=None, help="need to be specified when condition on class")
    parser.add_argument("--dataset", type=str, default='CIFAR10', help="MNIST, CIFAR10, CelebA, COCO2017, default: CIFAR10")
    parser.add_argument("--beta_schedule", type=str, default='cosine', help="beta schedule: cosine, linear, default: cosine")
    parser.add_argument("--loss_type", type=str, default='l1', help="loss type: l1, l2, huber, default: l1")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="apply lr decay or not")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="clips gradient norm of an iterable of parameters.")
    parser.add_argument("--p2_loss_weight_gamma", type=float, default=1.0, help="use p2 reweigh loss")
    parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
    parser.add_argument("--data_path", type=str, default='none', help="set your own data path")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="Exponential Moving Average, default: 0.9999")
    parser.add_argument("--num_workers", type=int, default=0, help="how many subprocesses to use for data loading, default: 0")
    parser.add_argument("--pin_memory", type=bool, default=False, help="default: False")
    arg = parser.parse_args()
    print(arg)
    
    if arg.data_path == 'none':
        root = pathlib.Path.cwd() / "dataset" / arg.dataset
        data_path = pathlib.Path(root)
        data_path.mkdir(exist_ok=True, parents=True)
    else:
        root = arg.data_path
    
    if arg.dataset == 'MNIST':     
        dataset = datasets.MNIST(
            root=root,
            download=True,
            transform=T.Compose(
                [T.Resize((arg.img_size, arg.img_size)), 
                 T.ToTensor(), 
                 T.Lambda(lambda t: (t * 2) - 1),]
                ),
            )
    elif arg.dataset == 'CIFAR10':
        dataset = datasets.CIFAR10(
            root=root,
            download=True,
            transform=T.Compose(
                [T.Resize((arg.img_size, arg.img_size)), 
                 T.RandomHorizontalFlip(), 
                 T.ToTensor(), 
                 T.Lambda(lambda t: (t * 2) - 1),]
                ),
            )
    elif arg.dataset == 'CelebA':
        dataset = datasets.CelebA(
            root=root,
            type='identity',
            transform=T.Compose(
                [T.Resize((arg.img_size, arg.img_size)), 
                 T.RandomHorizontalFlip(), 
                 T.ToTensor(), 
                 T.Lambda(lambda t: (t * 2) - 1),]
                ),
            )
    elif arg.dataset == 'COCO2017':
        dataset = datasets.Coco(
            root=root,
            dataType='train2017', 
            annType='captions', 
            transform=T.Compose(
                [T.Resize((arg.img_size, arg.img_size)), 
                 T.RandomHorizontalFlip(), 
                 T.ToTensor(), 
                 T.Lambda(lambda t: (t * 2) - 1),]
                ),
            )
    else:
        raise ValueError('Please select from MNIST, CIFAR10, CelebA, coco2017')

        
    model, diffusion = ddpm.create_model_and_diffusion(img_size=arg.img_size, 
                                                       in_channels=arg.channels,
                                                       num_res_blocks=arg.num_res_blocks,
                                                       out_channels=arg.out_channels, 
                                                       timesteps=arg.timesteps, 
                                                       beta_schedule=arg.beta_schedule, 
                                                       loss_type=arg.loss_type,
                                                       condition=arg.condition,
                                                       text_model_name=arg.text_model_name,
                                                       num_classes=arg.num_classes,
                                                       dropout=arg.dropout,
                                                       p2_loss_weight_gamma=arg.p2_loss_weight_gamma,
                                                       )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of learnable parameters: {n_parameters//1e6}M')            
    
    res_path = pathlib.Path.cwd() / "result"
    res_path.mkdir(exist_ok=True)
    print(f"result folder path: {res_path}")
    
    if arg.condition == 'text':
        dataloader = Txt2ImgDataloader(dataset, 
                                       batch_size=arg.bs, 
                                       shuffle=True,
                                       text_model_name=arg.text_model_name,
                                       num_workers=arg.num_workers, 
                                       pin_memory=arg.pin_memory,
                                      )
    else:
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=arg.bs, 
                                                 shuffle=True, 
                                                 num_workers=arg.num_workers, 
                                                 pin_memory=arg.pin_memory,
                                                )

    trainer = ddpm.Trainer(diffusion, 
                           timesteps=arg.timesteps, 
                           lr=arg.lr, 
                           wd=arg.wd, 
                           dataloader=dataloader,
                           lr_decay=arg.lr_decay,
                           condition=arg.condition,
                           ema_decay=arg.ema_decay,
                           max_grad_norm=arg.max_grad_norm,
                           result_folder=res_path,
                           sample_interval=arg.sample_interval,
                           )
    trainer.train(arg.n_epochs)


if __name__ == "__main__":
    main()
    
 
