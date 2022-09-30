import pathlib
import argparse
import torch
import torchvision.transforms as T
import joi.ddpm as ddpm
import joi.datasets as datasets

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
    parser.add_argument("--num_classes", type=int, default=None, help="if condition on class")
    parser.add_argument("--dataset", type=str, default='CIFAR10', help="MNIST, CIFAR10, CelebA, default: CIFAR10")
    parser.add_argument("--beta_schedule", type=str, default='cosine', help="beta schedule: cosine, linear, default: cosine")
    parser.add_argument("--loss_type", type=str, default='l1', help="loss type: l1, l2, huber, default: l1")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="apply lr decay or not")
    parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
    parser.add_argument("--data_path", type=str, default='none', help="set your own data path")
    parser.add_argument("--device", type=str, default='cuda', help="cuda or cpu, default: cuda")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="Exponential Moving Average, default: 0.99")
    arg = parser.parse_args()
    print(arg)
    
    if arg.data_path == 'none':
        root = f"./dataset/{arg.dataset}"
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

        
    model, diffusion = ddpm.create_model_and_diffusion(img_size=arg.img_size, 
                                                       in_channels=arg.channels,
                                                       num_res_blocks=arg.num_res_blocks,
                                                       out_channels=arg.out_channels, 
                                                       timesteps=arg.timesteps, 
                                                       beta_schedule=arg.beta_schedule, 
                                                       loss_type=arg.loss_type,
                                                       condition=arg.condition,
                                                       num_classes=arg.num_classes,
                                                       dropout=arg.dropout,
                                                       )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of learnable parameters: {n_parameters//1e6}M')            
    
    res_path = pathlib.Path.cwd() / "result"
    res_path.mkdir(exist_ok=True)
    print(f"result folder path: {res_path}")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=arg.bs, shuffle=True)

    trainer = ddpm.Trainer(diffusion, 
                           timesteps=arg.timesteps, 
                           lr=arg.lr, 
                           wd=arg.wd, 
                           dataloader=dataloader,
                           lr_decay=arg.lr_decay,
                           device=arg.device,
                           result_folder=res_path,
                           condition=arg.condition,
                           ema_decay=arg.ema_decay,
                           sample_interval=arg.sample_interval,
                           )
    trainer.train(arg.n_epochs)


if __name__ == "__main__":
    main()
    
 
