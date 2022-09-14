import pathlib
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
import joi.ddpm as ddpm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--bs", type=int, default=32, help="batch size: size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="adamw: learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="adamw: weight decay")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--timesteps", type=int, default=1000, help="timesteps, default: 1000") 
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--dataset", type=str, default='cifar10', help="mnist, cifar10, default: cifar10")
    parser.add_argument("--beta_schedule", type=str, default='cosine', help="beta schedule: cosine, linear, default: cosine")
    parser.add_argument("--loss_type", type=str, default='l2', help="loss type: l1, l2, huber, default: l2")
    parser.add_argument("--sample_interval", type=int, default=800, help="interval between image sampling")
    parser.add_argument("--data_path", type=str, default='none', help="set your own data path")
    arg = parser.parse_args()
    print(arg)
    
    if arg.dataset == 'mnist':
        if arg.data_path == 'none':
            root = "./dataset/MNIST"
            data_path = pathlib.Path(root)
            data_path.mkdir(exist_ok=True)
        else:
            root = arg.data_path
            
        dataset = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=T.Compose(
                [T.Resize(arg.img_size), 
                 T.ToTensor(), 
                 T.Lambda(lambda t: (t * 2) - 1)]
                ),
        )
    elif arg.dataset == 'cifar10':
        if arg.data_path == 'none':
            root = "./dataset/Cifar10"
            data_path = pathlib.Path(root)
            data_path.mkdir(exist_ok=True)
        else:
            root = arg.data_path
            
        dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=T.Compose(
                [T.Resize(arg.img_size), 
                 T.RandomHorizontalFlip(), 
                 T.ToTensor(), 
                 T.Lambda(lambda t: (t * 2) - 1)]
                ),
        )
        
    model, diffusion = ddpm.create_model_and_diffusion(img_size=arg.img_size, 
                                                       num_channels=arg.channels, 
                                                       num_classes=None, 
                                                       dropout=arg.dropout, 
                                                       timesteps=arg.timesteps, 
                                                       beta_schedule=arg.beta_schedule, 
                                                       loss_type=arg.loss_type)
                
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=arg.bs, shuffle=True)
    
    res_path = pathlib.Path.cwd() / "result"
    res_path.mkdir(exist_ok = True)
    print(f"images saved path: {res_path}")

    trainer = ddpm.DDPM_Trainer(diffusion, 
                                lr=arg.lr, 
                                weight_decay=arg.wd, 
                                dataloader=dataloader, 
                                sample_interval=arg.sample_interval, 
                                result_folder=res_path)
    trainer.train(arg.n_epochs)


if __name__ == "__main__":
    main()
    
 