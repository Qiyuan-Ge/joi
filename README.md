
## Denoising Diffusion Probability Model

<div align=center>
<img src="https://user-images.githubusercontent.com/53368178/190867851-6d84fb48-ead7-47f1-90d4-4b71c7622396.png">
</div>

## Install
````
python pip install git+https://github.com/Qiyuan-Ge/joi.git
````

## Display

#### MNIST
- without guidence

<div align=center>
<img src="https://user-images.githubusercontent.com/53368178/190886956-b83eaa4d-4154-42da-a40a-91f233a46e10.png">
</div>

````
python ddpm_train.py --bs=128 --lr=1e-4 --wd=1e-4 --dropout=0.1 --img_size=32 --channels=1 --timesteps=200 --dataset='mnist'
````

- classifier free guidance

<div align=center>
<img src="https://user-images.githubusercontent.com/53368178/190882823-17c86cdd-760d-430d-9686-feaf4cd2072c.png">
</div>

````
python ddpm_train.py --bs=128 --lr=1e-4 --wd=1e-4 --dropout=0.1 --img_size=32 --channels=1 --timesteps=200 --dataset='mnist' --num_classes=10
````

#### Cifar10
- classifier free guidance

<div align=center>
<img src="https://user-images.githubusercontent.com/53368178/190882823-17c86cdd-760d-430d-9686-feaf4cd2072c.png">
</div>

````
python ddpm_train.py --bs=128 --lr=1e-4 --wd=1e-4 --dropout=0.1 --img_size=32 --timesteps=1000 --num_classes=10
````

## Reference
````
- Lil’Log
  https://lilianweng.github.io/
- annotated-diffusion
  https://huggingface.co/blog/annotated-diffusion
````

