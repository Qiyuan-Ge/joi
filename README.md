
## Denoising Diffusion Probability Model

<div align=center>
<img src="https://user-images.githubusercontent.com/53368178/190867851-6d84fb48-ead7-47f1-90d4-4b71c7622396.png">
</div>

## Install
````
python pip install git+https://github.com/Qiyuan-Ge/joi.git
````

## Usage

#### MNIST
- without guidence

<div align=center>
<img src="https://github.com/Qiyuan-Ge/joi/blob/main/samples/mnist_without_guidence.png">
</div>

````
python ddpm_train.py --bs=128 --lr=1e-4 --wd=1e-4 --dropout=0.1 --img_size=32 --channels=1 --timesteps=200 --dataset='mnist'
````

- classifier free guidance

<div align=center>
<img src="https://github.com/Qiyuan-Ge/joi/blob/main/samples/mnist.png">
</div>

````
python ddpm_train.py --bs=128 --lr=1e-4 --wd=1e-4 --dropout=0.1 --img_size=32 --channels=1 --timesteps=200 --dataset='mnist' --num_classes=10
````

#### Cifar10
- classifier free guidance

<div align=center>
<img src="https://github.com/Qiyuan-Ge/joi/blob/main/samples/cifar-10.png">
</div>

````
python ddpm_train.py --n_epochs=800 --bs=64 --lr=1e-4 --timesteps=1000 --wd=1e-4 --dropout=0.1 --num_classes=10
````

## Reference
````
- Lil’Log
  https://lilianweng.github.io/
- annotated-diffusion
  https://huggingface.co/blog/annotated-diffusion
````

