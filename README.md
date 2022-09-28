
## Denoising Diffusion Probability Model

<div align=center>
<img src="https://user-images.githubusercontent.com/53368178/190867851-6d84fb48-ead7-47f1-90d4-4b71c7622396.png">
</div>

## Install
````
pip install git+https://github.com/Qiyuan-Ge/joi.git
````

## Usage

#### MNIST
- without guidence

<div align=center>
<img src="https://github.com/Qiyuan-Ge/joi/blob/main/samples/mnist_random.png">
</div>

````
python ddpm_train.py --n_epochs=200 --bs=128 --lr=1e-4 --timesteps=500 --wd=1e-4 --dropout=0.1 --dataset='mnist' --lr_decay=True --channels=1
````

- classifier free guidance

<div align=center>
<img src="https://github.com/Qiyuan-Ge/joi/blob/main/samples/mnist.png">
</div>

````
python ddpm_train.py --n_epochs=200 --bs=128 --lr=1e-4 --timesteps=500 --wd=1e-4 --dropout=0.1 --num_classes=10 --dataset='mnist' --lr_decay=True --channels=1
````

#### Cifar10
- classifier free guidance

<div align=center>
<img src="https://github.com/Qiyuan-Ge/joi/blob/main/samples/cifar-10.png">
</div>

````
python ddpm_train.py --n_epochs=800 --bs=64 --lr=1e-4 --timesteps=1000 --wd=1e-4 --dropout=0.1 --num_classes=10 --lr_decay=True
````

## Reference
````
- Lil’Log
  https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- annotated-diffusion
  https://huggingface.co/blog/annotated-diffusion
- Improved Denoising Diffusion Probabilistic Models
  https://arxiv.org/abs/2102.09672
  https://github.com/openai/improved-diffusion
- Cascaded Diffusion Models for High Fidelity Image Generation
  https://arxiv.org/abs/2106.15282
````

