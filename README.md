
# Denoising Diffusion Probability Model

<div align=center>
<img src="https://user-images.githubusercontent.com/53368178/190867851-6d84fb48-ead7-47f1-90d4-4b71c7622396.png">
</div>

## Install
````
pip install git+https://github.com/Qiyuan-Ge/joi.git
````

## Image Generation

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

## Super Resolution

## Image Inpainting

## Text-to-Image Generation

## Multi GPU Training
````
accelerate config
accelerate launch ddpm_train.py
````
or
````
accelerate launch --multi_gpu ddpm_train.py
````

## Log
````
2022/10/4 
add gradient clip
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
- GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models
  https://arxiv.org/abs/2112.10741
- Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
  https://arxiv.org/abs/2205.11487
- Hierarchical Text-Conditional Image Generation with CLIP Latents
- Perception Prioritized Training of Diffusion Models
  https://arxiv.org/abs/2204.00227
````

## Citing
````
@Misc{accelerate,
  title =        {Accelerate: Training and inference at scale made simple, efficient and adaptable.},
  author =       {Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar},
  howpublished = {\url{https://github.com/huggingface/accelerate}},
  year =         {2022}
}
````

