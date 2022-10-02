from setuptools import setup, find_packages

setup(
    name = 'joi',
    version = 0.0,
    author = 'Qiyuan Ge',
    author_email = 'geqiyuan1105@gmail.com',
    keywords = [
        'artificial intelligence',
        'deep learning',
        'diffusion models',
    ],      
    packages = find_packages(),
    install_requires=[
        'einops',
        'numpy',
        'pillow',
        'sentencepiece',
        'torch>=1.6',
        'torchvision',
        'transformers',
        'accelerate',
        'tqdm',
    ],
)
