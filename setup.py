from setuptools import setup, find_packages

setup(
    name = "joi",
    version = 0.0,
    author = 'Qiyuan Ge'
    author_email = 'geqiyuan1105@gmail.com',
    packages = find_packages(),
    install_requires = [
        'tqdm',
        'numpy',
        'einops',
        'torch>=1.6',
        'torchvision',
        'transformers',
        'sentencepiece',
    ],
)
