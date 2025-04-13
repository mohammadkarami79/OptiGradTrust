from setuptools import setup, find_packages

setup(
    name="federated_learning",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "scikit-learn",
        "tqdm",
        "pillow",
    ],
) 