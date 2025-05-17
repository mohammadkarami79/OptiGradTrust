from setuptools import setup, find_packages

setup(
    name="federated_learning",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
    ],
) 