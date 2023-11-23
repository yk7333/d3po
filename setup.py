from setuptools import setup, find_packages

setup(
    name="d3po-pytorch",
    version="0.0.1",
    packages=["d3po_pytorch"],
    python_requires=">=3.10",
    install_requires=[
        "ml-collections",
        "absl-py",
        "diffusers==0.17.1",
        "wandb",
        "torchvision",
        "inflect==6.0.4",
        "pydantic==1.10.9",
        "transformers==4.30.2",
        "accelerate==0.22.0",
        "torch==2.0.1"
    ],
)
