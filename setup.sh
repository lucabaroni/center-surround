#!/bin/bash
set -e

# Create uv venv named center_surround with Python 3.9
uv venv center_surround --python 3.9
source center_surround/bin/activate

# Install PyTorch 1.13.1 with CUDA 11.7 (compatible with your 12.9 driver)
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Create directory for cloned packages
# Create directory for cloned packages (in project directory)
mkdir -p github_packages
cd github_packages

# Clone and install nnvision
rm -rf nnvision
git clone -b model_builder https://github.com/sinzlab/nnvision.git
uv pip install -e nnvision/

# Clone and install data_port
rm -rf data_port
git clone --depth 1 --branch challenge https://github.com/sinzlab/data_port.git
uv pip install -e data_port/

# Return to project directory
cd -

# Install featurevis_mod from lucabaroni
uv pip install git+https://github.com/jiakunf/featurevis.git

# Install packages from Dockerfile
uv pip install \
    transformers==4.35.2 \
    "deeplake[enterprise]" \
    moviepy \
    imageio \
    tqdm \
    statsmodels \
    param==1.5.1 \
    matplotlib \
    ipykernel \
    opencv-python

# Install imagen from CSNG-MFF
uv pip install git+https://github.com/CSNG-MFF/imagen.git

uv pip install git+https://github.com/lucabaroni/classicalv1.git

# wandb
uv pip install wandb

# Install the project itself
uv pip install -e .