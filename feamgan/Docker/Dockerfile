ARG DEBIAN_FRONTEND=noninteractive

# Base
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update 
RUN apt-get -y upgrade
RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends git 
RUN git config --global --add safe.directory /root/feamgan

# BDD Dataset 
RUN apt-get install -y xvfb
RUN apt-get install -y ffmpeg
RUN pip install scikit-video==1.1.11

# Cityscapes Dataset 
RUN pip install cityscapesscripts==2.2.2

# FeaMGAN
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110==1.7.0
WORKDIR /tmp/unique_for_apex
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /tmp/unique_for_apex/apex
RUN git reset --hard a651e2c24ecf97cbf367fd3f330df36760e1c597
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN pip install termcolor==2.3.0
RUN pip install scipy==1.7.3
RUN pip install pandas==1.3.5
RUN pip install torch-fidelity==0.3.0
RUN pip install moviepy==1.0.3
RUN pip install imageio==2.9.0
RUN pip install wandb==0.15.5