ARG CUDA_VERSION=11.7.1
ARG OS_VERSION=22.04
# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}


# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: Most commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
ARG CUDA_ARCHITECTURES=86;80;75

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    protobuf-compiler \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    vim-tiny \
    wget \
    && \
    rm -rf /var/lib/apt/lists/*

# Add glog path to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# Upgrade pip and install packages.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip "setuptools<70.0" pathtools promise pybind11
SHELL ["/bin/bash", "-c"]
# Install pytorch and submodules
ENV TORCH_CUDA_ARCH_LIST="8.6 8.0 7.5"
RUN CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && python3.10 -m pip install --no-cache-dir \
    torch==2.0.0+cu${CUDA_VER} \
    torchvision==0.15.0+cu${CUDA_VER} \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}


RUN python3.10 -m pip install xformers==0.0.18
RUN python3.10 -m pip install timm==0.9.5
RUN python3.10 -m pip install pytorch_msssim==1.0.0

RUN python3.10 -m pip install omegaconf==2.3.0
RUN python3.10 -m pip install torchmetrics==0.10.3

RUN python3.10 -m pip install tensorboard==2.11.0
RUN python3.10 -m pip install wandb==0.15.8
RUN python3.10 -m pip install matplotlib==3.5.3
RUN python3.10 -m pip install plotly==5.13.1
RUN python3.10 -m pip install viser==0.2.1

RUN python3.10 -m pip install imageio
RUN python3.10 -m pip install imageio-ffmpeg
RUN python3.10 -m pip install scikit-image==0.20.0
RUN python3.10 -m pip install opencv-python

RUN python3.10 -m pip install open3d==0.16.0
RUN python3.10 -m pip install pyquaternion==0.9.9
RUN python3.10 -m pip install chumpy
RUN python3.10 -m pip install numpy==1.23.1
RUN python3.10 -m pip install kornia==0.7.2

RUN python3.10 -m pip install tqdm
RUN python3.10 -m pip install gdown

RUN python3.10 -m pip install nerfview==0.0.3
RUN python3.10 -m pip install lpips==0.1.4
RUN FORCE_CUDA=1 python3.10 -m pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
RUN FORCE_CUDA=1 python3.10 -m pip install git+https://github.com/facebookresearch/pytorch3d.git
RUN FORCE_CUDA=1 python3.10 -m pip install git+https://github.com/NVlabs/nvdiffrast
RUN python3.10 -m pip install git+https://github.com/scaleapi/pandaset-devkit.git#egg=pandaset&subdirectory=python
RUN python3.10 -m pip install nuscenes-devkit

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup default nightly-2024-02-04
RUN python3.10 -m pip install git+https://github.com/ziyc/av2-api

RUN mkdir /drivestudio
RUN mkdir /drivestudio/third_party
ADD ./third_party/smplx/ /drivestudio/third_party/smplx/
WORKDIR /drivestudio/third_party/smplx/
RUN python3.10 -m pip install -e .

WORKDIR /drivestudio
# Bash as default entrypoint.
CMD /bin/bash -l
