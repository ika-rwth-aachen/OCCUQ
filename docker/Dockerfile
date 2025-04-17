FROM nvcr.io/nvidia/pytorch:21.10-py3

# Set environment variables for CUDA
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

ENV DEBIAN_FRONTEND=noninteractive

# Install the required packages
RUN apt-get update \
    && apt-get install -y wget vim htop screen ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 git nano sudo python3-dev build-essential x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set PATH
ENV PATH="/root/.local/bin:${PATH}"

# Install MMEngine, MMCV and MMDetection
RUN pip install openmim && \
    mim install "mmengine" "mmcv==1.4.0" "mmdet==2.14.0" "mmsegmentation==0.14.1" \
    && pip install gpustat spconv-cu114 ipdb timm open3d-python

# Workarounds
RUN pip install tensorboard==2.9.0 scikit-image==0.18.3 scikit-learn==1.0.1 nuscenes-devkit==1.1.3 yapf==0.40.1 setuptools==59.5.0
RUN pip install netcal --no-deps
RUN pip install tikzplotlib==0.9.8
RUN pip install protobuf==3.20.*
RUN pip install pyro-ppl --no-deps
RUN pip install opt-einsum
RUN pip install einops --no-deps
RUN pip install numba==0.48.0 --ignore-installed
RUN pip install backports.tarfile
RUN pip install opencv-python==4.5.5.64
RUN pip install numpy==1.22.4

RUN git clone --branch v0.17.1 https://github.com/open-mmlab/mmdetection3d.git mmdetection3d && \
    cd mmdetection3d && \
    python setup.py install || true

RUN conda clean --all
