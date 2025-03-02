ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Set shell
SHELL ["/bin/bash", "-c"]

# Set environment variables for CUDA
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH
ENV CFLAGS="-I$CUDA_HOME/include $CFLAGS"
ENV FORCE_CUDA="1"

# Fix for non signed nvidia pub keys
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

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
    && pip install gpustat spconv-cu113 ipdb timm open3d-python

RUN conda install -c omgarcia gcc-6

# Workarounds
RUN pip install tensorboard==2.9.0 scikit-image==0.18.3 scikit-learn==1.0.1 nuscenes-devkit==1.1.3 yapf==0.40.1 setuptools==59.5.0
RUN pip install netcal --no-deps
RUN pip install tikzplotlib==0.9.8
RUN pip install protobuf==3.20.*
RUN pip install pyro-ppl --no-deps
RUN pip install opt-einsum
RUN pip install einops --no-deps
RUN pip install numba==0.48.0

RUN git clone --branch v0.17.1 https://github.com/open-mmlab/mmdetection3d.git mmdetection3d && \
    cd mmdetection3d && \
    python setup.py install || true

RUN conda clean --all
