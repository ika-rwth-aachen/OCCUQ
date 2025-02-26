## RWTH Aachen CLAIX Cluster Setup

**1. ssh onto the CLAIX GPU Node.**
```shell
ssh username@login23-g-1.hpc.itc.rwth-aachen.de
```

**2. Switch to GCC-based compiler.**
```shell
module switch intel foss
```

**2. Load the CUDA 11.3 module.**
```shell
module load CUDA/11.3
```

**3. Set environment variables.**
```shell
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=$(which g++)
```

## Conda Setup

**1. Create a conda virtual environment and activate it.**
```shell
conda create -n occuq python==3.8 -y
conda activate occuq
```

**2. Install netcal.**
```shell
pip install netcal
```

**3. Install PyTorch and torchvision (tested on torch==1.10.1 & cuda=11.3).**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install mkl==2024.0
```

**4. Install gcc>=5 in conda env.**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**5. Install MMCV following the [official instructions](https://github.com/open-mmlab/mmcv).**
```shell
pip install mmcv-full==1.4.0
```

**6. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**7. Install ninja.**
```shell
pip install ninja
```

**8. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**9. Install other dependencies.**
```shell
pip install scikit_image==0.19.3
pip install lyft-dataset-sdk==0.0.8 numba==0.48.0 nuscenes-devkit==1.1.10 plyfile==0.8.1 networkx==2.2 numpy==1.21.5
pip install timm
pip install open3d-python
```

**10. Install Chamfer Distance.**
```shell
cd OCCUQ/extensions/chamfer_dist
python setup.py install --user
```

**11. Install other dependencies.**
```shell
pip install yapf==0.40.1 setuptools==59.5.0
pip install einops --no-deps
```

## Docker Setup

### Build Docker Image

```shell
./docker/build.sh
```

### Run Docker Container

```shell
./docker/run.sh
```

Attach to running container either with VS Code or via terminal.
```shell
docker exec -it occuq bash
```

### Install Chamfer Distance
Inside the container execute the following commands to install the Chamfer Distance extension.
```shell
./docker/run_in_container.sh
```



