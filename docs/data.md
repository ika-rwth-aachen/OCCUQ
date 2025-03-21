<!-- omit in toc -->
# Outline
- [SurroundOcc Dataset](#surroundocc-dataset)
- [MultiCorrupt Dataset](#multicorrupt-dataset)
  - [Generate Dataset](#generate-dataset)
  - [Download Dataset](#download-dataset)


## SurroundOcc Dataset

**1. Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). Folder structure:**
```
SurroundOcc
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
```


**2. Download the generated [train](https://cloud.tsinghua.edu.cn/f/ebbed36c37b248149192/?dl=1)/[val](https://cloud.tsinghua.edu.cn/f/b3f169f4db034764bb87/?dl=1) pickle files and put them in data.**

**3. Download our generated dense occupancy labels (resolution 200x200x16 with voxel size 0.5m) and put and unzip it in data. We will also provide full-resolution mesh data, and you can subsample it with different resolution.**
| resolution | Subset | Link | Size |
| :---: | :---: | :---: | :---: |
| 200x200x16 | train | [link](https://cloud.tsinghua.edu.cn/f/ef8357724574491d9ddb/?dl=1) | 3.2G |
| 200x200x16 | val | [link](https://cloud.tsinghua.edu.cn/f/290276f4a4024896b733/?dl=1) | 627M |
| mesh vertices| train | [link](https://share.weiyun.com/rQXh35ME) | 170G |
| mesh vertices| val | [link](https://share.weiyun.com/Jdr5eFmZ) | 34G |

Please note that: <br/>
1. the shape of each npy file is (n,4), where n is the number of non-empty occupancies. Four dimensions represent xyz and semantic label respectively. <br/>
2. In our [dataloader](https://github.com/weiyithu/SurroundOcc/blob/d346e8ce476817dfd8492226e7b92660955bf89c/projects/mmdet3d_plugin/datasets/pipelines/loading.py#L32), we convert empty occupancies as label 0 and ignore class as label 255. <br/>
3. Our occupancy labels are the voxel indexes under LiDAR coordinate system, not the ego coordinate system. You can use the [code](https://github.com/weiyithu/SurroundOcc/blob/d346e8ce476817dfd8492226e7b92660955bf89c/projects/mmdet3d_plugin/datasets/evaluation_metrics.py#L19) to convert voxel indexes to the LiDAR points. <br/>


**Folder structure:**
```
SurroundOcc
├── data/
│   ├── nuscenes/
│   ├── nuscenes_occ/
│   ├── nuscenes_infos_train.pkl
│   ├── nuscenes_infos_val.pkl

```

## MultiCorrupt Dataset
To perform OOD detection task as in our paper you would need to *generate* or
*download* [MultiCorrupt](https://github.com/ika-rwth-aachen/MultiCorrupt).


### Generate Dataset
Follow the [instructions](https://github.com/ika-rwth-aachen/MultiCorrupt?tab=readme-ov-file#installation) 
to generate the dataset on your own, by corrupting the original nuScenes dataset.


### Download Dataset
Download the precompiled MultiCorrupt dataset from 
[Huggingface](https://huggingface.co/datasets/TillBeemelmanns/MultiCorrupt) 
and extract the compressed dataset with the following script:

```bash
#!/bin/bash

# Set directories
compressed_dir="multicorrupt"
destination_dir="multicorrupt_uncompressed"
mkdir -p "$destination_dir"

# Iterate over all split archives
for archive in "$compressed_dir"/*.tar.gz.part00; do
    base_name=$(basename "$archive" .tar.gz.part00)
    category=$(echo "$base_name" | cut -d'_' -f1)
    subfolder=$(echo "$base_name" | cut -d'_' -f2)
    
    # Create category directory if it doesn't exist
    mkdir -p "$destination_dir/$category/$subfolder"
    
    echo "Reconstructing and extracting $base_name..."
    cat "$compressed_dir/${base_name}.tar.gz.part"* | tar -xzvf - -C "$destination_dir/$category/$subfolder"
done
```