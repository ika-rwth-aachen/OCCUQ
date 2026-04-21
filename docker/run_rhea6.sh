#!/bin/bash

IMAGE_NAME=occuq
data="/work/data01/beemelmanns/occ"
multicorrupt="/work/data01/beemelmanns/multicorrupt_uncompressed"
repo_dir="/work/beemelmanns/ba_heidrich/occuq"

docker run \
--name occuq_container \
--rm \
--gpus 'all,"capabilities=compute,utility,graphics"' \
--env DISPLAY=${DISPLAY} \
--shm-size=64gb \
--net=host \
--user root \
--volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
--volume $HOME/.Xauthority:/root/.Xauthority:rw \
--mount source=$repo_dir,target=/workspace,type=bind,consistency=cached \
--mount source=$data,target=/workspace/data,type=bind,consistency=cached \
--mount source=$multicorrupt,target=/workspace/multicorrupt,type=bind,consistency=cached \
-it \
-d \
$IMAGE_NAME
