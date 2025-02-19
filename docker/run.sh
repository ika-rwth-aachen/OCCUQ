#!/bin/bash

IMAGE_NAME=occuq
data="/path/to/data"
multicorrupt="/path/to/multicorrupt"
repo_dir="/path/to/occuq"

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
