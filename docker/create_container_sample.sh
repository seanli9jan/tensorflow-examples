#!/bin/bash

sudo nvidia-docker run --privileged            \
-itd --hostname docker --name seanli9jan       \
-e COLUMNS=$(tput cols) -e LINES=$(tput lines) \
-v $HOME/docker/workdir:/root/workdir          \
-p 6006:6006 -p 8888:8888                      \
seanli9jan/nvidia:cuda9.0-cudnn7-devel
