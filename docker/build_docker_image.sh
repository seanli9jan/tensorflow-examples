#!/bin/bash

if [ -z $1 ]; then
	option='-help'
else
	option=$1
fi

if [ $option == '-cpu' ]; then
    sudo docker build -t seanli9jan/ubuntu:16.04 . -f Dockerfile.devel-cpu

elif [ $option == '-gpu' ]; then
    sudo docker build -t seanli9jan/nvidia:cuda9.0-cudnn7-devel . -f Dockerfile.devel-gpu

elif [ $option == '-all' ]; then
    sudo docker build -t seanli9jan/ubuntu:16.04 . -f Dockerfile.devel-cpu
    sudo docker build -t seanli9jan/nvidia:cuda9.0-cudnn7-devel . -f Dockerfile.devel-gpu

else
	option='-help'
fi

if [ $option == '-help' ]; then
	echo 'Usage: build_docker_image.sh [options]'
	echo 'Options:'
	echo '  -cpu  build tensorflow-cpu docker image'
	echo '  -gpu  build tensorflow-gpu docker image'
	echo '  -all  build tensorflow-{cpu, gpu} docker image'
fi
