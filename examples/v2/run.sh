#!bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

mkdir -p logs
mkdir -p cache
mkdir -p checkpoint

rm -rf logs/*
rm -rf cache/*

time python main.py
tensorboard --logdir=./logs --bind_all
