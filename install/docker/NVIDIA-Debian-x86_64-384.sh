#!/bin/bash

sudo apt update
sudo apt upgrade
sudo apt purge nvidia*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install -y --no-install-recommends nvidia-384
sudo apt install -y --no-install-recommends libcuda1-384
