#!/bin/bash

# Selecting an Installation Method :
#   - Installing TensorFlow on Ubuntu
#   - Installing TensorFlow from Sources

# This script does not support TensorFlow GPU install

if [ -z $1 ]; then
	option='--help'
else
	option=$1
fi

cd $(dirname $0)

if [ $option == '--Ubuntu' ]; then
	echo 'Installing TensorFlow on Ubuntu'
	# Installing with virtualenv
	sudo apt install python-pip python-dev python-virtualenv
	virtualenv ~/env
	source ~/env/bin/activate
	# Python 2.7
	pip install --upgrade tensorflow
	# Python 3.n
	#pip3 install --upgrade tensorflow

elif [ $option == '--Sources' ]; then
	echo 'Installing TensorFlow from Sources'
	# Install Bazel
	echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
	curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
	sudo apt update && sudo apt install bazel
	sudo apt upgrade bazel
	# Installing with virtualenv
	sudo apt install python-pip python-dev python-virtualenv
	virtualenv ~/env
	source ~/env/bin/activate
	# Install TensorFlow
	git clone https://github.com/tensorflow/tensorflow
	cd tensorflow
	#git checkout Branch # where Branch is the desired branch
	# Python 2.7
	pip install numpy wheel
	# Python 3.n
	#pip3 install numpy wheel
	./configure
	#bazel build --config=opt --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma //tensorflow/tools/pip_package:build_pip_package
	bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
	sudo bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
	# Can also installing with virtualenv
	# "tensorflow-1.2.1-cp27-cp27mu-linux_x86_64.whl" need to rename to your compile version
	pip install /tmp/tensorflow_pkg/tensorflow-1.2.1-cp27-cp27mu-linux_x86_64.whl

else
	echo 'Usage: Install.sh [options]'
	echo 'Options'
	echo '  --Ubuntu  Installing TensorFlow on Ubuntu'
	echo '  --Sources Installing TensorFlow from Sources'
fi

# Some computer vision library
#pip install pillow       # basic
#sudo apt install eog
#pip install scipy        # resize
#sudo apt install python-tk
#pip install matplotlib   # show 2d, 3d image
#pip install pygame       # webcam
#pip install scikit-image # convert 3d ary
#pip install pydicom
