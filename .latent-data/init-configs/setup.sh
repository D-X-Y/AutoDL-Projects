#!/bin/bash

script=$(readlink -f "$0")
script_dir=$(dirname "$script")
echo "script-directory: $script_dir"

cp ${script_dir}/tmux.conf ~/.tmux.conf
cp ${script_dir}/vimrc ~/.vimrc
cp ${script_dir}/bashrc ~/.bashrc
cp ${script_dir}/condarc ~/.condarc

wget https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
