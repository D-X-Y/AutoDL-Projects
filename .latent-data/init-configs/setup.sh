#!/bin/bash

script=$(readlink -f "$0")
script_dir=$(dirname "$script")
echo "script-directory: $script_dir"

cp ${script_dir}/tmux.conf ~/.tmux.conf
cp ${script_dir}/vimrc ~/.vimrc

wget https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh

