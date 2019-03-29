#!/bin/bash
#
echo "CHECK-DATA-DIR START"
cifar_dir="./data/data/cifar.python"
if [ -d ${cifar_dir} ]; then
  echo "Find cifar-dir: "${cifar_dir}
else
  echo "Can not find cifar-dir: "${cifar_dir}
  exit 1
fi
echo "CHECK-DATA-DIR DONE"

sh /home/HGCP_Program/software-install/afs_mount/bin/afs_mount.sh \
    COMM_KM_Data COMM_km_2018 \
    `pwd`/hadoop-data \
    afs://xingtian.afs.baidu.com:9902/user/COMM_KM_Data/dongxuanyi/datasets

echo "PWD: " `pwd`
echo "files::  " `ls`
echo "CUDA_VISIBLE_DEVICES: " ${CUDA_VISIBLE_DEVICES}

# config python
PYTHON_ENV=py36_pytorch1.0_env0.1.3.tar.gz
wget -e "http_proxy=cp01-sys-hic-gpu-02.cp01:8888" http://cp01-sys-hic-gpu-02.cp01/HGCP_DEMO/$PYTHON_ENV > screen.log 2>&1
tar xzf $PYTHON_ENV

alias python="./env/bin/python"

echo "Python:  " `which python`
