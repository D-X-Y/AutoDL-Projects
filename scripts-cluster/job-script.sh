#!/bin/bash
#
echo "CHECK-DATA-DIR START"
sh /home/HGCP_Program/software-install/afs_mount/bin/afs_mount.sh \
    COMM_KM_Data COMM_km_2018 \
    `pwd`/hadoop-data \
    afs://xingtian.afs.baidu.com:9902/user/COMM_KM_Data/dongxuanyi/datasets

export TORCH_HOME="./data/data/"
#wget -q http://10.127.2.44:8000/cifar.python.tar --directory-prefix=${TORCH_HOME}
#tar -xvf ${TORCH_HOME}/cifar.python.tar -C ${TORCH_HOME}
tar -xf ./hadoop-data/cifar.python.tar -C ${TORCH_HOME}
#rm ${TORCH_HOME}/cifar.python.tar
#tar xvf ./hadoop-data/ILSVRC2012.tar   -C ${TORCH_HOME}

cifar_dir="${TORCH_HOME}/cifar.python"
if [ -d ${cifar_dir} ]; then
  echo "Find cifar-dir: "${cifar_dir}
else
  echo "Can not find cifar-dir: "${cifar_dir}
  exit 1
fi
echo "CHECK-DATA-DIR DONE"

PID=$$

# config python
PYTHON_ENV=py36_pytorch1.0_env0.1.3.tar.gz
wget -e "http_proxy=cp01-sys-hic-gpu-02.cp01:8888" http://cp01-sys-hic-gpu-02.cp01/HGCP_DEMO/$PYTHON_ENV > screen.log 2>&1
tar xzf $PYTHON_ENV

echo "JOB-PID   : "${PID}
echo "JOB-PWD   : "$(pwd)
echo "JOB-files : "$(ls)
echo "JOB-CUDA_VISIBLE_DEVICES: " ${CUDA_VISIBLE_DEVICES}

./env/bin/python --version
echo "JOB-TORCH_HOME: "${TORCH_HOME}

# real commands
