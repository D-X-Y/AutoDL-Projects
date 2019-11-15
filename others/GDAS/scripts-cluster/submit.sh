#!/bin/bash
# bash ./scripts-cluster/submit.sh ${QUEUE} ${JOB-NAME} ${GPUs}
#find -name "._*" | xargs rm -rf
ODIR=$(pwd)
FDIR=$(cd $(dirname $0); pwd)
echo "Bash-Dir  : "${ODIR}
echo "File-Dir  : "${FDIR}
echo "File-Name : "${0}

if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for the queue-name, the job-name, and the number-of-GPUs"
  exit 1               
fi
find -name "__pycache__" | xargs rm -rf

QUEUE=$1
NAME=$2
GPUs=$3
CMD=$4
TIME=$(date +"%Y-%h-%d--%T")
TIME="${TIME//:/-}"

JOB_SCRIPT="${FDIR}/tmps/job-${TIME}.sh"
HDFS_DIR="/user/COMM_KM_Data/${USER}/logs/alljobs/${NAME}-${TIME}"
echo "JOB-SCRIPT: "${JOB_SCRIPT}

cat ${FDIR}/job-script.sh > ${JOB_SCRIPT}
echo ${CMD}              >> ${JOB_SCRIPT}

${HDP} -mkdir ${HDFS_DIR} 
echo "Create "${HDFS_DIR}" done!"
sleep 1s

HGCP_CLIENT_BIN="${HOME}/.hgcp/software-install/HGCP_client/bin"

${HGCP_CLIENT_BIN}/submit \
    --hdfs afs://xingtian.afs.baidu.com:9902 \
    --hdfs-user COMM_KM_Data \
    --hdfs-passwd COMM_km_2018 \
    --hdfs-path ${HDFS_DIR} \
    --file-dir ./ \
    --job-name ${NAME} \
    --queue-name ${QUEUE} \
    --num-nodes 1 \
    --num-task-pernode 1 \
    --gpu-pnode ${GPUs} \
    --time-limit 0 \
    --job-script ${JOB_SCRIPT}

#--job-script ${FDIR}/job-script.sh
#echo "JOB-SCRIPT: " ${JOB_SCRIPT}
