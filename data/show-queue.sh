#!/bin/bash
# Show High-priority
echo '-------------------------------'
echo 'Queue in high-priority clusters'
echo '-------------------------------'
queues="yq01-v100-box-1-8 yq01-v100-box-idl-2-8"
for queue in ${queues}
do
  showjob -p ${queue}
  sleep 0.3s
done

echo '-------------------------------'
echo 'Queue in low-priority clusters'
echo '-------------------------------'

#queues="yq01-p40-3-8 yq01-p40-2-8 yq01-p40-box-1-8 yq01-v100-box-2-8"
queues="yq01-p40-3-8 yq01-p40-box-1-8 yq01-v100-box-2-8"
for queue in ${queues}
do
  showjob -p ${queue}
  sleep 0.3s
done


echo '-------------------------------'
echo 'Queue for other IDL teams'
echo '-------------------------------'

queues="yq01-v100-box-idl-8 yq01-v100-box-idl-3-8"
for queue in ${queues}
do
  showjob -p ${queue}
  sleep 0.3s
done
