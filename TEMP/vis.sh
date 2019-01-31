#!/usr/bin/env sh

seeds="seed-8167  seed-908  seed-9242"
for seed in ${seeds}; do
python ./exps-nas/vis-arch.py --checkpoint ./snapshots/NAS/Search-cifar10-cut16-100/${seed}/checkpoint-search.pth \
                              --save_dir   ./snapshots/NAS-VIS/Search-cut16-100/${seed}
done


