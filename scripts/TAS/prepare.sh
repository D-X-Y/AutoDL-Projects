#!/bin/bash
# bash ./scripts/TAS/prepare.sh
#datasets='cifar10 cifar100 imagenet-1k'
#ratios='0.5 0.8 0.9'
ratios='0.5'
save_dir=./.latent-data/splits

for ratio in ${ratios}
do
  python ./exps/TAS/prepare.py --name cifar10  --root $TORCH_HOME/cifar.python  --save ${save_dir}/cifar10-${ratio}.pth --ratio ${ratio}
  python ./exps/TAS/prepare.py --name cifar100 --root $TORCH_HOME/cifar.python  --save ${save_dir}/cifar100-${ratio}.pth --ratio ${ratio}
  python ./exps/TAS/prepare.py --name imagenet-1k --root $TORCH_HOME/ILSVRC2012 --save ${save_dir}/imagenet-1k-${ratio}.pth --ratio ${ratio}
done
