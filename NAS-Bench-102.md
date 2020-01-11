# [NAS-BENCH-102: Extending the Scope of Reproducible Neural Architecture Search](https://openreview.net/forum?id=HJxyZkBKDr)

We propose an algorithm-agnostic NAS benchmark (NAS-Bench-102) with a fixed search space, which provides a unified benchmark for almost any up-to-date NAS algorithms.
The design of our search space is inspired by that used in the most popular cell-based searching algorithms, where a cell is represented as a directed acyclic graph.
Each edge here is associated with an operation selected from a predefined operation set.
For it to be applicable for all NAS algorithms, the search space defined in NAS-Bench-102 includes 4 nodes and 5 associated operation options, which generates 15,625 neural cell candidates in total.

In this Markdown file, we provide:
- [How to Use NAS-Bench-102](#how-to-use-nas-bench-102)
- [Instruction to re-generate NAS-Bench-102](#instruction-to-re-generate-nas-bench-102)
- [10 NAS algorithms evaluated in our paper](#to-reproduce-10-baseline-nas-algorithms-in-nas-bench-102)

Note: please use `PyTorch >= 1.2.0` and `Python >= 3.6.0`.

Simply type `pip install nas-bench-102` to install our api.

If you have any questions or issues, please post it at [here](https://github.com/D-X-Y/NAS-Projects/issues) or email me.

### Preparation and Download

The benchmark file of NAS-Bench-102 can be downloaded from [Google Drive](https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs) or [Baidu-Wangpan (code:6u5d)](https://pan.baidu.com/s/1CiaNH6C12zuZf7q-Ilm09w).
You can move it to anywhere you want and send its path to our API for initialization.
- v1.0: `NAS-Bench-102-v1_0-e61699.pth`, where `e61699` is the last six digits for this file. It contains all information except for the trained weights of each trial.
- v1.0: The full data of each architecture can be download from [Google Drive](https://drive.google.com/open?id=1X2i-JXaElsnVLuGgM4tP-yNwtsspXgdQ) (about 226GB). This compressed folder has 15625 files containing the the trained weights.
- v1.0: Checkpoints for 3 runs of each baseline NAS algorithm are provided in [Google Drive](https://drive.google.com/open?id=1eAgLZQAViP3r6dA0_ZOOGG9zPLXhGwXi).

The training and evaluation data used in NAS-Bench-102 can be downloaded from [Google Drive](https://drive.google.com/open?id=1L0Lzq8rWpZLPfiQGd6QR8q5xLV88emU7) or [Baidu-Wangpan (code:4fg7)](https://pan.baidu.com/s/1XAzavPKq3zcat1yBA1L2tQ).
It is recommended to put these data into `$TORCH_HOME` (`~/.torch/` by default). If you want to generate NAS-Bench-102 or similar NAS datasets or training models by yourself, you need these data.

## How to Use NAS-Bench-102

1. Creating an API instance from a file:
```
from nas_102_api import NASBench102API as API
api = API('$path_to_meta_nas_bench_file')
api = API('NAS-Bench-102-v1_0-e61699.pth')
api = API('{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-102-v1_0-e61699.pth'))
```

2. Show the number of architectures `len(api)` and each architecture `api[i]`:
```
num = len(api)
for i, arch_str in enumerate(api):
  print ('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))
```

3. Show the results of all trials for a single architecture:
```
# show all information for a specific architecture
api.show(1)
api.show(2)

# show the mean loss and accuracy of an architecture
info = api.query_meta_info_by_index(1)  # This is an instance of `ArchResults`
res_metrics = info.get_metrics('cifar10', 'train') # This is a dict with metric names as keys
cost_metrics = info.get_comput_costs('cifar100') # This is a dict with metric names as keys, e.g., flops, params, latency

# get the detailed information
results = api.query_by_index(1, 'cifar100') # a dict of all trials for 1st net on cifar100, where the key is the seed
print ('There are {:} trials for this architecture [{:}] on cifar100'.format(len(results), api[1]))
print ('Latency : {:}'.format(results[0].get_latency()))
print ('Train Info : {:}'.format(results[0].get_train()))
print ('Valid Info : {:}'.format(results[0].get_eval('x-valid')))
print ('Test  Info : {:}'.format(results[0].get_eval('x-test')))
# for the metric after a specific epoch
print ('Train Info [10-th epoch] : {:}'.format(results[0].get_train(10)))
```

4. Query the index of an architecture by string
```
index = api.query_index_by_arch('|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|')
api.show(index)
```

5. For other usages, please see `lib/nas_102_api/api.py`


### Detailed Instruction

In `nas_102_api`, we define three classes: `NASBench102API`, `ArchResults`, `ResultsCount`.

`ResultsCount` maintains all information of a specific trial. One can instantiate ResultsCount and get the info via the following codes (`000157-FULL.pth` saves all information of all trials of 157-th architecture):
```
from nas_102_api import ResultsCount
xdata  = torch.load('000157-FULL.pth')
odata  = xdata['full']['all_results'][('cifar10-valid', 777)]
result = ResultsCount.create_from_state_dict( odata )
print(result) # print it
print(result.get_train())   # print the final training loss/accuracy/[optional:time-cost-of-a-training-epoch]
print(result.get_train(11)) # print the training info of the 11-th epoch
print(result.get_eval('x-valid'))     # print the final evaluation info on the validation set
print(result.get_eval('x-valid', 11)) # print the info on the validation set of the 11-th epoch
print(result.get_latency())           # print the evaluation latency [in batch]
result.get_net_param()                # the trained parameters of this trial
arch_config = result.get_config(CellStructure.str2structure) # create the network with params
net_config  = dict2config(arch_config, None)
network    = get_cell_based_tiny_net(net_config)
network.load_state_dict(result.get_net_param())
```

`ArchResults` maintains all information of all trials of an architecture. Please see the following usages:
```
from nas_102_api import ArchResults
xdata   = torch.load('000157-FULL.pth')
archRes = ArchResults.create_from_state_dict(xdata['less']) # load trials trained with  12 epochs
archRes = ArchResults.create_from_state_dict(xdata['full']) # load trials trained with 200 epochs

print(archRes.arch_idx_str())      # print the index of this architecture 
print(archRes.get_dataset_names()) # print the supported training data
print(archRes.get_comput_costs('cifar10-valid')) # print all computational info when training on cifar10-valid 
print(archRes.get_metrics('cifar10-valid', 'x-valid', None, False)) # print the average loss/accuracy/time on all trials
print(archRes.get_metrics('cifar10-valid', 'x-valid', None,  True)) # print loss/accuracy/time of a randomly selected trial
```

`NASBench102API` is the topest level api. Please see the following usages:
```
from nas_102_api import NASBench102API as API
api = API('NAS-Bench-102-v1_0-e61699.pth') # This will load all the information of NAS-Bench-102 except the trained weights
api = API('{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-102-v1_0-e61699.pth')) # The same as the above line while I usually save NAS-Bench-102-v1_0-e61699.pth in ~/.torch/.
api.show(-1)  # show info of all architectures
api.reload('{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-BENCH-102-4-v1.0-archive'), 3) # This code will reload the information 3-th architecture with the trained weights

weights = api.get_net_param(3, 'cifar10', None) # Obtaining the weights of all trials for the 3-th architecture on cifar10. It will returns a dict, where the key is the seed and the value is the trained weights.
```


## Instruction to Re-Generate NAS-Bench-102

1. generate the meta file for NAS-Bench-102 using the following script, where `NAS-BENCH-102` indicates the name and `4` indicates the maximum number of nodes in a cell.
```
bash scripts-search/NAS-Bench-102/meta-gen.sh NAS-BENCH-102 4
```

2. train earch architecture on a single GPU (see commands in `output/NAS-BENCH-102-4/BENCH-102-N4.opt-full.script`, which is automatically generated by step-1).
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/NAS-Bench-102/train-models.sh 0     0   389 -1 '777 888 999'
```
This command will train 390 architectures (id from 0 to 389) using the following four kinds of splits with three random seeds (777, 888, 999).

|     Dataset     |     Train     |      Eval    |
|:---------------:|:-------------:|:------------:|
| CIFAR-10        | train         | valid / test |
| CIFAR-10        | train + valid | test         |
| CIFAR-100       | train         | valid / test |
| ImageNet-16-120 | train         | valid / test |

Note that the above `train`, `valid`, and `test` indicate the proposed splits in our NAS-Bench-102, and they might be different with the original splits.

3. calculate the latency, merge the results of all architectures, and simplify the results.
(see commands in `output/NAS-BENCH-102-4/meta-node-4.cal-script.txt` which is automatically generated by step-1).
```
OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python exps/NAS-Bench-102/statistics.py --mode cal --target_dir 000000-000389-C16-N5
```

4. merge all results into a single file for NAS-Bench-102-API.
```
OMP_NUM_THREADS=4 python exps/NAS-Bench-102/statistics.py --mode merge
```
This command will generate a single file `output/NAS-BENCH-102-4/simplifies/C16-N5-final-infos.pth` contains all the data for NAS-Bench-102.
This generated file will serve as the input for our NAS-Bench-102 API.

[option] train a single architecture on a single GPU.
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/NAS-Bench-102/train-a-net.sh resnet 16 5
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/NAS-Bench-102/train-a-net.sh '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|' 16 5
```


## To Reproduce 10 Baseline NAS Algorithms in NAS-Bench-102

We have tried our best to implement each method. However, still, some algorithms might obtain non-optimal results since their hyper-parameters might not fit our NAS-Bench-102.
If researchers can provide better results with different hyper-parameters, we are happy to update results according to the new experimental results. We also welcome more NAS algorithms to test on our dataset and would include them accordingly.

**Note that** you need to prepare the training and test data as described in [Preparation and Download](#preparation-and-download)

- [1] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V1.sh cifar10 -1`, where `cifar10` can be replaced with `cifar100` or `ImageNet16-120`.
- [2] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V2.sh cifar10 -1`
- [3] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/GDAS.sh     cifar10 -1`
- [4] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/SETN.sh     cifar10 -1`
- [5] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/ENAS.sh     cifar10 -1`
- [6] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/RANDOM-NAS.sh cifar10 -1`
- [7] `bash ./scripts-search/algos/R-EA.sh -1`
- [8] `bash ./scripts-search/algos/Random.sh -1`
- [9] `bash ./scripts-search/algos/REINFORCE.sh -1`
- [10] `bash ./scripts-search/algos/BOHB.sh -1`



# Citation

If you find that NAS-Bench-102 helps your research, please consider citing it:
```
@inproceedings{dong2020nasbench102,
  title     = {NAS-Bench-102: Extending the Scope of Reproducible Neural Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  url       = {https://openreview.net/forum?id=HJxyZkBKDr},
  year      = {2020}
}
```
