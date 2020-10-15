# [NAS-BENCH-201: Extending the Scope of Reproducible Neural Architecture Search](https://openreview.net/forum?id=HJxyZkBKDr)

**Since our NAS-BENCH-201 has been extended to NATS-Bench, this `README` is deprecated and not maintained. Please use [NATS-Bench](https://github.com/D-X-Y/AutoDL-Projects/blob/master/docs/NATS-Bench.md), which has 5x more architecture information and faster API than NAS-BENCH-201.**

We propose an algorithm-agnostic NAS benchmark (NAS-Bench-201) with a fixed search space, which provides a unified benchmark for almost any up-to-date NAS algorithms.
The design of our search space is inspired by that used in the most popular cell-based searching algorithms, where a cell is represented as a directed acyclic graph.
Each edge here is associated with an operation selected from a predefined operation set.
For it to be applicable for all NAS algorithms, the search space defined in NAS-Bench-201 includes 4 nodes and 5 associated operation options, which generates 15,625 neural cell candidates in total.

In this Markdown file, we provide:
- [How to Use NAS-Bench-201](#how-to-use-nas-bench-201)

For the following two things, please use [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects):
- [Instruction to re-generate NAS-Bench-201](#instruction-to-re-generate-nas-bench-201)
- [10 NAS algorithms evaluated in our paper](#to-reproduce-10-baseline-nas-algorithms-in-nas-bench-201)

Note: please use `PyTorch >= 1.2.0` and `Python >= 3.6.0`.

You can simply type `pip install nas-bench-201` to install our api. Please see source codes of `nas-bench-201` module in [this repo](https://github.com/D-X-Y/NAS-Bench-201).

**If you have any questions or issues, please post it at [here](https://github.com/D-X-Y/AutoDL-Projects/issues) or email me.**

### Preparation and Download

[deprecated] The **old** benchmark file of NAS-Bench-201 can be downloaded from [Google Drive](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view?usp=sharing) or [Baidu-Wangpan (code:6u5d)](https://pan.baidu.com/s/1CiaNH6C12zuZf7q-Ilm09w).

[recommended] The **latest** benchmark file of NAS-Bench-201 (`NAS-Bench-201-v1_1-096897.pth`) can be downloaded from [Google Drive](https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view?usp=sharing). The files for model weight are too large (431G) and I need some time to upload it. Please be patient, thanks for your understanding.

You can move it to anywhere you want and send its path to our API for initialization.
- [2020.02.25] APIv1.0/FILEv1.0: [`NAS-Bench-201-v1_0-e61699.pth`](https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs) (2.2G), where `e61699` is the last six digits for this file. It contains all information except for the trained weights of each trial.
- [2020.02.25] APIv1.0/FILEv1.0: The full data of each architecture can be download from [
NAS-BENCH-201-4-v1.0-archive.tar](https://drive.google.com/open?id=1X2i-JXaElsnVLuGgM4tP-yNwtsspXgdQ) (about 226GB). This compressed folder has 15625 files containing the the trained weights.
- [2020.02.25] APIv1.0/FILEv1.0: Checkpoints for 3 runs of each baseline NAS algorithm are provided in [Google Drive](https://drive.google.com/open?id=1eAgLZQAViP3r6dA0_ZOOGG9zPLXhGwXi).
- [2020.03.09] APIv1.2/FILEv1.0: More robust API with more functions and descriptions
- [2020.03.16] APIv1.3/FILEv1.1: [`NAS-Bench-201-v1_1-096897.pth`](https://drive.google.com/open?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_) (4.7G), where `096897` is the last six digits for this file. It contains information of more trials compared to `NAS-Bench-201-v1_0-e61699.pth`, especially all models trained by 12 epochs on all datasets are avaliable.
- [2020.06.30] APIv2.0: Use abstract class (NASBenchMetaAPI) for APIs of NAS-Bench-x0y.
- [2020.06.30] FILEv2.0: coming soon!

**We recommend to use `NAS-Bench-201-v1_1-096897.pth`**


The training and evaluation data used in NAS-Bench-201 can be downloaded from [Google Drive](https://drive.google.com/open?id=1L0Lzq8rWpZLPfiQGd6QR8q5xLV88emU7) or [Baidu-Wangpan (code:4fg7)](https://pan.baidu.com/s/1XAzavPKq3zcat1yBA1L2tQ).
It is recommended to put these data into `$TORCH_HOME` (`~/.torch/` by default). If you want to generate NAS-Bench-201 or similar NAS datasets or training models by yourself, you need these data.

## How to Use NAS-Bench-201

**More usage can be found in [our test codes](https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/NAS-Bench-201/test-nas-api.py)**.

1. Creating an API instance from a file:
```
from nas_201_api import NASBench201API as API
api = API('$path_to_meta_nas_bench_file')
# Create an API without the verbose log
api = API('NAS-Bench-201-v1_1-096897.pth', verbose=False)
# The default path for benchmark file is '{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_1-096897.pth')
api = API(None)
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
cost_metrics = info.get_compute_costs('cifar100') # This is a dict with metric names as keys, e.g., flops, params, latency

# get the detailed information
results = api.query_by_index(1, 'cifar100') # a dict of all trials for 1st net on cifar100, where the key is the seed
print ('There are {:} trials for this architecture [{:}] on cifar100'.format(len(results), api[1]))
for seed, result in results.items():
  print ('Latency : {:}'.format(result.get_latency()))
  print ('Train Info : {:}'.format(result.get_train()))
  print ('Valid Info : {:}'.format(result.get_eval('x-valid')))
  print ('Test  Info : {:}'.format(result.get_eval('x-test')))
  # for the metric after a specific epoch
  print ('Train Info [10-th epoch] : {:}'.format(result.get_train(10)))
```

4. Query the index of an architecture by string
```
index = api.query_index_by_arch('|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|')
api.show(index)
```
This string `|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|` means:
```
node-0: the input tensor
node-1: conv-3x3( node-0 )
node-2: conv-3x3( node-0 ) + avg-pool-3x3( node-1 )
node-3: skip-connect( node-0 ) + conv-3x3( node-1 ) + skip-connect( node-2 )
```

5. Create the network from api:
```
config = api.get_net_config(123, 'cifar10') # obtain the network configuration for the 123-th architecture on the CIFAR-10 dataset
from models import get_cell_based_tiny_net # this module is in AutoDL-Projects/lib/models
network = get_cell_based_tiny_net(config) # create the network from configurration
print(network) # show the structure of this architecture
```
If you want to load the trained weights of this created network, you need to use `api.get_net_param(123, ...)` to obtain the weights and then load it to the network.

6. `api.get_more_info(...)` can return the loss / accuracy / time on training / validation / test sets, which is very helpful. For more details, please look at the comments in the get_more_info function.

7. For other usages, please see `lib/nas_201_api/api.py`. We provide some usage information in the comments for the corresponding functions. If what you want is not provided, please feel free to open an issue for discussion, and I am happy to answer any questions regarding NAS-Bench-201.


### Detailed Instruction

In `nas_201_api`, we define three classes: `NASBench201API`, `ArchResults`, `ResultsCount`.

`ResultsCount` maintains all information of a specific trial. One can instantiate ResultsCount and get the info via the following codes (`000157-FULL.pth` saves all information of all trials of 157-th architecture):
```
from nas_201_api import ResultsCount
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
from nas_201_api import ArchResults
xdata   = torch.load('000157-FULL.pth')
archRes = ArchResults.create_from_state_dict(xdata['less']) # load trials trained with  12 epochs
archRes = ArchResults.create_from_state_dict(xdata['full']) # load trials trained with 200 epochs

print(archRes.arch_idx_str())      # print the index of this architecture 
print(archRes.get_dataset_names()) # print the supported training data
print(archRes.get_comput_costs('cifar10-valid')) # print all computational info when training on cifar10-valid 
print(archRes.get_metrics('cifar10-valid', 'x-valid', None, False)) # print the average loss/accuracy/time on all trials
print(archRes.get_metrics('cifar10-valid', 'x-valid', None,  True)) # print loss/accuracy/time of a randomly selected trial
```

`NASBench201API` is the topest level api. Please see the following usages:
```
from nas_201_api import NASBench201API as API
api = API('NAS-Bench-201-v1_1-096897.pth') # This will load all the information of NAS-Bench-201 except the trained weights
api = API('{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-Bench-201-v1_1-096897.pth')) # The same as the above line while I usually save NAS-Bench-201-v1_1-096897.pth in ~/.torch/.
api.show(-1)  # show info of all architectures
api.reload('{:}/{:}'.format(os.environ['TORCH_HOME'], 'NAS-BENCH-201-4-v1.0-archive'), 3) # This code will reload the information 3-th architecture with the trained weights

weights = api.get_net_param(3, 'cifar10', None) # Obtaining the weights of all trials for the 3-th architecture on cifar10. It will returns a dict, where the key is the seed and the value is the trained weights.
```

To obtain the training and evaluation information (please see the comments [here](https://github.com/D-X-Y/AutoDL-Projects/blob/master/lib/nas_201_api/api_201.py#L142)):
```
api.get_more_info(112, 'cifar10', None, hp='200', is_random=True)
# Query info of last training epoch for 112-th architecture
# using 200-epoch-hyper-parameter and randomly select a trial.
api.get_more_info(112, 'ImageNet16-120', None, hp='200', is_random=True)
```

# Citation

If you find that NAS-Bench-201 helps your research, please consider citing it:
```
@inproceedings{dong2020nasbench201,
  title     = {{NAS-Bench-201}: Extending the Scope of Reproducible Neural Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  url       = {https://openreview.net/forum?id=HJxyZkBKDr},
  year      = {2020}
}
```
