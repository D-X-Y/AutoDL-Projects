# NAS Algorithms evaluated in [NATS-Bench](https://arxiv.org/abs/2009.00437)

The Python files in this folder are used to re-produce the results in ``NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size''.

- [`search-size.py`](https://github.com/D-X-Y/AutoDL-Projects/blob/main/exps/NATS-algos/search-size.py) contains codes for weight-sharing-based search on the size search space.
- [`search-cell.py`](https://github.com/D-X-Y/AutoDL-Projects/blob/main/exps/NATS-algos/search-cell.py) contains codes for weight-sharing-based search on the topology search space.
- [`bohb.py`](https://github.com/D-X-Y/AutoDL-Projects/blob/main/exps/NATS-algos/bohb.py) contains the BOHB algorithm for both size and topology search spaces.
- [`random_wo_share.py`](https://github.com/D-X-Y/AutoDL-Projects/blob/main/exps/NATS-algos/random_wo_share.py) contains the random search algorithm for both search spaces.
- [`regularized_ea.py`](https://github.com/D-X-Y/AutoDL-Projects/blob/main/exps/NATS-algos/regularized_ea.py) contains the REA algorithm for both search spaces.
- [`reinforce.py`](https://github.com/D-X-Y/AutoDL-Projects/blob/main/exps/NATS-algos/reinforce.py) contains the REINFORCE algorithm for both search spaces.

## Requirements

- `nats_bench`>=v1.2 : you can use `pip install nats_bench` to install or from [sources](https://github.com/D-X-Y/NATS-Bench)
- `hpbandster` : if you want to run BOHB

## Citation

If you find that this project helps your research, please consider citing the related paper:
```
@article{dong2021nats,
  title   = {{NATS-Bench}: Benchmarking NAS Algorithms for Architecture Topology and Size},
  author  = {Dong, Xuanyi and Liu, Lu and Musial, Katarzyna and Gabrys, Bogdan},
  doi     = {10.1109/TPAMI.2021.3054824},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year    = {2021},
  note    = {\mbox{doi}:\url{10.1109/TPAMI.2021.3054824}}
}
```
