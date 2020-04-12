<p align="center">
<img src="https://xuanyidong.com/resources/images/AutoDL-log.png" width="400"/>
</p>

---------
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Automated Deep Learning (AutoDL-Projects) is an open source, lightweight, but useful project for everyone.
This project implemented several neural architecture search (NAS) and hyper-parameter optimization (HPO) algorithms.
中文介绍见[README_CN.md](README_CN.md)

**Who should consider using AutoDL-Projects**

- Beginners who want to **try different AutoDL algorithms**
- Engineers who want to **try AutoDL** to investigate whether AutoDL works on your projects
- Researchers who want to **easily** implement and experiement **new** AutoDL algorithms.

**Why should we use AutoDL-Projects**
- Simple library dependencies
- All algorithms are in the same codebase
- Active maintenance


## AutoDL-Projects Capabilities

At the moment, this project provides the following algorithms and scripts to run them. Please see the details in the link provided in the description column.

<table>
 <tbody>
    <tr align="center" valign="bottom">
      <th>Type</th>
      <th>ABBRV</th>
      <th>Algorithms</th>
      <th>Description</th>
    </tr>
    <tr> <!-- (1-st row) -->
    <td rowspan="6" align="center" valign="middle" halign="middle"> NAS </td>
    <td align="center" valign="middle"> TAS </td>
    <td align="center" valign="middle"> <a href="https://arxiv.org/abs/1905.09717">Network Pruning via Transformable Architecture Search</a> </td>
    <td align="center" valign="middle"> <a href="https://github.com/D-X-Y/AutoDL-Projects/tree/master/docs/NIPS-2019-TAS.md">NIPS-2019-TAS.md</a> </td>
    </tr>
    <tr> <!-- (2-nd row) -->
    <td align="center" valign="middle"> DARTS </td>
    <td align="center" valign="middle"> <a href="https://arxiv.org/abs/1806.09055">DARTS: Differentiable Architecture Search</a> </td>
    <td align="center" valign="middle"> <a href="https://github.com/D-X-Y/AutoDL-Projects/tree/master/docs/ICLR-2019-DARTS.md">ICLR-2019-DARTS.md</a> </td>
    </tr>
    <tr> <!-- (3-nd row) -->
    <td align="center" valign="middle"> GDAS </td>
    <td align="center" valign="middle"> <a href="https://arxiv.org/abs/1910.04465">Searching for A Robust Neural Architecture in Four GPU Hours</a> </td>
    <td align="center" valign="middle"> <a href="https://github.com/D-X-Y/AutoDL-Projects/tree/master/docs/CVPR-2019-GDAS.md">CVPR-2019-GDAS.md</a> </td>
    </tr>
    <tr> <!-- (4-rd row) -->
    <td align="center" valign="middle"> SETN </td>
    <td align="center" valign="middle"> <a href="https://arxiv.org/abs/1910.05733">One-Shot Neural Architecture Search via Self-Evaluated Template Network</a> </td>
    <td align="center" valign="middle"> <a href="https://github.com/D-X-Y/AutoDL-Projects/tree/master/docs/ICCV-2019-SETN.md">ICCV-2019-SETN.md</a> </td>
    </tr>
    <tr> <!-- (5-th row) -->
    <td align="center" valign="middle"> NAS-Bench-201 </td>
    <td align="center" valign="middle"> <a href="https://openreview.net/forum?id=HJxyZkBKDr"> NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search</a> </td>
    <td align="center" valign="middle"> <a href="https://github.com/D-X-Y/AutoDL-Projects/tree/master/docs/NAS-Bench-201.md">NAS-Bench-201.md</a> </td>
    </tr>
    <tr> <!-- (6-th row) -->
    <td align="center" valign="middle"> ... </td>
    <td align="center" valign="middle"> ENAS / REA / REINFORCE / BOHB </td>
    <td align="center" valign="middle"> <a href="https://github.com/D-X-Y/AutoDL-Projects/tree/master/docs/NAS-Bench-201.md">NAS-Bench-201.md</a> </td>
    </tr>
    <tr> <!-- (start second block) -->
    <td rowspan="1" align="center" valign="middle" halign="middle"> HPO </td>
    <td align="center" valign="middle"> HPO-CG </td>
    <td align="center" valign="middle"> Hyperparameter optimization with approximate gradient </td>
    <td align="center" valign="middle"> coming soon </a> </td>
    </tr>
    <tr> <!-- (start third block) -->
    <td rowspan="1" align="center" valign="middle" halign="middle"> Basic </td>
    <td align="center" valign="middle"> ResNet </td>
    <td align="center" valign="middle"> Deep Learning-based Image Classification </td>
    <td align="center" valign="middle"> <a href="https://github.com/D-X-Y/AutoDL-Projects/tree/master/docs/BASELINE.md">BASELINE.md</a> </a> </td>
    </tr>
 </tbody>
</table>



## Requirements and Preparation

Please install `Python>=3.6` and `PyTorch>=1.3.0`. (You could also run this project in lower versions of Python and PyTorch, but may have bugs).
Some visualization codes may require `opencv`.

CIFAR and ImageNet should be downloaded and extracted into `$TORCH_HOME`.
Some methods use knowledge distillation (KD), which require pre-trained models. Please download these models from [Google Drive](https://drive.google.com/open?id=1ANmiYEGX-IQZTfH8w0aSpj-Wypg-0DR-) (or train by yourself) and save into `.latent-data`.

## Citation

If you find that this project helps your research, please consider citing some of the following papers:
```
@inproceedings{dong2020nasbench201,
  title     = {NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  url       = {https://openreview.net/forum?id=HJxyZkBKDr},
  year      = {2020}
}
@inproceedings{dong2019tas,
  title     = {Network Pruning via Transformable Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}
@inproceedings{dong2019one,
  title     = {One-Shot Neural Architecture Search via Self-Evaluated Template Network},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages     = {3681--3690},
  year      = {2019}
}
@inproceedings{dong2019search,
  title     = {Searching for A Robust Neural Architecture in Four GPU Hours},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1761--1770},
  year      = {2019}
}
```

# Others

If you want to contribute to this repo, please see [CONTRIBUTING.md](.github/CONTRIBUTING.md).
Besides, please follow [CODE-OF-CONDUCT.md](.github/CODE-OF-CONDUCT.md).

# License
The entire codebase is under [MIT license](LICENSE.md)
