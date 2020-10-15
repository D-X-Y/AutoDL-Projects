<p align="center">
<img src="https://xuanyidong.com/resources/images/AutoDL-log.png" width="400"/>
</p>

---------
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

自动深度学习库 (AutoDL-Projects) 是一个开源的，轻量级的，功能强大的项目。
台项目目前实现了多种网络结构搜索(NAS)和超参数优化(HPO)算法。

**谁应该考虑使用AutoDL-Projects**

- 想尝试不同AutoDL算法的初学者
- 想调研AutoDL在特定问题上的有效性的工程师
- 想轻松实现和实验新AutoDL算法的研究员

**为什么我们要用AutoDL-Projects**
- 最简化的python依赖库
- 所有算法都在一个代码库下
- 积极地维护


## AutoDL-Projects 能力简述

目前，该项目提供了下列算法和以及对应的运行脚本。请点击每个算法对应的链接看他们的细节描述。


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
    <td align="center" valign="middle"> <a href="https://github.com/D-X-Y/AutoDL-Projects/tree/master/docs/NeurIPS-2019-TAS.md">NeurIPS-2019-TAS.md</a> </td>
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
    <td align="center" valign="middle"> NATS-Bench </td>
    <td align="center" valign="middle"> <a href="https://xuanyidong.com/assets/projects/NATS-Bench"> NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size</a> </td>
    <td align="center" valign="middle"> <a href="https://github.com/D-X-Y/AutoDL-Projects/tree/master/docs/NATS-Bench.md">NATS-Bench.md</a> </td>
    </tr>
    <tr> <!-- (7-th row) -->
    <td align="center" valign="middle"> ... </td>
    <td align="center" valign="middle"> ENAS / REA / REINFORCE / BOHB </td>
    <td align="center" valign="middle"> Please check the original papers. </td>
    <td align="center" valign="middle"> <a href="https://github.com/D-X-Y/AutoDL-Projects/tree/master/docs/NAS-Bench-201.md">NAS-Bench-201.md</a> <a href="https://github.com/D-X-Y/AutoDL-Projects/tree/master/docs/NATS-Bench.md">NATS-Bench.md</a> </td>
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


## 准备工作

Please install `Python>=3.6` and `PyTorch>=1.3.0`. (You could also run this project in lower versions of Python and PyTorch, but may have bugs).
Some visualization codes may require `opencv`.

CIFAR and ImageNet should be downloaded and extracted into `$TORCH_HOME`.
Some methods use knowledge distillation (KD), which require pre-trained models. Please download these models from [Google Drive](https://drive.google.com/open?id=1ANmiYEGX-IQZTfH8w0aSpj-Wypg-0DR-) (or train by yourself) and save into `.latent-data`.

## 引用

如果您发现该项目对您的科研或工程有帮助，请考虑引用下列的某些文献：
```
@article{dong2020nats,
  title={{NATS-Bench}: Benchmarking NAS Algorithms for Architecture Topology and Size},
  author={Dong, Xuanyi and Liu, Lu and Musial, Katarzyna and Gabrys, Bogdan},
  journal={arXiv preprint arXiv:2009.00437},
  year={2020}
}
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
  pages     = {760--771},
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

# 其他

如果你想要给这份代码库做贡献，请看[CONTRIBUTING.md](.github/CONTRIBUTING.md)。
此外，使用规范请参考[CODE-OF-CONDUCT.md](.github/CODE-OF-CONDUCT.md)。

# 许可证
The entire codebase is under [MIT license](LICENSE.md)
