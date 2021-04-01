# Git Commands

```
git clone --recurse-submodules git@github.com:D-X-Y/AutoDL-Projects.git

git submodule init
git submodule update
git pull orign main

git submodule update --remote --recursive
```

Pylint check for Q-lib:
```
python -m black __init__.py -l 120

pytest -W ignore::DeprecationWarning qlib/tests/test_all_pipeline.py
```


```
conda update --all
```

## [phillytools](https://phillytools.azurewebsites.net/master/get_started/2_installation.html)

```
conda create -n pt6 python=3.7

conda activate pt6

pip install -U phillytools --extra-index-url https://msrpypi.azurewebsites.net/stable/7e404de797f4e1eeca406c1739b00867 --extra-index-url https://azuremlsdktestpypi.azureedge.net/K8s-Compute/D58E86006C65
```
