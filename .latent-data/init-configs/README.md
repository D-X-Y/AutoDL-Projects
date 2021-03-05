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
