#!/bin/bash
# bash ./scripts/black.sh

# script=$(readlink -f "$0")
# scriptpath=$(dirname "$script")
# echo $scriptpath

# delete Python cache files
find . | grep -E "(__pycache__|\.pyc|\.DS_Store|\.pyo$)" | xargs rm -rf

black ./tests/
black ./xautodl/procedures
black ./xautodl/datasets
black ./xautodl/xlayers
black ./exps/trading
rm -rf ./xautodl.egg-info
rm -rf ./build
rm -rf ./dist
rm -rf ./.pytest_cache
