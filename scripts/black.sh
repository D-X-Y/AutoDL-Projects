#!/bin/bash
# bash ./scripts/black.sh

black ./tests/
black ./xautodl/procedures
black ./xautodl/datasets
black ./xautodl/xlayers
black ./exps/LFNA
black ./exps/trading
rm -rf ./xautodl.egg-info
rm -rf ./build
rm -rf ./dist
rm -rf ./.pytest_cache
