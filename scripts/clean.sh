#!/bin/bash
find -name "__pycache__" | xargs rm -rf
find -name "._.DS_Store" | xargs rm -rf
find -name ".DS_Store"   | xargs rm -rf
rm -rf output
rm -rf ./scripts-cluster/tmps/job*
