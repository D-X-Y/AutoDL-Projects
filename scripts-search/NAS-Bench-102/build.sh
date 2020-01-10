#!/bin/bash
# bash scripts-search/NAS-Bench-102/build.sh
echo script name: $0
echo $# arguments
if [ "$#" -ne 0 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 0 parameters"
  exit 1
fi

save_dir=./output/nas_bench_102_package
echo "Prepare to build the package in ${save_dir}"
rm -rf ${save_dir}
mkdir -p ${save_dir}

#cp NAS-Bench-102.md ${save_dir}/README.md
sed '125,187d' NAS-Bench-102.md > ${save_dir}/README.md
cp LICENSE.md ${save_dir}/LICENSE.md
cp -r lib/nas_102_api ${save_dir}/
rm -rf ${save_dir}/nas_102_api/__pycache__
cp exps/NAS-Bench-102/dist-setup.py ${save_dir}/setup.py

cd ${save_dir}
# python setup.py sdist bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
