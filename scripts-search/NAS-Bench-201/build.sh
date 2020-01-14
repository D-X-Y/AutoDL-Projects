#!/bin/bash
# bash scripts-search/NAS-Bench-201/build.sh
echo script name: $0
echo $# arguments
if [ "$#" -ne 0 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 0 parameters"
  exit 1
fi

save_dir=./output/nas_bench_201_package
echo "Prepare to build the package in ${save_dir}"
rm -rf ${save_dir}
mkdir -p ${save_dir}

#cp NAS-Bench-201.md ${save_dir}/README.md
sed '125,187d' NAS-Bench-201.md > ${save_dir}/README.md
cp LICENSE.md ${save_dir}/LICENSE.md
cp -r lib/nas_201_api ${save_dir}/
rm -rf ${save_dir}/nas_201_api/__pycache__
cp exps/NAS-Bench-201/dist-setup.py ${save_dir}/setup.py

cd ${save_dir}
# python setup.py sdist bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
