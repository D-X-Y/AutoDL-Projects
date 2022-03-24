#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.05 #
#####################################################
"""The setup function for pypi."""
# The following is to make nats_bench avaliable on Python Package Index (PyPI)
#
# conda install -c conda-forge twine  # Use twine to upload nats_bench to pypi
#
# python setup.py sdist bdist_wheel
# python setup.py --help-commands
# twine check dist/*
#
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
# https://pypi.org/project/xautodl
#
# TODO(xuanyidong): upload it to conda
#
# [2021.06.01] v0.9.9
# [2021.08.14] v1.0.0
# 
import os
from setuptools import setup, find_packages

NAME = "xautodl"
REQUIRES_PYTHON = ">=3.6"
DESCRIPTION = "Automated Deep Learning Package"

VERSION = "1.0.0"


def read(fname="README.md"):
    with open(
        os.path.join(os.path.dirname(__file__), fname), encoding="utf-8"
    ) as cfile:
        return cfile.read()


# What packages are required for this module to be executed?
REQUIRED = ["numpy>=1.16.5", "pyyaml>=5.0.0", "fvcore"]

packages = find_packages(
    exclude=("tests", "scripts", "scripts-search", "lib*", "exps*")
)
print("packages: {:}".format(packages))

setup(
    name=NAME,
    version=VERSION,
    author="Xuanyi Dong",
    author_email="dongxuanyi888@gmail.com",
    description=DESCRIPTION,
    license="MIT Licence",
    keywords="NAS Dataset API DeepLearning",
    url="https://github.com/D-X-Y/AutoDL-Projects",
    packages=packages,
    install_requires=REQUIRED,
    python_requires=REQUIRES_PYTHON,
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
