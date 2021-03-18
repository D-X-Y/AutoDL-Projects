#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
# [2020.02.25] Initialize the API as v1.1
# [2020.03.09] Upgrade the API to v1.2
# [2020.03.16] Upgrade the API to v1.3
# [2020.06.30] Upgrade the API to v2.0
import os
from setuptools import setup


def read(fname="README.md"):
    with open(
        os.path.join(os.path.dirname(__file__), fname), encoding="utf-8"
    ) as cfile:
        return cfile.read()


setup(
    name="nas_bench_201",
    version="2.0",
    author="Xuanyi Dong",
    author_email="dongxuanyi888@gmail.com",
    description="API for NAS-Bench-201 (a benchmark for neural architecture search).",
    license="MIT",
    keywords="NAS Dataset API DeepLearning",
    url="https://github.com/D-X-Y/NAS-Bench-201",
    packages=["nas_201_api"],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
