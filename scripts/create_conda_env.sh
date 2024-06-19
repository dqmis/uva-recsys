#!/bin/bash

conda create -n recsys2024 python=3.10.14
source activate recsys2024
conda install pip

pip install \                       130 ↵
    --extra-index-url=https://pypi.nvidia.com \
    'dask-cuda==24.2.0' 'cudf-cu12==24.2.*' 'dask-cudf-cu12==24.2.*'
pip install git+https://github.com/NVIDIA/dllogger.git
pip install -r requirements.txt
