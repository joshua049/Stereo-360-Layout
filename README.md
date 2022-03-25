# Stereo-360-Layout
This is the implementation of [arxiv link]

## Overview
![](https://i.imgur.com/fOyeHXW.jpg)

## Installation
For PyTorch and PyTorch3d, please follow the instructions below:
```
conda create -n [yourname] python=3.9
conda activate [yourname]
conda install -c pytorch pytorch=1.9.1 torchvision cudatoolkit=10.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```
Other needed package are provided in `requirements.txt`, you can install them via PyPI.

## Data Preparation
For MatterportLayout, we follow the preparation on HorizonNet, please refer to [here](https://github.com/sunset1995/HorizonNet/blob/master/README_PREPARE_DATASET.md) for detailed.

For ZInD, You can preprocess it with our provided script `preprocess_zind.py`.

## Data Selection
For active data selection, just execute `data_selection.py`. The script will evaluate each sample with our proposed label-free metric with the pretrained weight provided in argument `--pth`. If the path of pretrained weight is not provided, data will be sampled randomly. The sample result will be recorded in the argument `--stored_file`.

## Training
```
python train.py --id [yourname] --valid_root_dir /path/to/valid
```
- arguments
    - `--unsup_root_dir`: Root directory to unsupervised training dataset.
    - `--sup_root_dir`: Root directory to supervised training dataset. 
    - `--valid_root_dir`: Root directory to validation dataset.
    - `--sample_num`: Number of sampled data for supervised training.
    - `--sample_file`: The csv file for data selection (generated from `data_selection.py`).
    - `--eval_only`: evaluate on the valid dataset only.