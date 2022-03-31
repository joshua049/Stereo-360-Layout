# Stereo-360-Layout
This is the implementation of [our paper](https://arxiv.org/abs/2203.16057)

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
Besides, to make our custom dataset work, you have to copy the `room_shape_simplicity_labels.json` from [ZInD repo](https://github.com/zillow/zind) to the root of ZInD dataset on your device.

## Data Selection
For active data selection, just execute `data_selection.py`. The script will evaluate each sample with our proposed label-free metric with the pretrained weight provided in argument `--pth`. If the path of pretrained weight is not provided, data will be sampled randomly. The sample result will be recorded in the argument `--stored_file`.

## Training
- arguments
    - `--unsup_root_dir`: Root directory to unsupervised training dataset.
    - `--sup_root_dir`: Root directory to supervised training dataset. 
    - `--valid_root_dir`: Root directory to validation dataset.
    - `--sample_num`: Number of sampled data for supervised training.
    - `--sample_file`: The csv file for data selection (generated from `data_selection.py`).
    - `--eval_only`: evaluate on the valid dataset only.

### Examples
- self-supervised only
    - normal
    ```
    python train.py --id [yourid] --unsup_root_dir /path/to/unsup --valid_root_dir /path/to/valid
    ```
    - disable some losses
    ```
    python train.py --id [yourid] --unsup_root_dir /path/to/unsup --valid_root_dir /path/to/valid --no_[loss_to_disable]
    ```
- supervised only
    - normal (train from scratch)
    ```
    python train.py --id [yourid] --sup_root_dir /path/to/sup --valid_root_dir /path/to/valid 
    ```
    - finetune on your checkpoint
    ```
    python train.py --id [yourid] --sup_root_dir /path/to/sup --valid_root_dir /path/to/valid --pth /path/to/ckpt
    ```
    - finetune on your checkpoint with partial data
    ```
    python train.py --id [yourid] --sup_root_dir /path/to/sup --valid_root_dir /path/to/valid --pth /path/to/ckpt --sample_num [your_num] --sample_file /path/to/splits
    ```
- self-pretrained + finetune
```
python train.py --id [yourid] --sup_root_dir /path/to/sup --unsup_root_dir /path/to/unsup --valid_root_dir /path/to/valid
```
- evaluation
```
python train.py --id [yourid] --valid_root_dir /path/to/valid --eval_only
```

