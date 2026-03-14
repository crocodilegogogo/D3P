# $Distribution-aware Dynamic Data Pruning for Deep Learning$

PyTorch implementation of our paper ‘Distribution-aware Dynamic Data Pruning for Deep Learning’.

---

## Introduction

$\mathrm{D}^3\mathrm{P}$ is a plug-and-play dynamic data pruning module. It can be seamlessly integrated into existing training pipelines with minimal modification.

In this repository, we demonstrate $\mathrm{D}^3\mathrm{P}$ by inserting it into the pre-training stage of Masked Autoencoders (MAE).  
Under the original MAE pipeline, $\mathrm{D}^3\mathrm{P}$ can be enabled by simply inserting the module into the pre-training process.

---

## Overview

![](Figs/overview.jpg)

---

## Installation

Please follow the environment setup of the original [MAE](https://github.com/facebookresearch/mae) repository.

Additionally, install the following dependencies:

```bash
pip install scipy tensorboard tqdm numpy==1.23.5
```

---

## Train

### 1. Prepare training data

1.1 1.1 Download the [ImageNet-1K](https://www.image-net.org/) dataset, ensuring that the extracted directory contains the standard `train/` and `val/` subdirectories.

1.2 In `train_d3p.sh`, replace `/path/to/data` with the absolute path to your local ImageNet-1K dataset.

### 2. Training

Run MAE pre-training with $\mathrm{D}^3\mathrm{P}$, followed by the downstream finetuning task:

```bash
./train_d3p.sh
```

---

## Acknowledgement

This repository is heavily built upon the amazing works [MAE](https://github.com/facebookresearch/mae). Thanks for their great effort to community.

