# SPBCMI

This repository includes the implementation of "SPBCMI: Predicting circRNA-miRNA Interactions Utilizing Transformer-based RNA Sequential Learning and High-Order Proximity preserved Embedding" 

In this package, we provides resources including: source codes of the SPBCMI model, raw data utilized in our model.

## Table of Contents

- [Background](#background)
- [Environment setup](#Environment-setup)
- [Usage](#usage)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

This repository is prepared to assist readers of our SPBCMI model in successfully replicating the experimental results. We understand the importance of reproducibility in research, and thus, this repository aims to provide all the necessary resources and guidance to facilitate the replication process.

## 1. Environment setup
We recommend you to build a python virtual environment with [Anaconda](https://docs.anaconda.com/anaconda/) Also, please make sure you have at least one NVIDIA GPU.

#### 1.1 Create and activate a new virtual environment

```
conda create -n SPBCMI python=3.6
conda activate SPBCMI
```

#### 1.2 Install the package and other requirements

(Required)

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

git clone https://github.com/ntitatip/SPBCMI
cd SPBCMI
python3 -m pip install --editable .
python3 -m pip install -r requirements.txt
```
## 2. Usage

#### 2.1 Data processing
