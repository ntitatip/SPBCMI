# SPBCMI

This repository includes the implementation of "SPBCMI: Predicting circRNA-miRNA Interactions Utilizing Transformer-based RNA Sequential Learning and High-Order Proximity preserved Embedding" 

In this package, we provides resources including: source codes of the SPBCMI model, raw data utilized in our model.

## Table of Contents

- [Background](#background)
- [Environment setup](#Environment-setup)
- [Usage](#usage)
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

We have processed all our RNA sequence data according to the following procedure:
1. We have provided the necessary tools to convert all RNA sequences from the letter 'U' to 'T'. This change in representation does not alter the biological meaning of the sequences; it is merely a difference in notation.

2. Next, we have reverse transcribed all RNA sequences into DNA fragments using the appropriate tools and methods.

3. Subsequently, all sequences longer than 510 nucleotides have been cut and spliced according to the scheme "first 128 nucleotides + last 382 nucleotides." This process ensures uniformity in the data length for further analysis.

4. Additionally, we have added the special tokens [CLS] and [SEP] at the beginning and end of each processed sequence, respectively, to facilitate subsequent tasks.

You can find all the processed RNA sequence data in the folder "SPBCMI/1-kmer." These prepared datasets are now ready for use in various experiments and analyses. We also encourage researchers who have their own RNA sequence data to process it according to the procedure above.

#### 2.1 Fine-tuned model

Based on the [DNABERT](https://github.com/jerryji1993/DNABERT.git) model, we have conducted fine-tuning for k-mer values of k=1, 2, 3, and 4. Due to the large size of the models, we are unable to upload them here. Therefore, we kindly request you to download the fine-tuned models from the following link: [SPBCMI](https://drive.google.com/drive/folders/154LhzAD498l96Sua-tATNcI7y20l7kUV?usp=sharing).

Additionally, if you have your own data and wish to fine-tune the DNABERT model for specific tasks, we recommend downloading the original DNABERT model from its official source and proceeding with the fine-tuning process on your own data. Please ensure that you comply with the licensing and usage terms associated with the DNABERT model during the download and fine-tuning process.

#### 2.2 Comparative Experiment of Word Segementaion


The cloud drive link contains the files named "pairs.csv" and "allnode.csv," which we have organized. Please begin by copying these files into their respective folders and placing them together with the corresponding model. Next, run the "extract_feature.py" file. This process will generate and save the resulting files in this directory. Within all folders, you will find the feature files. Copy all of them into the "comparison" folder.  Once this is done, execute the program "comparison.py"

For the classification task, we have generated random negative samples to complement the dataset. The prediction results, including the AUC (Area Under the Curve) and AUPR (Area Under the Precision-Recall curve), will be saved in the "image" folder. Additionally, other evaluation metrics will be stored in the "metrics" folder.

Please ensure that you have the necessary dependencies and libraries installed for the classification task. The CSV files containing the extracted features will be used as inputs for the classification algorithm.

#### 2.3 Comparative Experiment of Different NLP Method

In this stage of the project, we conducted a comprehensive comparison of various sequece feature extraction methods. We have prepared multiple methods for feature extraction. To proceed, copy the "allnode.csv" file into the corresponding folders. Next, run the Python files within these folders. Upon completion, locate the generated CSV files in the directory. Copy these files into the "comparison" folder. By following these steps, you will obtain the comparative results, similar to what was demonstrated in the previous section. 
#### 2.4 Comparative Experiment of Different Structural Feature Extracting Method

We have prepared multiple methods for extracting structural features. To begin, copy the "pairs" file into the corresponding folders. Next, run the Python files contained within these folders. Once the execution is completed, follow the same procedure as described in the previous section to proceed further. By doing so, you will obtain the comparative results as illustrated earlier.

#### 2.5 Comparative Experiment of Utilizing Different Features

In this phase of the project, we aim to enhance our feature representation by combining the previously extracted features. We will concatenate the feature files obtained from different feature extraction methods and integrate them into this specific file. The remaining steps and methodologies will remain consistent with those used before.

#### 2.6 Comparative Experiment of Final Results

The final cross-validation results, representing the aggregated performance of the models, will be directly saved in the "5-final" directory. Within this directory, you will find comprehensive files containing the evaluation metrics and outcomes for each fold.

## 3. Contributing

Ji-Ren Zhou authored and maintains this code. If you encounter any issues or have questions, please feel free to contact the author at zhoujiren@mail.nwpu.edu.cn.

## Q & A
