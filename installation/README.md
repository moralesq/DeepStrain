
# Table of Contents 
    
- [Getting Started](#Getting-Started)
- [Pre-trained Models](#Pre-trained-Models)
- [Tutorial Jupyter Notebooks](#Tutorial-Jupyter-Notebooks)
  * [Replication of Paper Results](#Replication-of-Paper-Results)
    + [Global Strain](#Global-Strain)
      - [ACDC](https://github.com/moralesq/DeepStrain/blob/main/notebooks/2_replicate_paper_results_ACDC_global_strain_from_scratch.ipynb)
      - [CMAC](https://github.com/moralesq/DeepStrain/blob/main/notebooks/3_replicate_paper_results_CMAC_global_strain_from_scratch.ipynb)
  * [Training](#Training)
- [Automated Scripts](#Automated-Scripts)
  * [Segmentation](#Automated-Scripts)
  * [Motion Estimation](#Motion-Estimation)
  * [Strain Analysis](#Strain-Analysis)

## Getting Started 

Install the requirements: 

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

- Clone this repo:
```bash
git clone -b master --single-branch https://github.com/moralesq/DeepStrain.git
cd DeepStrain
```
- Install TensorFlow and dependencies from https://www.tensorflow.org/install
- Install python libraries   

For pip users:
```bash
bash ./scripts/install_pip.sh
```

## Pre-trained Models

- Download some test data in nifti format:
```bash
bash ./datasets/download_sample_dataset.sh
```
- Download the pre-trained models (i.e., carson and carmen):
```bash
bash ./pretrained_models/download_model.sh
```

## Tutorial Jupyter Notebooks

Our goal is to provide various Jupyter notebooks to illustrate the various ways our method can be used. Although we already provide automated scripts, these notebooks will be use useful to provide some flexibility and potentially lead to applications in other domains. 

### Replication of Paper Results

The first set of notebooks will focus on replicating some of the results reported in our paper. This will be particularly useful if others would like to propose an alternative method and would like to use DeepStrain for comparison. In addition, this serves as a first quality check step before using DeepStrain in more clinically-oriented research applications. 

#### Global Strain 

We start by first reproducing the global strain results for the public [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) and [CMAC](https://www.cardiacatlas.org/challenges/motion-tracking-challenge/) datasets. For the ACDC dataset we reported global end-systolic strain for healthy subjects in the training set, and global strain for the entire cardiac cycle for all subjects in the training dataset (see [notebook](https://github.com/moralesq/DeepStrain/blob/main/notebooks/2_replicate_paper_results_ACDC_global_strain_from_scratch.ipynb)). For the CMAC dataset we reported global end-systolic strain for healthy subjects (see [notebook](https://github.com/moralesq/DeepStrain/blob/main/notebooks/3_replicate_paper_results_CMAC_global_strain_from_scratch.ipynb).  


### Training 


## Automated Scripts

### Segmentation

- Generate segmentations with the model for 3D niftis:
```bash
bash ./scripts/test_segmentation.sh ./datasets/sample_nifti_3D NIFTI ./results/sample_nifti_3D
```
The test results will be saved to a nifti file here: `./results/sample_nifti_3D/`.

- Generate segmentations with the model for 4D (3D+time) niftis:
```bash
bash ./scripts/test_segmentation.sh ./datasets/sample_nifti_4D NIFTI ./results/sample_nifti_3D
```
The test results will be saved to a nifti file here: `./results/sample_nifti_4D/`.

### Motion Estimation

- Generate motion estimates with the model for 4D (3D+time) niftis. Note that for motion estimation only 4D is supported:
```bash
bash ./scripts/test_motion.sh ./datasets/sample_nifti_4D NIFTI ./results/sample_nifti_4D
```
The test results will be saved to a h5 file here: `./results/sample_nifti_4D/`.

- Generate both segmentation and motion estimates with the model for 4D (3D+time) niftis:
```bash
bash ./scripts/test_segmentation_motion.sh ./datasets/sample_nifti_4D NIFTI ./results/sample_nifti_4D
```
The test results will be saved to nifti and h5 files here: `./results/sample_nifti_4D/`.

### Strain Analysis

- After the segmentations and motion estimates have been generated, we can use both to calculate myocardial strain. Note that we're passing the output folder from the previous runs:
```bash
bash ./scripts/test_strain.sh ./results/sample_nifti_4D
```
The test results (4D radial and circumferential strain) will be saved to nifti files here: `./results/sample_nifti_4D/`.