# Installation 

## Table of Contents 
    
- [Getting Started](#Getting-Started)
- [Pre-trained Models](#Pre-trained-Models)
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