# DeepStrain

<p align="center">
    Evaluated Relative to Tagging CMR
    <img src='imgs/landmarks.gif' width=640>
    <br><br><br><br>
    Evaluated in Patients Across MRI Vendors
    <img src='imgs/DeepStrain_vs_CVI_Vid3.gif' width=440>
    <br><br><br><br>
</p>




Tensorflow implementation for cardiac segmentation, motion estimation, and strain analysis from cinematic magnetic resonance imaging (cine-MRI) data. For example,  given a 4D (3D+time) nifti dataset, our model is able to provide segmentations, motion estimates, and global measures of myocardial strain.

**Note**: The current software works well with Tensorflow 2.3.1+.

<img src="imgs/Fig_1.png" width="800">

**DeepStrain: A Deep Learning Workflow for the Automated Characterization of Cardiac Mechanics.**  

[Manuel A. Morales](https://catanalab.martinos.org/lab-members/manuel-a-morales/), [Maaike van den Boomen](https://nguyenlab.mgh.harvard.edu/maaike-van-den-boomen-ms/), [Christopher Nguyen](https://nguyenlab.mgh.harvard.edu/christopher-nguyen-phd-2/), [Jayashree Kalpathy-Cramer](https://www.ccds.io/leadership-team/jayashree-kalpathy-cramer/), [Bruce R. Rosen](https://www.martinos.org/investigator/bruce-rosen/), [Collin M. Stultz](https://mitibmwatsonailab.mit.edu/people/collin-m-stultz/), [David Izquierdo-Garcia](https://catanalab.martinos.org/lab-members/david-izquierdo-garcia/),  [Ciprian Catana](https://catanalab.martinos.org/lab-members/ciprian-catana/)

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

## Publications

If you find DeepStrain or some part of the code useful, please cite as appropiate:

- **DeepStrain: A Deep Learning Workflow for the Automated Characterization of Cardiac Mechanics.** [Manuel A. Morales](https://catanalab.martinos.org/lab-members/manuel-a-morales/), [Maaike van den Boomen](https://nguyenlab.mgh.harvard.edu/maaike-van-den-boomen-ms/), [Christopher Nguyen](https://nguyenlab.mgh.harvard.edu/christopher-nguyen-phd-2/), [Jayashree Kalpathy-Cramer](https://www.ccds.io/leadership-team/jayashree-kalpathy-cramer/), [Bruce R. Rosen](https://www.martinos.org/investigator/bruce-rosen/), [Collin M. Stultz](https://mitibmwatsonailab.mit.edu/people/collin-m-stultz/), [David Izquierdo-Garcia](https://catanalab.martinos.org/lab-members/david-izquierdo-garcia/),  [Ciprian Catana](https://catanalab.martinos.org/lab-members/ciprian-catana/). Frontiers in Cardiovascular Medicine, 2021. DOI: https://doi.org/10.3389/fcvm.2021.730316.

- **DeepStrain Evidence of Asymptomatic Left Ventricular Diastolic and Systolic Dysfunction in Young Adults With Cardiac Risk Factors.** [Manuel A. Morales](https://catanalab.martinos.org/lab-members/manuel-a-morales/), Gert J. H. Snel, [Maaike van den Boomen](https://nguyenlab.mgh.harvard.edu/maaike-van-den-boomen-ms/), Ronald J. H. Borra, Vincent M. van Deursen, Riemer H. J. A. Slart, [David Izquierdo-Garcia](https://catanalab.martinos.org/lab-members/david-izquierdo-garcia/), Niek H. J. Prakken,  [Ciprian Catana](https://catanalab.martinos.org/lab-members/ciprian-catana/). Frontiers in Cardiovascular Medicine, 2022. DOI: https://doi.org/10.3389/fcvm.2022.831080

- **Comparison of DeepStrain and Feature Tracking for Cardiac MRI Strain Analysis.** [Manuel A. Morales](https://cardiacmr.hms.harvard.edu/people/manuel-morales-phd), [Julia Cirillo](https://cardiacmr.hms.harvard.edu/people/julia-cirillo), [Kei Nakata](https://cardiacmr.hms.harvard.edu/people/kei-nakata-md-phd), [Selcuk Kucukseymen](https://cardiacmr.hms.harvard.edu/people/selcuk-kucukseymen-md), [Long H. Ngo](https://www.bidmc.org/research/research-by-department/medicine/general-medicine-research/research-faculty/long-h-ngo-phd), [David Izquierdo-Garcia](https://catanalab.martinos.org/lab-members/david-izquierdo-garcia/),  [Ciprian Catana](https://catanalab.martinos.org/lab-members/ciprian-catana/), [Reza Nezafat](https://cardiacmr.hms.harvard.edu/people/reza-nezafat). Journal of Magnetic Resonance Imaging, 2022. DOI:
