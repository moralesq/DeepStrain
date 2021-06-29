<img src='imgs/landmarks.gif' align="right" width=440>

<br><br><br><br>

# DeepStrain
[Paper](https://www.biorxiv.org/content/10.1101/2021.01.05.425266v1)


Tensorflow implementation for cardiac segmentation, motion estimation, and strain analysis from cinematic magnetic resonance imaging (cine-MRI) data. For example,  given a 4D (3D+time) nifti dataset, our model is able to segmentations, motion estimates, and global measures of myocardial strain.

**Note**: The current software works well with Tensorflow 2.3.1+.

<img src="imgs/Fig_1.png" width="800">

**DeepStrain: A Deep Learning Workflow for the Automated Characterization of Cardiac Mechanics.**  

[Manuel A. Morales](https://catanalab.martinos.org/lab-members/manuel-a-morales/), [Maaike van den Boomen](https://nguyenlab.mgh.harvard.edu/maaike-van-den-boomen-ms/), [Christopher Nguyen](https://nguyenlab.mgh.harvard.edu/christopher-nguyen-phd-2/), [Jayashree Kalpathy-Cramer](https://www.ccds.io/leadership-team/jayashree-kalpathy-cramer/), [Bruce R. Rosen](https://www.martinos.org/investigator/bruce-rosen/), [Collin M. Stultz](https://mitibmwatsonailab.mit.edu/people/collin-m-stultz/), [David Izquierdo-Garcia](https://catanalab.martinos.org/lab-members/david-izquierdo-garcia/),  [Ciprian Catana](https://catanalab.martinos.org/lab-members/ciprian-catana/)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started ###
### Installation
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

### Use a Pre-trained Model
- Download some test data in nifti format:
```bash
bash ./datasets/download_sample_dataset.sh
```
- Download the pre-trained models (i.e., carson and carmen):
```bash
bash ./pretrained_models/download_model.sh
```
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

- After the segmentations and motion estimates have been generated, we can use both to calculate myocardial strain. Note that we're passing the output folder from the previous runs:
```bash
bash ./scripts/test_strain.sh ./results/sample_nifti_4D
```
The test results (4D radial and circumferential strain) will be saved to nifti files here: `./results/sample_nifti_4D/`.

We are actively working on: 

- support for dicom inputs 
- support for regional analyses 
- optimizing implementation
