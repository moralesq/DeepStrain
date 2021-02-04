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
- Install PyTorch and dependencies from https://www.tensorflow.org/
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

- Generate results with the model for 3D niftis (segmentations only for single time frame!):
```bash
bash ./scripts/test_sample_nifti_3D.sh
```
The test results will be saved to a html file here: `./results/sample_nifti_3D/`.

- Generate results with the model for 4D niftis (full pipeline):
```bash
bash ./scripts/test_sample_nifti_4D.sh
```
Results can be found at `./results/sample_nifti_4D/`.

We are actively working on: 

- support for dicom inputs 
- support for regional analyses 
