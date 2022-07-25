<img src='imgs/landmarks.gif' align="right" width=440>

<br><br><br><br>

# DeepStrain

<img src="imgs/Fig_1.png" width="800">

**DeepStrain: A Deep Learning Workflow for the Automated Characterization of Cardiac Mechanics.**  

Tensorflow implementation for cardiac segmentation, motion estimation, and strain analysis from balanced steady-state free-precession (bSSFP) cine MRI images.
**Note**: The current software works well with Tensorflow 2.3.1+.

# Evaluation

<p align="center">
    DeepStrain Tracking in Patients Across MRI Vendors
    <br>
    <img src='imgs/DeepStrain_vs_CVI_Vid3.gif' width=440>
    <br>
    <p align="justify">
    Short-axis bSSFP cine MRI images are shown at the mid-ventricle. To visualize tracking, myocardial contours of the endocardial (red) and epicardial (green) left ventricular wall defined at end diastole were deformed to end systole using displacement vectors based on DeepStrain. (a) 64-year-old female with prior myocardial infarction. (b) 54-year-old male with prior myocardial infarction and ventricular tachycardia. (c) 69-year-old male with prior myocardial infarction, ventricular tachycardia, and dilated cardiomyopathy. (d) 19-year-old male with hypertrophic cardiomyopathy. (a-b) Vendor 1 = 1.5T (Achieva; Philips Medical Systems, Best, the Netherlands). (c-d) Vendor 2 = 3T (MAGNETOM Vida; Siemens Healthcare, Erlangen, Germany). 
    </p>
    <br><br><br><br>
    Evaluation Relative to Feature Tracking in Patients
    <br>
    <img src='imgs/DeepStrain_vs_CVI_figure_4.png' width=840>
    <br><br><br><br>
| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |
</p>

| Command | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |
## Publications

If you find DeepStrain or some part of the code useful, please cite as appropiate:

- **DeepStrain: A Deep Learning Workflow for the Automated Characterization of Cardiac Mechanics.** [Manuel A. Morales](https://catanalab.martinos.org/lab-members/manuel-a-morales/), [Maaike van den Boomen](https://nguyenlab.mgh.harvard.edu/maaike-van-den-boomen-ms/), [Christopher Nguyen](https://nguyenlab.mgh.harvard.edu/christopher-nguyen-phd-2/), [Jayashree Kalpathy-Cramer](https://www.ccds.io/leadership-team/jayashree-kalpathy-cramer/), [Bruce R. Rosen](https://www.martinos.org/investigator/bruce-rosen/), [Collin M. Stultz](https://mitibmwatsonailab.mit.edu/people/collin-m-stultz/), [David Izquierdo-Garcia](https://catanalab.martinos.org/lab-members/david-izquierdo-garcia/),  [Ciprian Catana](https://catanalab.martinos.org/lab-members/ciprian-catana/). Frontiers in Cardiovascular Medicine, 2021. DOI: https://doi.org/10.3389/fcvm.2021.730316.

- **DeepStrain Evidence of Asymptomatic Left Ventricular Diastolic and Systolic Dysfunction in Young Adults With Cardiac Risk Factors.** [Manuel A. Morales](https://catanalab.martinos.org/lab-members/manuel-a-morales/), Gert J. H. Snel, [Maaike van den Boomen](https://nguyenlab.mgh.harvard.edu/maaike-van-den-boomen-ms/), Ronald J. H. Borra, Vincent M. van Deursen, Riemer H. J. A. Slart, [David Izquierdo-Garcia](https://catanalab.martinos.org/lab-members/david-izquierdo-garcia/), Niek H. J. Prakken,  [Ciprian Catana](https://catanalab.martinos.org/lab-members/ciprian-catana/). Frontiers in Cardiovascular Medicine, 2022. DOI: https://doi.org/10.3389/fcvm.2022.831080

- **Comparison of DeepStrain and Feature Tracking for Cardiac MRI Strain Analysis.** [Manuel A. Morales](https://cardiacmr.hms.harvard.edu/people/manuel-morales-phd), [Julia Cirillo](https://cardiacmr.hms.harvard.edu/people/julia-cirillo), [Kei Nakata](https://cardiacmr.hms.harvard.edu/people/kei-nakata-md-phd), [Selcuk Kucukseymen](https://cardiacmr.hms.harvard.edu/people/selcuk-kucukseymen-md), [Long H. Ngo](https://www.bidmc.org/research/research-by-department/medicine/general-medicine-research/research-faculty/long-h-ngo-phd), [David Izquierdo-Garcia](https://catanalab.martinos.org/lab-members/david-izquierdo-garcia/),  [Ciprian Catana](https://catanalab.martinos.org/lab-members/ciprian-catana/), [Reza Nezafat](https://cardiacmr.hms.harvard.edu/people/reza-nezafat). Journal of Magnetic Resonance Imaging, 2022. DOI:
