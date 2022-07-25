# Examples



## Table of Contents 
    
- [Tutorial Jupyter Notebooks](#Tutorial-Jupyter-Notebooks)
  * [Replication of Paper Results](#Replication-of-Paper-Results)
    + [Global Strain](#Global-Strain)
      - [ACDC](https://github.com/moralesq/DeepStrain/blob/main/examples/2_replicate_paper_results_ACDC_global_strain_from_scratch.ipynb)
      - [CMAC](https://github.com/moralesq/DeepStrain/blob/main/examples/3_replicate_paper_results_CMAC_global_strain_from_scratch.ipynb)


## Tutorial Jupyter Notebooks and Scripts

Our goal is to provide various Jupyter notebooks and scripts to illustrate the various ways our method can be used. Although we already provide automated scripts, these notebooks will be use useful to provide some flexibility and potentially lead to applications in other domains. 

### Replication of DeepStrain vs cvi42 Feature Tracking paper 

If you're interested in using DeepStrain for processsing of large databases, I recommend you take a close look at the [script](https://github.com/moralesq/DeepStrain/blob/main/examples/deepstrain_vs_cvi.py) we used to compare DeepStrain to cvi42 Feature tracking. 

### Replication of Paper Results

The first set of notebooks will focus on replicating some of the results reported in our paper. This will be particularly useful if others would like to propose an alternative method and would like to use DeepStrain for comparison. In addition, this serves as a first quality check step before using DeepStrain in more clinically-oriented research applications. 

#### Global Strain 

We start by first reproducing the global strain results for the public [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) and [CMAC](https://www.cardiacatlas.org/challenges/motion-tracking-challenge/) datasets. For the ACDC dataset we reported global end-systolic strain for healthy subjects in the training set, and global strain for the entire cardiac cycle for all subjects in the training dataset (see [notebook](https://github.com/moralesq/DeepStrain/blob/main/examples/2_replicate_paper_results_ACDC_global_strain_from_scratch.ipynb)). For the CMAC dataset we reported global end-systolic strain for healthy subjects (see [notebook](https://github.com/moralesq/DeepStrain/blob/main/examples/3_replicate_paper_results_CMAC_global_strain_from_scratch.ipynb).  
