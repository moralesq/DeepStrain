# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

import os
import h5py
import glob
import warnings
import numpy as np
import nibabel as nib
import pandas as pd

from dipy.align.reslice import reslice
from data.base_dataset import BaseDataset, Transforms
from data.image_folder import make_dataset


class H5PYDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.filenames = sorted(make_dataset(opt.dataroot, opt.max_dataset_size, 'H5PY'))
        
    def __len__(self):
        return len(self.filenames)
                
    def __getitem__(self, idx):      
        return self.loader_h5(self.filenames[idx])
    
    def loader_h5(self,filename):
        HF = h5py.File(filename, 'r')

        output = []
        for key in HF.keys():
            for subkey in HF[key].keys():
                output += [np.array(HF[key][subkey])]

        HF.close()
        return np.stack(output,-1)
