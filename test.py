import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from options.test_options import TestOptions
from models import deep_strain_model
from data import nifti_dataset

# options
opt = TestOptions().parse()
os.makedirs(opt.results_dir, exist_ok=True)

dataset = nifti_dataset.NiftiDataset(opt)
model   = deep_strain_model.DeepStrain(Adam, opt)

if 'segmentation' in opt.pipeline:

    netS    = model.get_netS()
    for i, data in enumerate(dataset):

        filename = os.path.basename(dataset.filenames[i]).split('.')[0]

        x, nifti, nifti_resampled = data

        y = netS(x).numpy()
        y = dataset.transform.apply_inv(y)
        nifti_dataset.save_as_nifti(y, nifti, nifti_resampled,
                                    filename=os.path.join(opt.results_dir, filename+'_segmentation'))
        

