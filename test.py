import os
import h5py
import timeit
import numpy as np
from tensorflow.keras.optimizers import Adam
from options.test_options import TestOptions
from models import deep_strain_model
from data import nifti_dataset, h5py_dataset

# options
opt = TestOptions().parse()
os.makedirs(opt.results_dir, exist_ok=True)
preprocess = opt.preprocess
model   = deep_strain_model.DeepStrain(Adam, opt)

if 'segmentation' in opt.pipeline:
    
    opt.preprocess = opt.preprocess_carson + '_' + preprocess    
    dataset = nifti_dataset.NiftiDataset(opt)
    netS    = model.get_netS()
    for i, data in enumerate(dataset):

        filename = os.path.basename(dataset.filenames[i]).split('.')[0]

        x, nifti, nifti_resampled = data
        
        y = netS(x).numpy()
        y = dataset.transform.apply_inv(y)
        nifti_dataset.save_as_nifti(y, nifti, nifti_resampled,
                                    filename=os.path.join(opt.results_dir, filename+'_segmentation'))
        
        if len(nifti.shape)==4:
            # Save masks for strain analysis
            HF = h5py.File(os.path.join(opt.results_dir, filename+'_segmentation.h5'), 'w')
            for time_frame in range(y.shape[-2]):
                hf = HF.create_group('frame_%d' %(time_frame))
                hf.create_dataset('M', data=y[:,:,:,time_frame])
            HF.close()

    del netS

if 'motion' in opt.pipeline:
    opt.number_of_slices = 16 
    opt.preprocess = opt.preprocess_carmen + '_' + preprocess
    
    model   = deep_strain_model.DeepStrain(Adam, opt)
    dataset = nifti_dataset.NiftiDataset(opt)
    netME   = model.get_netME()
    
    
    for i, data in enumerate(dataset):
        filename = os.path.basename(dataset.filenames[i]).split('.')[0]

        x, nifti, nifti_resampled = data
        x_0, x_t = np.array_split(x,2,-1)
        y_t = netME([x_0, x_t]).numpy()
        y_t = dataset.transform.apply_inv(y_t)
        
        HF = h5py.File(os.path.join(opt.results_dir, filename+'_motion.h5'), 'w')
        for time_frame in range(y_t.shape[-2]):
            hf = HF.create_group('frame_%d' %(time_frame))
            hf.create_dataset('u', data=y_t[:,:,:,time_frame])
        HF.close()

    del netME   
    
# Development
##if 'strain' in opt.pipeline:   
##    opt.dataroot = opt.results_dir
##    dataset = h5py_dataset.H5PYDataset(opt)
##    
##    for i, x in enumerate(dataset):
##        
##        print(x.shape)