import os
import h5py
import timeit
import numpy as np
import nibabel as nib
from tensorflow.keras.optimizers import Adam
from options.test_options import TestOptions
from models import deep_strain_model
from data import nifti_dataset, h5py_dataset
from utils import myocardial_strain

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
    
if 'strain' in opt.pipeline:   
    dataset = h5py_dataset.H5PYDataset(opt)
  
    
    for idx, u in enumerate(dataset): 
        
        filename  = dataset.filenames[idx].split('_motion.h5')[0]
        mask_path = filename+'_segmentation.nii'
        try:
            mask_nifti = nib.load(mask_path)

        except:
            print('Missing segmentation')
            continue

        mask_zooms = mask_nifti.header.get_zooms()
        mask_nifti = nifti_dataset.resample_nifti(mask_nifti, in_plane_resolution_mm=1.25, number_of_slices=16)
        mask = mask_nifti.get_fdata()

        Radial = np.zeros(mask.shape)
        Circumferential = np.zeros(mask.shape)
        for time_frame in range(u.shape[-1]):
            strain = myocardial_strain.MyocardialStrain(mask=mask[:,:,:,0], flow=u[:,:,:,:,time_frame])
            strain.calculate_strain(lv_label=3)

            strain.Err[strain.mask_rot!=2] = 0.0
            strain.Ecc[strain.mask_rot!=2] = 0.0

            Radial[:,:,:,time_frame]          += strain.Err
            Circumferential[:,:,:,time_frame] += strain.Ecc

            GRS = strain.Err[strain.mask_rot==2].mean()
            GCS = strain.Ecc[strain.mask_rot==2].mean()
            print(GRS, GCS)


        Radial = nib.Nifti1Image(Radial, mask_nifti.affine)
        Circumferential = nib.Nifti1Image(Circumferential, mask_nifti.affine)

        Radial.to_filename(filename+'_radial_strain.nii')
        Circumferential.to_filename(filename+'_circumferential_strain.nii')






