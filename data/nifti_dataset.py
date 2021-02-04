import os
import glob
import warnings
import numpy as np
import nibabel as nib

from dipy.align.reslice import reslice
from data.base_dataset import BaseDataset, Transforms
from data.image_folder import make_dataset


class NiftiDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.filenames = sorted(make_dataset(opt.dataroot, opt.max_dataset_size, opt.dataformat))
        self.transform = Transforms(opt)
    
    def __len__(self):
        return len(self.filenames)
                
    def __getitem__(self, idx):      
        nifti           = nib.load(self.filenames[idx])
        nifti_resampled = resample_nifti(nifti, 
                                         order=self.opt.order,
                                         mode=self.opt.mode,
                                         in_plane_resolution_mm=self.opt.in_plane_resolution_mm,
                                         slice_thickness_mm=self.opt.slice_thickness_mm,
                                         number_of_slices=self.opt.number_of_slices)
        
        x = self.transform.apply(nifti_resampled.get_fdata())
        return x, nifti, nifti_resampled
    

def save_as_nifti(y, nifti, nifti_resampled, filename):
    
    if len(y.shape) == 4:
        y_nifti_resampled = nib.Nifti1Image(y, nifti_resampled.affine)
        y_nifti = resample_nifti(y_nifti_resampled, 
                                 in_plane_resolution_mm=nifti.header.get_zooms()[0],
                                 slice_thickness_mm=nifti.header.get_zooms()[2])
        
        y_nifti = nib.Nifti1Image(np.array(np.argmax(y_nifti.get_fdata(),-1)), 
                                  affine=nifti.affine,
                                  header=nifti.header)
        
    elif len(y.shape) == 5:
        
        Y = []
        for label in range(y.shape[-1]):
            y_nifti_resampled = nib.Nifti1Image(y[:,:,:,:,label], nifti_resampled.affine)
            y_nifti = resample_nifti(y_nifti_resampled, 
                                    in_plane_resolution_mm=nifti.header.get_zooms()[0],
                                    slice_thickness_mm=nifti.header.get_zooms()[2])
            Y += [y_nifti.get_fdata()]
            
        y_nifti = nib.Nifti1Image(np.argmax(np.stack(Y,-1),-1).astype(int), nifti.affine)

    nib.Nifti1Image(np.array(y_nifti.get_fdata()), affine=y_nifti.affine).to_filename(filename+'.nii') 
 

def resample_nifti(nifti, 
                   order=1,
                   mode='nearest',
                   in_plane_resolution_mm=1.25,
                   slice_thickness_mm=None,
                   number_of_slices=None):
    
    # sometimes dicom to nifti programs don't define affine correctly.
    resolution = np.array(nifti.header.get_zooms()[:3] + (1,))
    if (np.abs(nifti.affine)==np.identity(4)).all():
        nifti.set_sform(nifti.affine*resolution)
        warnings.warn("Affine in nifti might be set incorrectly. Setting to affine=affine*zooms")

    data   = nifti.get_fdata().copy()
    shape  = nifti.shape[:3]
    affine = nifti.affine.copy()
    zooms  = nifti.header.get_zooms()[:3]

    if number_of_slices is not None:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     (zooms[2] * shape[2]) / number_of_slices)
    elif slice_thickness_mm is not None:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     slice_thickness_mm)            
    else:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     zooms[2])

    new_zooms = np.array(new_zooms)
    for i, (n_i, res_i, res_new_i) in enumerate(zip(shape, zooms, new_zooms)):
        n_new_i = (n_i * res_i) / res_new_i
        # to avoid rounding ambiguities
        if (n_new_i  % 1) == 0.5: 
            new_zooms[i] -= 0.001

    data_resampled, affine_resampled = reslice(data, affine, zooms, new_zooms, order=order, mode=mode)
    nifti_resampled = nib.Nifti1Image(data_resampled, affine_resampled)

    x=nifti_resampled.header.get_zooms()[:3]
    y=new_zooms
    if not np.allclose(x,y, rtol=1e-02):
        print(x,y)
        warnings.warn('Output resolutions are different than expected!')

    return nifti_resampled       
    
    
    
    