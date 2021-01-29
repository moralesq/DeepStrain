import os
import warnings
import numpy as np
import nibabel as nib
from dipy.align.reslice import reslice
from dipy.align.imaffine import AffineMap

def resample_nifti(nifti, order=1, mode='nearest',
                   in_plane_resolution_mm=1.25, slice_thickness_mm=None, number_of_slices=None):
    
    resolution = np.array(nifti.header.get_zooms()[:3] + (1,))
    if not (np.abs(np.diag(nifti.affine))== resolution).all():
        # Affine is not properly set, fix. 
        nifti.set_sform(nifti.affine*resolution)
        
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


def normalize_zscore(x, axis=None):
    return (x - x.mean(axis=axis, keepdims=True))/(x.std(axis=axis, keepdims=True) + 1e-8)
    
    
def load_from_file(filename):
    basename = os.path.basename(filename)
    
    if basename.split('.')[1] == 'nii':
        # NIFTI FILE
        nifti = nib.load(filename)
        nifti_resampled = resample_nifti(nifti)
        
        if len(nifti_resampled.shape) == 3:
            # SINGLE FRAME
            nx, ny, nz = nifti_resampled.shape
            x = nifti_resampled.get_fdata()[nx//2-64:nx//2+64, ny//2-64:ny//2+64].transpose(2,0,1)
            x = normalize_zscore(x, axis=(1,2))
            return x, nifti, nifti_resampled

        elif len(nifti_resampled.shape) == 4:
            # MULTI FRAME
            nx, ny, nz, nf = nifti_resampled.shape
            x = nifti_resampled.get_fdata()[nx//2-64:nx//2+64, ny//2-64:ny//2+64].transpose(3,2,0,1)
            x = x.reshape((nf*nz, nx, ny))
            x = normalize_zscore(x, axis=(1,2))
            return x, nifti, nifti_resampled
        
def save_to_file_nifti(filename, y, x_nifti, x_nifti_resampled):
    basename     = os.path.basename(filename)
    basename_seg = basename.split('.')[0] + '_segmentation.nii.gz'
    filename_seg = os.path.join(os.path.dirname(filename), basename_seg)
    
    if len(x_nifti_resampled.shape) == 3:
        # SINGLE FRAME
        nx, ny, nz = x_nifti_resampled.shape
        y = y.transpose((1,2,0,3))
    elif len(x_nifti_resampled.shape) == 4:
        # MULTI FRAME
        nx, ny, nz, nf = x_nifti_resampled.shape
        y = y.reshape((nf, nz, nx, ny, 4))
        y = y.transpose((2,3,1,0,4))
        
    y_nifti_resampled = np.zeros(x_nifti_resampled.shape+(4,))
    y_nifti_resampled[nx//2-64:nx//2+64, ny//2-64:ny//2+64] += y
    
    y_nifti_resampled = nib.Nifti1Image(y_nifti_resampled, x_nifti_resampled.affine)
    y_nifti = resample_nifti(y_nifti_resampled, in_plane_resolution_mm=x_nifti.header.get_zooms()[0])
    y_nifti.to_filename(filename_seg)
    
    return y_nifti, y_nifti_resampled    