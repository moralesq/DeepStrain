import warnings
import numpy as np
import nibabel as nib
from dipy.align.reslice import reslice


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