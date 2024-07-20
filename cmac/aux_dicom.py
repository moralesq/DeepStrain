import os
import glob
import pydicom
import numpy as np
import nibabel as nib
from dipy.align.reslice import reslice
from dipy.align.imaffine import AffineMap

def extract_cosines(ImageOrientationPatient):
    row_cosine    = np.array(ImageOrientationPatient[:3])
    column_cosine = np.array(ImageOrientationPatient[3:])
    slice_cosine  = np.cross(row_cosine, column_cosine)
    return row_cosine, column_cosine, slice_cosine

def read_RefDs(dicom_folder_path):
    cine_dicom_files = glob.glob(dicom_folder_path)
    SliceLocations   = [pydicom.read_file(cine_dicom_file).SliceLocation 
                        for cine_dicom_file in cine_dicom_files]
    SliceOriginID    = np.array(SliceLocations, dtype=float).argmin()
    
    RefDs = pydicom.read_file(cine_dicom_files[SliceOriginID])
    ImageOrientationPatient = RefDs.ImageOrientationPatient
    ImagePositionPatient    = np.array(list(RefDs.ImagePositionPatient), dtype=float)
    Zooms                   = np.array(list(RefDs.PixelSpacing) + [RefDs.SliceThickness], dtype=float)
    
    return ImageOrientationPatient, ImagePositionPatient, Zooms

def read_affine_info(dicom_folder_path, viewer='slicer'):
    
    ImageOrientationPatient, ImagePositionPatient, Zooms = read_RefDs(dicom_folder_path)
    
    affine_axial        = np.diag(list(Zooms)+[1])
    affine_axial[:3,3] += ImagePositionPatient
    
    row_cos, column_cos, slice_cos = extract_cosines(ImageOrientationPatient)
    
    ijk2ras = np.stack((row_cos, column_cos, slice_cos))
    if viewer == "slicer":
        ijk2ras = (ijk2ras*np.array([-1,-1,1])).T
        ImagePositionPatient = ImagePositionPatient*np.array([-1,-1,1])

    affine  = np.stack((ijk2ras[:,0]*Zooms[0],
                        ijk2ras[:,1]*Zooms[1],
                        ijk2ras[:,2]*Zooms[2],
                        ImagePositionPatient), axis=1)
    
    return ijk2ras, np.vstack((affine,[[0,0,0,1]])), affine_axial   

def reslice_3d(domain_nifti, codomain_nifti, mode='linear'):
    
    affine_map = AffineMap(affine=np.eye(4),
                  domain_grid_shape=domain_nifti.shape, domain_grid2world=domain_nifti.affine, 
                  codomain_grid_shape=codomain_nifti.shape, codomain_grid2world=codomain_nifti.affine)

    return nib.Nifti1Image(affine_map.transform(codomain_nifti.get_fdata(),mode),domain_nifti.affine)

def reslice_4d(domain_nifti, codomain_nifti, mode='linear'):
    
    reslice_arr = np.zeros(domain_nifti.shape[:3]+(codomain_nifti.shape[-1],))
    affine_map  = AffineMap(affine=np.eye(4),
                  domain_grid_shape=domain_nifti.shape[:3],     domain_grid2world=domain_nifti.affine, 
                  codomain_grid_shape=codomain_nifti.shape[:3], codomain_grid2world=codomain_nifti.affine)
    
    for frame_id in range(reslice_arr.shape[-1]):
        reslice_arr[:,:,:,frame_id] += affine_map.transform(codomain_nifti.get_fdata()[:,:,:,frame_id],mode)
    
    return nib.Nifti1Image(reslice_arr, domain_nifti.affine)


def resample_nifti(subject_nifti, resolution=(1.5, 1.5), Nz=16, order=1, mode='nearest'):
    """Resample a 3D or 4D (3D+time) cine-MRI nifti to a new in-plane `resolution` with `Nz` slices."""
 
    data   = subject_nifti.get_fdata()
    affine = subject_nifti.affine
    zooms  = subject_nifti.header.get_zooms()[:3]
    
    new_zooms = (resolution[0], resolution[1], (zooms[2] * data.shape[2]) / Nz)
     
    data_resampled, affine_resampled = reslice(data, affine, zooms, new_zooms, order=order, mode=mode)
    subject_nifti_resampled = nib.Nifti1Image(data_resampled, affine_resampled)
     
    x=subject_nifti_resampled.header.get_zooms()[:3]
    y=new_zooms
    if not np.allclose(x,y, rtol=1e-02):
        print(subject_nifti.affine,affine)
        print(zooms)
        print(x,y)
        warnings.warn('Output resolutions are different than expected!')
         
    return subject_nifti_resampled 