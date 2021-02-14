# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

import os
import h5py
import glob
import pydicom
import warnings
import numpy as np
import pandas as pd
import nibabel as nib

from data.base_dataset import BaseDataset, Transforms
from data.image_folder import make_dataset

class DICOMDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.filenames = sorted(make_dataset(opt.dataroot, opt.max_dataset_size, 'DICOM'))
        self.metadata, self.acquisitions  = self.read_metadata()
        
    def __len__(self):
        return len(self.acquisitions)
                
    def __getitem__(self, idx): 
        pid = '_'.join(self.acquisitions[idx].split('_')[:-1]) #PatientName
        uid = self.acquisitions[idx].split('_')[-1] #AcquisitionInstanceUID
        df  = self.metadata[(self.metadata.PatientName==pid)&(self.metadata.AcquisitionInstanceUID==uid)]
        return self.load_acquisition(df)
        
    def read_metadata(self):       
        
        metadata = {'FileName':[],
                    'PatientName':[], 
                    'SeriesInstanceUID':[], 
                    'StudyInstanceUID':[], 
                    'ProtocolName':[], 
                    'SeriesTime':[], 
                    'TriggerTime':[], 
                    'InstanceNumber':[], 
                    'ImageOrientationPatient':[],  
                    'ImagePositionPatient':[],
                    'SliceLocation':[], 
                    'PixelSpacing':[], 
                    'SliceThickness':[], 
                    'AcquisitionInstanceUID':[], 
                    'SliceInstanceUID':[]}
        
        for filename in self.filenames:
            dicom = pydicom.read_file(filename)

            metadata['FileName']                += [filename]
            metadata['PatientName']             += [str(dicom[0x0010, 0x0010].value)]
            metadata['SeriesInstanceUID']       += [dicom.SeriesInstanceUID]
            metadata['StudyInstanceUID']        += [dicom.StudyInstanceUID]
            metadata['ProtocolName']            += [dicom.ProtocolName]
            metadata['SeriesTime']              += [dicom.SeriesTime]
            metadata['TriggerTime']             += [dicom.TriggerTime]
            metadata['InstanceNumber']          += [dicom.InstanceNumber]
            metadata['ImageOrientationPatient'] += [dicom.ImageOrientationPatient]
            metadata['ImagePositionPatient']    += [dicom.ImagePositionPatient]
            metadata['SliceLocation']           += [dicom.SliceLocation]
            metadata['PixelSpacing']            += [dicom.PixelSpacing]
            metadata['SliceThickness']          += [dicom.SliceThickness]

            metadata['AcquisitionInstanceUID']  += [dicom.SeriesInstanceUID.split('.')[9]]
            metadata['SliceInstanceUID']        += [dicom.SeriesInstanceUID.split('.')[10]]

        metadata = pd.DataFrame(metadata)   
        
        acquisitions = []
        print('Found %d patient(s):'%(len(metadata.PatientName.unique())))
        for patient in metadata.PatientName.unique():
            acqs = metadata[metadata.PatientName==patient].AcquisitionInstanceUID.unique().tolist()
            print(patient, ': with', len(acqs), 'acquisitions:')
            for acq in sorted(acqs):
                acquisitions += [patient+'_'+acq]
                print('  ', acquisitions[-1])

        return metadata, acquisitions


    def load_acquisition(self, df):

        slices = df.SeriesInstanceUID.unique().tolist()
        phases = df.SeriesInstanceUID.value_counts().unique()
        assert len(phases)==1, 'Number of phases does not match!'
        number_of_slices = len(slices) 
        number_of_phases = int(phases)
        pixel_array = pydicom.read_file(df.iloc[0].FileName).pixel_array

        sax_4D = np.zeros((pixel_array.shape +(number_of_slices, number_of_phases)), dtype=pixel_array.dtype)

        for z_slice, series in enumerate(slices):
            for phase in range(number_of_phases):
                filename = df[df.SeriesInstanceUID==series].sort_values('InstanceNumber').iloc[phase].FileName
                dicom = pydicom.read_file(filename)

                sax_4D[:,:,z_slice,phase] += dicom.pixel_array

        affine = read_affine(df.iloc[df.SliceLocation.argmin()])

        return nib.Nifti1Image(sax_4D, affine)
    
def extract_cosines(ImageOrientationPatient):
    row_cosine    = np.array(ImageOrientationPatient[:3])
    column_cosine = np.array(ImageOrientationPatient[3:])
    slice_cosine  = np.cross(row_cosine, column_cosine)
    return np.stack((row_cosine, column_cosine, slice_cosine))

def read_affine(df, viewer='slicer'):
    Zooms = np.array(list(df.PixelSpacing)+[df.SliceThickness], dtype=float)
    ImageOrientationPatient = np.array(df.ImageOrientationPatient, dtype=float)
    ImagePositionPatient    = np.array(df.ImagePositionPatient, dtype=float)
    
    ijk2ras = extract_cosines(ImageOrientationPatient)
    if viewer == "slicer":
        ijk2ras = (ijk2ras*np.array([-1,-1,1])).T
        ImagePositionPatient = ImagePositionPatient*np.array([-1,-1,1])

    affine  = np.stack((ijk2ras[:,0]*Zooms[0],
                        ijk2ras[:,1]*Zooms[1],
                        ijk2ras[:,2]*Zooms[2],
                        ImagePositionPatient), axis=1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return np.vstack((affine,[[0,0,0,1]]))        