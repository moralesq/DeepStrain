# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

import numpy as np
from abc import ABC, abstractmethod
from tensorflow.keras.utils import Sequence
from scipy.ndimage.measurements import center_of_mass

import nibabel as nib
from dipy.align.reslice import reslice

class BaseDataset(Sequence, ABC):
    """This class is an abstract base class (ABC) for datasets."""

    def __init__(self, opt):
        self.opt  = opt
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        """Return the size of the dataset."""
        return 
    
    @abstractmethod
    def __getitem__(self, idx):
        """Return a data point and its metadata information."""
        pass
                       
class Transforms():
    
    def __init__(self, opt):
        self.opt = opt 
        self.transform, self.transform_inv = self.get_transforms(opt)
       
    def __crop__(self, x, inv=False):
        
        if inv:
            nx, ny = self.original_shape[:2]
            xinv = np.zeros(self.original_shape[:2] + x.shape[2:])
            xinv[nx//2-64:nx//2+64, ny//2-64:ny//2+64] += x
            return xinv
        else:
            nx, ny = x.shape[:2]
            return x[nx//2-64:nx//2+64, ny//2-64:ny//2+64]
    
    def __reshape_to_carson__(self, x, inv=False):
        
        if inv:
            if len(self.original_shape)==3:
                x = x.transpose(1,2,0,3)
            elif len(self.original_shape)==4:
                nx,ny,nz,nt=self.original_shape
                Nx, Ny = x.shape[1:3]
                x = x.reshape((nt, nz, Nx, Ny, self.opt.nlabels))
                x = x.transpose(2,3,1,0,4)                
        else:
            if len(x.shape) == 3:
                nx,ny,nz=x.shape
                x=x.transpose(2,0,1)
            elif len(x.shape) == 4:
                nx,ny,nz,nt=x.shape
                x=x.transpose(3,2,0,1)
                x=x.reshape((nt*nz,nx,ny))            
        return x

    def __reshape_to_carmen__(self, x, inv=False):
        if inv:
            x = np.concatenate((np.zeros(x[:1].shape), x))
            x = x.transpose((1,2,3,0,4)) 
        else:
            assert len(x.shape) == 4
            nx,ny,nz,nt=x.shape
            x=x.transpose(3,0,1,2)
            x=np.stack((np.repeat(x[:1],nt-1,axis=0), x[1:nt]), -1)
        return x  
    
    def __zscore__(self, x):

        if len(x.shape) == 3:
            axis=(1,2) # normalize in-plane images independently
        elif len(x.shape) == 5:
            axis=(1,2,3) # normalize volumes independently

        self.mu = x.mean(axis=axis, keepdims=True)
        self.sd = x.std(axis=axis, keepdims=True)
        return (x - self.mu)/(self.sd + 1e-8)

    def get_transforms(self, opt):

        transform_list     = []
        transform_inv_list = []
        if 'crop' in opt.preprocess:
            transform_list.append(self.__crop__)
            transform_inv_list.append(lambda x:self.__crop__(x,inv=True))
        if 'reshape_to_carson' in opt.preprocess:
            transform_list.append(self.__reshape_to_carson__)
            transform_inv_list.append(lambda x:self.__reshape_to_carson__(x,inv=True))
        elif 'reshape_to_carmen' in opt.preprocess:
            transform_list.append(self.__reshape_to_carmen__)
            transform_inv_list.append(lambda x:self.__reshape_to_carmen__(x,inv=True))
        if 'zscore' in opt.preprocess:
            transform_list.append(self.__zscore__)                
        
        return transform_list, transform_inv_list
          
    def apply(self, x):
        
        self.original_shape = x.shape
        for transform in self.transform:
            x = transform(x)
        return x
    
    def apply_inv(self, x):
        
        for transform in self.transform_inv[::-1]:
            x = transform(x)
        return x    
    

def _centercrop(x):
    nx, ny = x.shape[:2]
    return x[nx//2-64:nx//2+64,ny//2-64:ny//2+64]

def _roll(x,rx,ry):
    x = np.roll(x,rx,axis=0)
    x = np.roll(x,ry,axis=1)
    return x

def _roll2center(x, center):
    return _roll(x, int(x.shape[0]//2-center[0]), int(x.shape[1]//2-center[1]))
    
def _roll2center_crop(x, center):
    x = _roll2center(x, center)
    return _centercrop(x)
    
    
#####################################################
## FUNCTIONS TO ADD MORE FLEXIBILITY IN SEGMENTATION
#####################################################

def resample_nifti_inv(nifti_resampled, zooms, order=1, mode='nearest'):
    """ Resample `nifti_resampled` to `zooms` resolution.
    """
    data_resampled   = nifti_resampled.get_fdata()
    zooms_resampled  = nifti_resampled.header.get_zooms()[:3]
    affine_resampled = nifti_resampled.affine 
        
    data_resampled, affine_resampled = reslice(data_resampled, 
                                               affine_resampled, zooms_resampled, zooms, order=order, mode=mode)

    nifti = nib.Nifti1Image(data_resampled, affine_resampled)
    
    return nifti
    
def convert_back_to_nifti(data_resampled, nifti_info_subject, inv_256x256=False, order=1, mode='nearest'):

    if inv_256x256:
        data_resampled_mod_corr = roll_and_pad_256x256_to_center_inv(data_resampled, nifti_info=nifti_info_subject)
    else:
        data_resampled_mod_corr = data_resampled
        
    affine           = nifti_info_subject['affine']
    affine_resampled = nifti_info_subject['affine_resampled']
    zooms            = nifti_info_subject['zooms'][:3]
    zooms_resampled  = nifti_info_subject['zooms_resampled'][:3]
    
    data_resampled, affine_resampled = reslice(data_resampled_mod_corr, 
                                               affine_resampled, zooms_resampled, zooms, order=order, mode=mode)
    nifti = nib.Nifti1Image(data_resampled, affine_resampled)
    
    return nifti

def roll(x,rx,ry):
        x = np.roll(x,rx,axis=0)
        x = np.roll(x,ry,axis=1)
        return x
    
def roll2center(x, center):
    return roll(x, int(x.shape[0]//2-center[0]), int(x.shape[1]//2-center[1]))
    
def pad_256x256(x):
        xpad = (512-x.shape[0])//2, (512-x.shape[0])-(512-x.shape[0])//2
        ypad = (512-x.shape[1])//2, (512-x.shape[1])-(512-x.shape[1])//2
        pads = (xpad,ypad)+((0,0),)*(len(x.shape)-2)
        vals = ((0,0),)*len(x.shape)
        x = np.pad(x, pads, 'constant', constant_values=vals)
        x = x[512//2-256//2:512//2+256//2,512//2-256//2:512//2+256//2]
        return x
    
def roll_and_pad_256x256_to_center(x, center):
    x = roll2center(x, center)
    x = pad_256x256(x)
    return x

def roll_and_pad_256x256_to_center_inv(x, nifti_info):

    # Recover 256x256 array that was center-cropped to 128x128!
    x_256_256 = np.zeros((256,256)+x.shape[2:])
    x_256_256[128-64:128+64,128-64:128+64] += x
    
    # Coordinates to put the image in its original location.
    cx, cy         = nifti_info['center_resampled'][:2]
    cx_mod, cy_mod = nifti_info['center_resampled_256x256'][:2]
    
    x_inv = np.zeros(nifti_info['shape_resampled'][:3]+x.shape[3:])

    dx = min(int(cx),64)
    dy = min(int(cy),64)
    if (dx!=64)|(dy!=64):
        print('WARNING:FOV < 128x128!')

    x_inv[int(cx-dx):int(cx+dx),int(cy-dy):int(cy+dy)] += x_256_256[int(cx_mod-dx):int(cx_mod+dx),
                                                                    int(cy_mod-dy):int(cy_mod+dy)]
    return x_inv
