# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

import numpy as np
from abc import ABC, abstractmethod
from tensorflow.keras.utils import Sequence
from scipy.ndimage.measurements import center_of_mass

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
    
    