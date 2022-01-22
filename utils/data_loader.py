import os
import h5py
import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pylab as plt
from dipy.align.reslice import reslice
from scipy.ndimage.measurements import center_of_mass
from scipy.special import softmax
from tensorflow.keras.utils import to_categorical
from skimage.exposure import rescale_intensity

class ACDC():
    
    def __init__(self, acdc_path='DeepStrain/data/ACDC/ACDC',seed=None):
        """Python wrapper for the Automated Cardiac Diagnosis Challenge (ACDC) Dataset."""
        
        self.seed      = seed
        self.acdc_path = acdc_path
        self.dtrn_path = os.path.join(self.acdc_path,'training','patient%.3d')
        self.dtst_path = os.path.join(self.acdc_path,'testing','patient%.3d')
        
        self.dtrn_params = self.read_info(os.path.join(self.dtrn_path,'info.cfg'),subject_ids=range(1,101))
        self.dtst_params = self.read_info(os.path.join(self.dtst_path,'Info.cfg'),subject_ids=range(101,151))
        self.list_SIDs   = np.random.RandomState(seed=seed).permutation(100)+1
        
        # THIS IS NEEDED TO SPLIT VOLUMES BY SLICE 
        # 1-100 sids at diastole whole volume, 101-200 sids at systole whole volume
        # 1001-1100 sids at diastole slice z=1, 1101-1200 sids at systole slice z=1
        # 2001-2100 sids at diastole slice z=2, 2101-2200 sids at systole slice z=2 (and so on until z=16)
        self.SID_network_keys_carson = {}
        for sid in self.list_SIDs:
            self.SID_network_keys_carson[sid]     = sid, 'diastole', None
            self.SID_network_keys_carson[sid+100] = sid, 'systole', None
            for z in range(1,17):
                self.SID_network_keys_carson[sid+1000*z]     = sid, 'diastole', z
                self.SID_network_keys_carson[sid+100+1000*z] = sid, 'systole', z
                
    def read_info(self, path, subject_ids):
        """Read subject information (ED,ES,Height,NbFrame,Weight)."""

        subject_parameters = []
        for subject_id in subject_ids:
            d = pd.read_fwf(path %(subject_id), header=None, names=['params'])
            d = d.params.str.split(':', expand=True)
            d.set_index(0, inplace=True)
            d.columns = ['subject %d' %(subject_id)]
            subject_parameters.append(d.T)
            
        return pd.concat(subject_parameters)       
 
    def _load_subject_4d(self,sid,resample=True,nifti_only=False):
        """Load subject cine-MRI sequence."""
        path  = os.path.join(self.dtrn_path, 'patient%.3d_4d.nii.gz')
        nifti = nib.load(path%(sid,sid))
        if resample:
            nifti = resample_nifti(nifti)
        if nifti_only:
            return nifti
        return nifti.get_data()

    def _load_subject_phase(self,sid,phase=0,gt='',resample=True,order=1,nifti_only=False):
        """Load subject cine-MRI volume at diastole (phase=0) or systole (phase=1). Use gt='_gt' for labels."""
        path  = os.path.join(self.dtrn_path,'patient%.3d_frame%.2d'+'%s.nii.gz'%(gt))
        frame = int(self.dtrn_params.iloc[sid-1,phase])
        nifti = nib.load(path%(sid,sid,frame))
        if resample:
            if gt == '_gt':
                order=0
            # set scale_affine=True since these niftis were saved without the proper affine
            nifti = resample_nifti(nifti,order=order)
        if nifti_only:
            return nifti            
        return nifti.get_data()
    
    def convert_to_h5(self):
        
        os.makedirs(self.acdc_path+'_DS', exist_ok=True)
        HF = h5py.File(os.path.join(self.acdc_path+'_DS','acdc.h5'), 'w')
        for sid in range(1,101):
            print('converting subject %d ...'%(sid))
            hf = HF.create_group('subject_%d' %(sid))
            _4d          = pad_to_256x256(self._load_subject_4d(sid))
            _diastole    = pad_to_256x256(self._load_subject_phase(sid,0))
            _diastole_gt = pad_to_256x256(self._load_subject_phase(sid,0,gt='_gt'))
            _systole     = pad_to_256x256(self._load_subject_phase(sid,1))
            _systole_gt  = pad_to_256x256(self._load_subject_phase(sid,1,gt='_gt'))
            
            hf.create_dataset('4d',         data=_4d)
            hf.create_dataset('diastole',   data=_diastole)
            hf.create_dataset('diastole_gt',data=_diastole_gt)
            hf.create_dataset('systole',    data=_systole)
            hf.create_dataset('systole_gt', data=_systole_gt)
            
            for k in range(16):
                hf.create_dataset('diastole_%d' %(k),   data=_diastole[:,:,k])
                hf.create_dataset('diastole_gt_%d' %(k),data=_diastole_gt[:,:,k])
                hf.create_dataset('systole_%d' %(k),    data=_systole[:,:,k])
                hf.create_dataset('systole_gt_%d' %(k), data=_systole_gt[:,:,k])
            
        HF.close()    
    
    def load_subject(self,subject_id,dataset='4d',z=''):
        """Basic function to load a subject in .h5 format.
            
            Params:
            ------
            subject_id : Subject to load (1-100)
            dataset    : String of data load: 4d,diastole,diastole_gt,systole,systole_gt   
        """
        HF   = h5py.File(os.path.join(self.acdc_path+'_DS','acdc.h5'), 'r')
        data = np.array(HF['subject_%d/%s%s' %(subject_id,dataset,z)]); HF.close()
        return data
    
    def load_subject_gt_set(self,subject_id):
        V = np.zeros((2,256,256,16),dtype='float32')
        M = np.zeros((2,256,256,16),dtype='float32')
        V[0,:,:,:] += self.load_subject(subject_id,dataset='diastole')
        V[1,:,:,:] += self.load_subject(subject_id,dataset='systole')
        M[0,:,:,:] += self.load_subject(subject_id,dataset='diastole_gt')
        M[1,:,:,:] += self.load_subject(subject_id,dataset='systole_gt')
        return V, M
    
    def load_subject_vcn(self, SID, roll2center=False, joint_augmentation=False, crop=False, **kwargs):

        sid, phase, z = self.SID_network_keys_carson[SID]
        M = self.load_subject(subject_id=sid, dataset=phase+'_gt')
        V = self.load_subject(subject_id=sid, dataset=phase)
        
        #M = self.load_subject(subject_id=SID, dataset='diastole_gt')
        #V = self.load_subject(subject_id=SID, dataset='diastole')
        G = gauss3d(M)
        
        if roll2center:
            V, G = _roll2center(V, G, M)
        if joint_augmentation:
            V, G = _joint_augmentation_subject(V, G, **kwargs)
        if crop:
            V, G = _crop(V, G)
        
        V = _normalize(V)

        return V[:,:,:,None].astype('float32'), G[:,:,:,None].astype('float32')
    
    def load_subject_carson(self, SID, 
                            roll2center=False, 
                            joint_augmentation=False, 
                            single_augmentation=False,
                            slice_misaligned_augmentation=True,
                            crop=False, label=None, **kwargs):
        
        sid, phase, z = self.SID_network_keys_carson[SID]
        if z is not None: # load slice only
            slice_misaligned_augmentation = False # not needed for single slice
            M = self.load_subject(subject_id=sid, dataset=phase+'_gt', z='_%d' %(z-1))
            V = self.load_subject(subject_id=sid, dataset=phase, z='_%d' %(z-1))
            for label in [1,2,3]: # quick fix to prevent `to_categorical` errors
                if M[M==label].sum() == 0:
                    M[label,label] = label # shouldn't be a problem for learning
        else:
            M = self.load_subject(subject_id=sid, dataset=phase+'_gt')
            V = self.load_subject(subject_id=sid, dataset=phase)
            
        if roll2center:
            V, M = _roll2center(V, M, M)
        if slice_misaligned_augmentation:
            for k in range(16):
                if np.random.rand()<0.10:
                    r  = int(10*np.random.rand())
                    axis = np.random.randint(2)
                    V[:,:,k] = np.roll(V[:,:,k], r, axis=axis)
                    M[:,:,k] = np.roll(M[:,:,k], r, axis=axis)
        if single_augmentation:
            V = _single_augmentation_subject(V, **kwargs)
        if joint_augmentation:
            V, M = _joint_augmentation_subject(V, M, **kwargs)
        if crop:
            V, M = _crop(V, M)
        
        V = _normalize(V)
        V = np.expand_dims(V, -1).astype('float32')
        M = to_categorical(M)
        return V, M

    def load_subject_carmen(self, SID, 
                            roll2center=False, 
                            single_augmentation=False,
                            joint_augmentation=False, 
                            crop=False, **kwargs):    

        V, M = self.load_subject_gt_set(subject_id=SID)
        
        if roll2center:
            V[0], V[1] = _roll2center(V[0], V[1], M[0])
            M[0], M[1] = _roll2center(M[0], M[1], M[0]) 
        # treat 16x2 slices as channels of 2D images (256x256x16), augmentation on both channels 
        V, M = np.concatenate((V[0],V[1]),axis=-1), np.concatenate((M[0],M[1]),axis=-1)  
        if single_augmentation:
            V = _single_augmentation_subject(V, **kwargs)
        if joint_augmentation:
            V, M = _joint_augmentation_subject(V, M, **kwargs)
        V, M = np.stack(np.array_split(V,2,-1),axis=0), np.stack(np.array_split(M,2,-1),axis=0)
        if crop:
            V, M = _crop(V, M)
        
        V[0] = _normalize(V[0])
        V[1] = _normalize(V[1])
        diastole = np.stack((V[0],M[0]),-1)
        systole  = np.stack((V[1],M[1]),-1)
        
        return diastole, systole

class CMAC():
    
    def __init__(self, cmac_path='/tf/DeepStrain/data/CMAC'):
        """Python wrapper for the Cardiac Motion Analysis Challenge (CMAC) Dataset."""
        self.cmac_path   = cmac_path
        self.dtst_path   = os.path.join(self.cmac_path,'subject%.3d')
        self.dtst_params = self.read_info(os.path.join(self.dtst_path,'info.cfg'),subject_ids=range(1,17))        
                
    def read_info(self, path, subject_ids):
        """Read subject information (ED,ES,Height,NbFrame,Weight)."""

        subject_parameters = []
        for subject_id in subject_ids:
            d = pd.read_fwf(path %(subject_id), header=None, names=['params'])            
            d = d.params.str.split(':', expand=True)
            d.set_index(0, inplace=True)
            d.columns = ['subject %d' %(subject_id)]
            subject_parameters.append(d.T)

        return pd.concat(subject_parameters)  
     
    def _load_subject_4d(self,sid,resample=True,nifti_only=False):
        """Load subject cine-MRI sequence."""
        path  = os.path.join(self.dtst_path, 'subject%.3d_4d.nii.gz')
        nifti = nib.load(path%(sid,sid))
        if resample:
            nifti = resample_nifti(nifti,CMAC=True)
        if nifti_only:
            return nifti
        else:
            return pad_to_256x256(nifti.get_data())[:,:,::-1]
    
    def _load_subject_phase(self,sid,phase=0,gt='',order=0,nifti_only=False):
        """Load subject cine-MRI volume at diastole (phase=0) or systole (phase=1). Use gt='_gt' for labels."""
        path  = os.path.join(self.dtst_path,'subject%.3d_frame%.2d_resampled'+'%s.nii.gz'%(gt))

        frame = int(self.dtst_params.iloc[sid-1,phase])
        nifti = nib.load(path%(sid,sid,frame))
        if nifti_only:
            return nifti
        else:
            return pad_to_256x256(nifti.get_data())[:,:,::-1]       
        
#########################################################################
############################## MARTINOS #################################
#########################################################################
class MARTINOS():
    
    def __init__(self, martinos_path='data/MARTINOS'):
        
        self.martinos_path = martinos_path
        self.patients      = [file for root, dirs, files in os.walk(self.martinos_path, topdown=False) 
                                   for file in files if file.endswith('.nii')]
        self.remove_bads()
        
    def remove_bads(self):
        
        patients = []
        for patient in self.patients:
            subject_nifti = nib.load(os.path.join(self.martinos_path, patient))
            shape = subject_nifti.shape

            if len(shape) < 4:
                print('skipping file with dim < 4')
                continue
            if shape[2] < 6:
                print('Warning:%s with shape %d,%d,%d,%d,has less than 6 slices! skipping file' %((patient,) + shape))
                continue
                
            patients += [patient]
        self.patients = patients
        
    def convert_to_h5(self):
        
        patients = []
        os.makedirs(self.martinos_path+'_DS', exist_ok=True)
        HF = h5py.File(os.path.join(self.martinos_path+'_DS','martinos.h5'), 'w')
        for patient in self.patients:
            subject_nifti = nib.load(os.path.join(self.martinos_path,patient))
            shape = subject_nifti.shape
            
            if len(shape) < 4:
                print('skipping file with dim < 4')
                continue
            if shape[2] < 6:
                print('Warning:%s with shape %d,%d,%d,%d,has less than 6 slices! skipping file' %((patient,) + shape))
                continue

            subject_nifti_resampled = resample_nifti(subject_nifti, inv=True)

            hf = HF.create_group('%s' %(patient))
            
            _4d            = pad_to_256x256(subject_nifti_resampled.get_data())
            resolution     = subject_nifti.header.get_zooms()
            resolution_new = subject_nifti_resampled.header.get_zooms()
            
            hf.create_dataset('4d',             data=_4d)
            hf.create_dataset('resolution',     data=resolution)
            hf.create_dataset('resolution_new', data=resolution_new)
            
            patients += [patient]
        
        self.patients = patients
        HF.close()    
        
    def categorize_patients(self):
        
        patients_categorized = {'normal':{}, 'dm':{}}
        for patient in self.patients:
            patient = patient.split('.nii')[0]
            if patient.startswith('MSTAT_DM'):
                pid = patient.split('MSTAT_DM_VOL')[1].split('_')[0]
                if pid not in patients_categorized['dm']:
                    patients_categorized['dm'][pid] = {}
                patients_categorized['dm'][pid][patient] = {}
            elif patient.startswith('MSTAT_VOL'):    
                pid = patient.split('MSTAT_VOL')[1].split('_')[0]
                if pid not in patients_categorized['normal']:
                    patients_categorized['normal'][pid] = {}
                patients_categorized['normal'][pid][patient] = {}   
                
        self.patients_categorized = patients_categorized
        
    def load_subject(self,patient,dataset='4d'):
        """Basic function to load a subject in .h5 format.
        """
        HF   = h5py.File(os.path.join(self.martinos_path+'_DS','martinos.h5'), 'r')
        data = np.array(HF['%s/%s' %(patient, dataset)]); HF.close()
        return data
    
    
    
#########################################################################
#########################################################################
        
def _normalize(x, axis=None):
    return (x - x.mean(axis=axis, keepdims=True))/(x.std(axis=axis, keepdims=True) + 1e-8)


def normalize(x, axis=(1,2,3)):
    return (x - x.mean(axis=axis, keepdims=True))/(x.std(axis=axis, keepdims=True)+1e-8)
 

def resample_nifti(subject_nifti, resolution=(1.5, 1.5), Nz=16, order=1, mode='nearest', inv=False,CMAC=False):
    """Resample a 3D or 4D (3D+time) cine-MRI nifti to a new in-plane `resolution` with `Nz` slices."""
 
    data   = subject_nifti.get_fdata()
    affine = np.abs(subject_nifti.affine)
    zooms  = subject_nifti.header.get_zooms()[:3]
    
    if inv:
        # This is required for the martinos data 
        zooms = subject_nifti.header.get_zooms()[1:]
    elif CMAC:
        zooms = zooms
    else:
        # This is required for the ACDC diastole/systole/gt data
        if affine[0,0] != zooms[0]:
            affine[0] *= zooms[0]
            affine[1] *= zooms[1]
            affine[2] *= zooms[2]

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

def resample_nifti_inv(data, subject_nifti_resampled, subject_nifti, order=1, mode='nearest'):
    """Resample 3D cine-MRI numpy data to its original resolution"""
    import nibabel as nib
    from dipy.align.reslice import reslice

    affine = subject_nifti_resampled.affine
    zooms  = subject_nifti_resampled.header.get_zooms()[:3]

    new_zooms = subject_nifti.header.get_zooms()[:3]

     
    data_resampled, affine_resampled = reslice(data, affine, zooms, new_zooms, order=order, mode=mode)
    subject_nifti_resampled_inv      = nib.Nifti1Image(data_resampled, affine_resampled)
    #subject_nifti_resampled_inv      = nib.Nifti1Image(data, affine)
    return subject_nifti_resampled_inv  

def pad_to_256x256(data):
    "Pad 4D array of size nx,ny,nz,nt to size size 256x256 in the x,y dimensions"
    nx, ny = data.shape[:2]
    xpad   = (512-nx)//2, (512-nx)-(512-nx)//2
    ypad   = (512-ny)//2, (512-ny)-(512-ny)//2

    start,stop = 512//2-256//2, 512//2+256//2

    pads = (xpad,ypad)+((0,0),)*(len(data.shape)-2)
    vals = ((0,0),)*len(data.shape)
    
    return np.pad(data, pads, 'constant', constant_values=vals)[start:stop,start:stop]

def gauss_mean_variance(M):
    zmax     = np.argmax((M==3).sum(axis=(0,1)))
    cx,cy,cz = center_of_mass(M==3)
    sy=0.15*((M[:,:,zmax]==3).sum(axis=0)>0).sum()
    sx=0.15*((M[:,:,zmax]==3).sum(axis=1)>0).sum()
    return cx,cy,max(sx,sy)    
 
def gauss2d(N=16, mx=0, my=0, sx=1, sy=1):
    X,Y = np.meshgrid(np.linspace(0, 255, 256), np.linspace(0, 255, 256))
    G = np.exp(-((X - mx)**2./(2.*sx**2.)+(Y-my)**2./(2.*sy**2.)))
    return np.stack([G.T]*N,axis=-1)

def gauss3d(M):
    cx,cy,s = gauss_mean_variance(M)
    return gauss2d(N=M.shape[2], mx=cx, my=cy, sx=s, sy=s)


def _roll(x,rx,ry):
    x = np.roll(x,rx,axis=0)
    return np.roll(x,ry,axis=1)

def _roll2center(V,G,M):
    if len(M.shape)==2:
        cx,cy = center_of_mass(M==3)
    else:
        cx,cy,cz = center_of_mass(M==3)
    V = _roll(V, int(128-cx), int(128-cy))
    G = _roll(G, int(128-cx), int(128-cy))
    return V, G

def _crop(V,G):
    if V.shape[0] == 256:
        return V[64:192,64:192], G[64:192,64:192]
    else:
        return V[:,64:192,64:192], G[:,64:192,64:192]
    
def _single_augmentation_subject(x, **kwargs):
    
    import imgaug.augmenters as iaa
    import imgaug as ia

    sometimes     = lambda aug: iaa.Sometimes(0.5, aug)
    gamma         = kwargs.pop('gamma', 1.5);    
    gamma_channel = kwargs.pop('gamma_channel', True)
   
    if gamma_channel:
        joint_params=[sometimes(iaa.OneOf([iaa.GammaContrast(gamma),iaa.GammaContrast(gamma, True)]))
                     ] 
    else:
        joint_params=[sometimes(iaa.GammaContrast(gamma))]         

    xseq = iaa.Sequential(joint_params,name='vcn')
    x    = xseq.augment_images(x[None])[0]
    del xseq
    return x

def _joint_augmentation_subject(x, y, **kwargs):
    
    import imgaug.augmenters as iaa
    import imgaug as ia

    yorder = kwargs.pop('yorder',0);
    rotate = kwargs.pop('angle',360);
    dxy    = kwargs.pop('translate_percent',0.1)
    often_affine = kwargs.pop('translate_percent',0.75)
    sometimes = lambda aug: iaa.Sometimes(often_affine, aug)

    xjoint_params=[sometimes(iaa.Affine(rotate=(-rotate, rotate),
                                       translate_percent={"x":(-dxy,dxy),"y":(-dxy,dxy)}, order=1)),
                   iaa.Fliplr(0.5), 
                   iaa.Flipud(0.5)
                   ]
    yjoint_params=[sometimes(iaa.Affine(rotate=(-rotate, rotate),
                                       translate_percent={"x":(-dxy,dxy),"y":(-dxy,dxy)}, order=yorder)),
                   iaa.Fliplr(0.5), 
                   iaa.Flipud(0.5)
                  ]                 

    xseq = iaa.Sequential(xjoint_params,name='vcn')
    yseq = iaa.Sequential(yjoint_params,name='vcn')
    xseq.reseed()
    yseq.reseed()
    xseq   = xseq.localize_random_state()
    xseq_i = xseq.to_deterministic()
    yseq_i = yseq.to_deterministic()

    yseq_i = yseq_i.copy_random_state(xseq_i, matching='name')

    return xseq_i.augment_images(x[None])[0], yseq_i.augment_images(y[None])[0]
