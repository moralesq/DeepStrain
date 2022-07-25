import os
import glob
import time
import pydicom
import numpy as np
import pandas as pd
import nibabel as nib

PREPARE_INPUT_DATA_WITH_CARSON = False
PREDICT = False

if PREPARE_INPUT_DATA_WITH_CARSON:
    
    from data import base_dataset
    from data.nifti_dataset import resample_nifti
    from tensorflow.keras.optimizers import Adam
    from options.test_options import TestOptions
    from models import deep_strain_model

    def normalize(x, axis=(0,1,2)):
        # normalize per volume (x,y,z) frame
        mu = x.mean(axis=axis, keepdims=True)
        sd = x.std(axis=axis, keepdims=True)
        return (x-mu)/(sd+1e-8)

    def get_mask(V, netS):
        nx, ny, nz, nt = V.shape
        
        M = np.zeros((nx,ny,nz,nt))
        v = V.transpose((2,3,0,1)).reshape((-1,nx,ny)) # (nz*nt,nx,ny)
        v = normalize(v)
        m = netS(v[:,nx//2-64:nx//2+64,ny//2-64:ny//2+64,None])
        M[nx//2-64:nx//2+64,ny//2-64:ny//2+64] += np.argmax(m, -1).transpose((1,2,0)).reshape((128,128,nz,nt))
        
        return M

    # options
    opt   = TestOptions().parse()
    model = deep_strain_model.DeepStrain(Adam, opt)
    netS  = model.get_netS()
    netS.load_weights('/home/mmorales/main_python/DeepStrain/pretrained_models/carson_Jan2021.h5')

    time_resample = []
    time_carson   = []
    
    # load subjects by batches
    batches = ['batch_%d'%(j) for j in range(1,11)] + ['HFpEF_batch_%d'%(j) for j in range(1,5)]

    for batch in batches:

        niftis_folder    = '/mnt/alp/Research Data Sets/DeepStrain_vs_CVI/%s/niftis/standard'%(batch)
        niftis_folder_out_carson = '/mnt/alp/Research Data Sets/DeepStrain_vs_CVI/%s/input_to_DeepStrain_with_CarSON'%(batch)

        for SubjectID_folder in glob.glob(os.path.join(niftis_folder, '*')):
            for nifti_path in glob.glob(os.path.join(SubjectID_folder, '*.nii.gz')):
                
                try:
                    V_nifti = nib.load(nifti_path)
                    start = time.time()
                    V_nifti_resampled = resample_nifti(V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=None)
                    end = time.time()
                    time_resample += [end - start]

                    # here we normalize per image, not volume
                    V = V_nifti_resampled.get_fdata()
                    V = normalize(V, axis=(0,1))

                    # In this case we don't yet have a segmentation we can use to crop the image. 
                    # In most cases we can simply center crop (see `get_mask` function): 
                    start = time.time()
                    M = get_mask(V, netS)
                    end = time.time()
                    time_carson += [end - start]

                    # ONLY IF YOU KNOW YOUR IMAGE IS ROUGHLY NEAR CENTER 
                    M_nifti_resampled = nib.Nifti1Image(M, affine=V_nifti_resampled.affine)
                    # resample back to original resolution
                    start = time.time()
                    M_nifti = base_dataset.resample_nifti_inv(nifti_resampled=M_nifti_resampled, 
                                                              zooms=V_nifti.header.get_zooms()[:3], 
                                                              order=0, mode='nearest')
                    end = time.time()
                    time_resample += [end - start]
                    fname = os.path.basename(nifti_path).strip('.nii.gz').replace('(','').replace(')','')
                    output_folder = os.path.join(niftis_folder_out_carson, os.path.basename(SubjectID_folder))

                    os.makedirs(output_folder, exist_ok=True)

                    V_nifti.to_filename(os.path.join(output_folder, fname+'.nii.gz'))
                    M_nifti.to_filename(os.path.join(output_folder, fname+'_segmentation.nii.gz'))
                except:
                    print("Error here, check!", nifti_path)
                    continue

    np.save('/mnt/alp/Research Data Sets/DeepStrain_vs_CVI/time_resample', time_resample)
    np.save('/mnt/alp/Research Data Sets/DeepStrain_vs_CVI/time_carson', time_carson)





if PREDICT:

    from data.nifti_dataset import resample_nifti
    from data.base_dataset import _roll2center_crop
    from scipy.ndimage.measurements import center_of_mass

   
    from aux import myocardial_strain
    from scipy.ndimage import gaussian_filter

    from tensorflow.keras.optimizers import Adam
    from options.test_options import TestOptions
    from models import deep_strain_model

    def normalize(x):
        # normalize per volume (x,y,z) frame
        mu = x.mean(axis=(0,1,2), keepdims=True)
        sd = x.std(axis=(0,1,2), keepdims=True)
        return (x-mu)/(sd+1e-8)

    # options
    opt = TestOptions().parse()
    preprocess = opt.preprocess
    model   = deep_strain_model.DeepStrain(Adam, opt)
    
    opt.number_of_slices = 16 
    opt.preprocess = opt.preprocess_carmen + '_' + preprocess
    opt.pretrained_models_netME = '/home/mmorales/main_python/DeepStrain/pretrained_models/carmenJan2021.h5'
    model   = deep_strain_model.DeepStrain(Adam, opt)
    netME   = model.get_netME()
    netME.load_weights('/home/mmorales/main_python/DeepStrain/pretrained_models/carmen_Jan2021.h5')

    batches = ['batch_%d'%(j) for j in range(1,11)] + ['HFpEF_batch_%d'%(j) for j in range(1,5)]

    # calculate using CarSON segmentations. Note that segmentations based on other segmentation models is also possible
    for method in ['_with_CarSON']:    
        # verify these labels!
        if method == '_with_CarSON':
            tissue_label_blood_pool=3; tissue_label_myocardium=2; tissue_label_rv=1
        else:
            tissue_label_blood_pool=1; tissue_label_myocardium=2; tissue_label_rv=3
            
        for batch in batches:
            print(batch)
            # only use data whose cines and corresponding segmentations have been prepared
            niftis_folder_out = '/mnt/alp/Research Data Sets/DeepStrain_vs_CVI/%s/input_to_DeepStrain%s'%(batch, method)

            RUN_CARMEN = True
            if RUN_CARMEN:
                for SubjectID_folder in glob.glob(os.path.join(niftis_folder_out, '*')):
                    
                    for nifti_path in glob.glob(os.path.join(SubjectID_folder, '*_segmentation.nii.gz')):

                        output_folder = os.path.join(os.path.dirname(niftis_folder_out), 
                                                    'output_from_DeepStrain%s'%(method),
                                                    os.path.basename(SubjectID_folder))
                        
                        if os.path.isdir(output_folder): continue

                        print(output_folder)

                        V_nifti = nib.load(nifti_path.replace('_segmentation', ''))
                        M_nifti = nib.load(nifti_path)

                        V_nifti = resample_nifti(V_nifti, order=1, number_of_slices=16)
                        M_nifti = resample_nifti(M_nifti, order=0, number_of_slices=16)
                        
                        

                        center = center_of_mass(M_nifti.get_fdata()==tissue_label_myocardium)
                        V = _roll2center_crop(x=V_nifti.get_fdata(), center=center)
                        M = _roll2center_crop(x=M_nifti.get_fdata(), center=center)

                        I = np.argmax((M==tissue_label_rv).sum(axis=(0,1,3)))
                        if I > M.shape[2]//2:
                            print('Apex to Base. Inverting.')
                            V = V[:,:,::-1]
                            M = M[:,:,::-1]
                        
                        V = normalize(V)

                        nx, ny, nz, nt = V.shape

                        try:
                            # calculate volume across the mid-ventricular section to estimate end-diastole
                            volumes = (M_nifti.get_fdata()[:,:,nz//2-2:nz+3]==tissue_label_blood_pool).sum(axis=(0,1,2))
                        except:
                            print('Need to use all volume to estimate ED/ES')
                            volumes = (M_nifti.get_fdata()==tissue_label_blood_pool).sum(axis=(0,1,2))

                        ED = np.argmax(volumes)
                        ES = np.argmin(volumes)
                        
                        # set end-diastole as the reference time frame
                        M_0 = M[..., ED]
                        V_0 = np.repeat(np.expand_dims(V[..., ED],-1), nt, axis=-1)
                        V_t = V

                        # move time frames to the batch dimension to predict all at onces
                        V_0 = np.transpose(V_0, (3,0,1,2))
                        V_t = np.transpose(V_t, (3,0,1,2))
                        y_t = netME([V_0, V_t]).numpy()

                        
                        os.makedirs(output_folder, exist_ok=True)

                        # save for calculation. Only the the end-diastolic mask is necessary
                        np.save(os.path.join(output_folder, 'V_0.npy'), V_0)
                        np.save(os.path.join(output_folder, 'V_t.npy'), V_t)
                        np.save(os.path.join(output_folder, 'y_t.npy'), y_t)
                        np.save(os.path.join(output_folder, 'M_0.npy'), M_0)


                
            folder = '/mnt/alp/Research Data Sets/DeepStrain_vs_CVI/%s/output_from_DeepStrain%s'%(batch, method)

            df = {'SubjectID':[], 'RadialStain':[], 'CircumferentialStrain':[], 'TimeFrame':[]}
            for j, subject_folder in enumerate(glob.glob(os.path.join(folder, '*'))):
                V_0 = np.load(os.path.join(subject_folder, 'V_0.npy'))
                V_t = np.load(os.path.join(subject_folder, 'V_t.npy'))
                y_t = np.load(os.path.join(subject_folder, 'y_t.npy'))
                M_0 = np.load(os.path.join(subject_folder, 'M_0.npy'))

                y_t = gaussian_filter(y_t, sigma=(0,2,2,0,0))

                for time_frame in range(len(y_t)):
                    try:
                        strain = myocardial_strain.MyocardialStrain(mask=M_0, flow=y_t[time_frame,:,:,:,:])
                        strain.calculate_strain(lv_label=tissue_label_blood_pool)

                        df['SubjectID']             += [os.path.basename(subject_folder)]
                        df['RadialStain']           += [100*strain.Err[strain.mask_rot==tissue_label_myocardium].mean()]   
                        df['CircumferentialStrain'] += [100*strain.Ecc[strain.mask_rot==tissue_label_myocardium].mean()]
                        df['TimeFrame']             += [time_frame]
                    except:
                        print('Error in ', subject_folder)

            df = pd.DataFrame(df)
            df.to_csv('/mnt/alp/Research Data Sets/DeepStrain_vs_CVI/%s/output_from_DeepStrain%s.csv'%(batch, method))

