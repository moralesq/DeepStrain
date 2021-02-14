# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

import numpy as np
import pandas as pd
from medpy.metric.binary import hd, dc

def get_geometric_metrics(M_gt, M_pred, voxelspacing, 
                          tissue_labels=[1, 2, 3], tissue_label_names=['RV','LVM','LV'], phase=0):
    """Calculate the Dice Similarity Coefficient and Hausdorff distance. 
    """

    Dice        = []
    Hausdorff   = []
    TissueClass = []
    for label in tissue_labels:
        TissueClass += [tissue_label_names[label-1]]
        
        gt_label = np.copy(M_gt)
        gt_label[gt_label != label] = 0

        pred_label = np.copy(M_pred)
        pred_label[pred_label != label] = 0

        gt_label   = np.clip(gt_label, 0, 1)
        pred_label = np.clip(pred_label, 0, 1)

        dice      = dc(gt_label, pred_label)
        hausdorff = hd(gt_label, pred_label, voxelspacing=voxelspacing)
        
        Dice.append(dice)
        Hausdorff.append(hausdorff)
        
    output = {'DSC':Dice,'HD':Hausdorff,'TissueClass':TissueClass,'Phase':[phase]*len(tissue_labels)} 
    return pd.DataFrame(output)
        
def get_volume_ml(M, voxel_spacing_mm, tissue_label=1):
    voxel_vol_cm3 = np.prod(voxel_spacing_mm) / 1000
    if len(M.shape) == 4:
        M = np.argmax(M, -1)
    volume_ml = (M==tissue_label).sum()*voxel_vol_cm3
    return volume_ml

def get_mass_g(M, voxel_spacing_mm, tissue_label=2, tissue_density_g_per_ml=1.05):
    volume_ml = get_volume_ml(M, voxel_spacing_mm, tissue_label=tissue_label)
    mass_g    = volume_ml * tissue_density_g_per_ml
    return mass_g

def get_volumes_ml_and_ef(M_ed, M_es, voxel_spacing_mm, tissue_label=1):
    EDV_ml = get_volume_ml(M_ed, voxel_spacing_mm, tissue_label=tissue_label) 
    ESV_ml = get_volume_ml(M_es, voxel_spacing_mm, tissue_label=tissue_label)
    EF     = (EDV_ml-ESV_ml)/EDV_ml
    return EDV_ml, ESV_ml, EF*100

def get_clinical_parameters_rv(M_ed, M_es, voxel_spacing_mm):
    RV_EDV_ml, RV_ESV_ml, RV_EF = get_volumes_ml_and_ef(M_ed, M_es, voxel_spacing_mm, tissue_label=1)
    return RV_EDV_ml, RV_ESV_ml, RV_EF

def get_clinical_parameters_lv(M_ed, M_es, voxel_spacing_mm):
    LV_EDV_ml, LV_ESV_ml, LV_EF = get_volumes_ml_and_ef(M_ed, M_es, voxel_spacing_mm, tissue_label=3)
    LV_mass_g = get_mass_g(M_ed, voxel_spacing_mm, tissue_label=2)
    return LV_EDV_ml, LV_ESV_ml, LV_EF, LV_mass_g

def get_clinical_parameters(M_ed, M_es, voxel_spacing_mm):
    RV_EDV_ml, RV_ESV_ml, RV_EF = get_clinical_parameters_rv(M_ed, M_es, voxel_spacing_mm)
    LV_EDV_ml, LV_ESV_ml, LV_EF, LV_mass_g = get_clinical_parameters_lv(M_ed, M_es, voxel_spacing_mm)
    clinical_parameters = {'RV_EDV_ml':RV_EDV_ml, 'RV_ESV_ml':RV_ESV_ml, 'RV_EF':RV_EF, 
                           'LV_EDV_ml':LV_EDV_ml, 'LV_ESV_ml':LV_ESV_ml, 'LV_EF':LV_EF, 'LV_mass_g':LV_mass_g}
    return clinical_parameters


## Stats

def clinical_metrics_statistics(x, y):
    """Calculate correlation (corr), bias, standard deviation (std), mean absolute error between x and y measures. 
    
    Bias: The bias between the two tests is measured by the mean of the differences. 
    std : The standard deviation (also known as limits of agreement) between the two tests are defined by a 95% 
          prediction interval of a particular value of the difference.
    
    See: https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Bland-Altman_Plot_and_Analysis.pdf

    """
    dk   = x-y
    bias = np.mean(dk)
    std  = np.sqrt(np.sum((dk-bias)**2)/(len(x)-1))
    mae  = np.mean(np.abs(dk))
    return bias, std, mae, x.corrwith(y)

def get_clinical_metrics_on_dataloader(loading_fn, listSIDs, ED_ids=0, ES_ids=1):
    """Calculate clinical metrics on data loader function `loading_fn` for subjects in `listSIDs`.
    Assumes end-diastole and end-systole time frame = `end_diastolic_frame_id`, `end_systolic_frame_id`.
    """
    Clinical_params_pred = pd.DataFrame({'RV_EDV_ml':[], 'RV_ESV_ml':[], 'RV_EF':[], 
                                         'LV_EDV_ml':[], 'LV_ESV_ml':[], 'LV_EF':[], 'LV_mass_g':[]})
    if type(ED_ids) == int: ED_ids = [ED_ids] * len(listSIDs)
    if type(ES_ids) == int: ES_ids = [ES_ids] * len(listSIDs)    
    for subject_id, ED_id, ES_id in zip(listSIDs, ED_ids, ES_ids):
        V, M_pred_ed, affine, zooms = loading_fn(subject_id, ED_id)
        V, M_pred_es, affine, zooms = loading_fn(subject_id, ES_id)

        clinical_params_pred = get_clinical_parameters(np.argmax(M_pred_ed,-1), 
                                                               np.argmax(M_pred_es,-1), 
                                                               voxel_spacing_mm=zooms[:3])

        Clinical_params_pred = Clinical_params_pred.append(clinical_params_pred,ignore_index=True)

    Clinical_params_pred.index = pd.Index(listSIDs, name='SubjectID') 

    return Clinical_params_pred

def compare_clinical_metrics_on_dataloader(loading_fn, listSIDs, ED_ids=0, ES_ids=1):
    """Calculate clinical metrics on data loader function `loading_fn` for subjects in `listSIDs`.
    Assumes end-diastole and end-systole time frame = `end_diastolic_frame_id`, `end_systolic_frame_id`.
    """
    Clinical_params_gt   = pd.DataFrame({'RV_EDV_ml':[], 'RV_ESV_ml':[], 'RV_EF':[], 
                                         'LV_EDV_ml':[], 'LV_ESV_ml':[], 'LV_EF':[], 'LV_mass_g':[]})
    Clinical_params_pred = Clinical_params_gt.copy()
    
    if type(ED_ids) == int: ED_ids = [ED_ids] * len(listSIDs)
    if type(ES_ids) == int: ES_ids = [ES_ids] * len(listSIDs)    
    for subject_id, ED_id, ES_id in zip(listSIDs, ED_ids, ES_ids):
        V, M_ed, M_pred_ed, affine, zooms = loading_fn(subject_id, ED_id)
        V, M_es, M_pred_es, affine, zooms = loading_fn(subject_id, ES_id)
        
        clinical_params_gt   = get_clinical_parameters(np.argmax(M_ed,-1), 
                                                               np.argmax(M_es,-1), 
                                                               voxel_spacing_mm=zooms[:3])
        clinical_params_pred = get_clinical_parameters(np.argmax(M_pred_ed,-1), 
                                                               np.argmax(M_pred_es,-1), 
                                                               voxel_spacing_mm=zooms[:3])

        Clinical_params_gt   = Clinical_params_gt.append(clinical_params_gt,ignore_index=True)
        Clinical_params_pred = Clinical_params_pred.append(clinical_params_pred,ignore_index=True)

    Clinical_params_gt.index   = pd.Index(listSIDs, name='SubjectID') 
    Clinical_params_pred.index = pd.Index(listSIDs, name='SubjectID') 
    
    stats_df = clinical_metrics_statistics(Clinical_params_gt,Clinical_params_pred)
    stats_df = pd.DataFrame(stats_df,index=['bias','std','MAE','corr']).T[['corr','bias','std','MAE']]

    return Clinical_params_gt, Clinical_params_pred, stats_df


def compare_geometric_metrics_on_dataloader(loading_fn, listSIDs, listTimeFrames,
                                    tissue_labels=[1, 2, 3], tissue_label_names=['RV','LVM','LV']):
    """Calculate geometric metrics on data loader function `loading_fn` for subjects in `listSIDs`.
    Metrics are calculated for all frames in `listTimeFrames`.
    """
    Geometric_params = pd.DataFrame({'DSC':[],'HD':[],'TissueClass':[], 'Phase':[]})    
    for subject_id in listSIDs:
        for time_frame in listTimeFrames:
            V, M, M_pred, affine, zooms = loading_fn(subject_id, time_frame)
        
            # GEOMETRIC METRICS
            geometric_metrics = get_geometric_metrics(np.argmax(M,-1), np.argmax(M_pred,-1), 
                                                      voxelspacing=zooms[:3], phase=time_frame,
                                                      tissue_labels=tissue_labels,tissue_label_names=tissue_label_names)

            Geometric_params = Geometric_params.append(geometric_metrics, ignore_index=True)
        
    Geometric_params.index = pd.Index(np.repeat(listSIDs, len(tissue_labels)*len(listTimeFrames)), name='SubjectID') 
    return Geometric_params



