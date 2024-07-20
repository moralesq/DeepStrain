import os
import glob
import pydicom
import aux_dicom
import numpy as np
import pandas as pd
import nibabel as nib
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import warnings
from skimage.measure import find_contours

class Tag:
    def __init__(self, subject_id=1, data_dir='../datasets/CMAC', work_dir='results/data', results_group='MEVIS'):
        
        self.subject_id = subject_id

        self.data_dir = os.path.abspath(data_dir)
        self.work_dir = os.path.abspath(work_dir)
        self.raw_dir     = os.path.join(self.data_dir, 'raw', f'v{subject_id}', '3DTAG')
        self.gt_dir      = os.path.join(self.data_dir, 'GT', '3DTAG', f'v{subject_id}')
        self.results_dir = os.path.join(self.data_dir, 'RESULTS', results_group, '3DTAG', f'v{subject_id}')
        
        self.results_group = results_group

        self.affine_axial = nib.load(os.path.join(self.raw_dir, 'NIFTI', 'NIFTI00.nii')).affine

    def load_pts_gt(self, lmks_obs=1, lmks_frame_id=0):
        """Load GT mesh and landmarks."""
        coords = {
            'MEVIS': 'VTK_COORDINATES',
            'UPF': 'VTK_COORDINATES',
            'IUCL': 'DICOM_COORDINATES',
            'INRIA': 'INRIA_COORDINATES'
        }[self.results_group]

        mesh_gt_mm, region_ids, subpart_ids = load_pts_mm(
            os.path.join(self.gt_dir, 'MESH', coords, f'v{self.subject_id}.vtk'),
            regionID=True, subpartID=True
        )
        self.mesh_gt_mm = mesh_gt_mm
        self.regionIDs = region_ids
        self.subpartIDs = subpart_ids

        self.lmks_gt_mm = load_pts_mm(
            os.path.join(self.gt_dir, 'LMKS', coords, f'obs{lmks_obs}_groundTruth{lmks_frame_id:02d}.vtk')
        )
        self.mesh_gt_mm[:, :2] *= -1
        self.lmks_gt_mm[:, :2] *= -1

    def load_pts(self, frame_id=0, obs=1):
        """Load predicted mesh and landmarks."""
        self.mesh_pred_mm = load_pts_mm(
            os.path.join(self.results_dir, 'MESH', f'finalMesh{frame_id:03d}.vtk')
        )
        self.lmks_pred_mm = load_pts_mm(
            os.path.join(self.results_dir, 'LMKS', f'obs{obs}_results{frame_id:03d}.vtk')
        )
        self.mesh_pred_mm[:, :2] *= -1
        self.lmks_pred_mm[:, :2] *= -1

    def load_nifti_3d(self, frame_id, set_affine=None):
        nifti = nib.load(os.path.join(self.raw_dir, 'NIFTI', f'NIFTI{frame_id:02d}.nii'))
        return nib.Nifti1Image(nifti.get_fdata(), set_affine if set_affine is not None else nifti.affine)

    def load_nifti_4d(self, n_frames=20, set_affine=None):
        shape = self.load_nifti_3d(frame_id=0).shape
        arr_4d = np.zeros(shape + (n_frames,))
        for frame_id in range(n_frames):
            arr_3d = self.load_nifti_3d(frame_id=frame_id).get_fdata()
            arr_4d[:, :, :, frame_id] = arr_3d
        
        affine = set_affine if set_affine is not None else self.affine_axial
        return nib.Nifti1Image(arr_4d, affine)


class Cine:
    def __init__(self, subject_id=1, data_dir='cMAC', work_dir='results/data', results_group='UPF'):
        
        self.subject_id = subject_id
        
        self.data_dir    = os.path.abspath(data_dir)
        self.work_dir    = os.path.abspath(work_dir)
        self.raw_dir     = os.path.join(self.data_dir, 'raw', f'v{subject_id}', 'cSAX')
        self.gt_dir      = os.path.join(self.data_dir, 'GT', 'SSFP', f'v{subject_id}')
        self.results_dir = os.path.join(self.data_dir, 'RESULTS', results_group, 'SSFP', f'v{subject_id}')

        self.results_group = results_group

        self.ijk2ras, self.affine, self.affine_axial = aux_dicom.read_affine_info(os.path.join(self.raw_dir, 'time_1', '*'))

    def load_pts_gt(self, lmks_frame_id=0, obs=1):
        """Load GT mesh and landmarks."""
        coords = {
            'MEVIS': 'VTK_COORDINATES',
            'UPF': 'VTK_COORDINATES',
            'IUCL': 'DICOM_COORDINATES',
            'INRIA': 'INRIA_COORDINATES'
        }[self.results_group]

        mesh_gt_mm, region_ids, subpart_ids = load_pts_mm(
            os.path.join(self.gt_dir, 'MESH', 'VTK_COORDINATES', f'v{self.subject_id}.vtk'),
            regionID=True, subpartID=True
        )
        self.mesh_gt_mm = mesh_gt_mm
        self.regionIDs = region_ids
        self.subpartIDs = subpart_ids

        self.lmks_gt_mm = load_pts_mm(
            os.path.join(self.gt_dir, 'LMKS', coords, f'obs{obs}_groundTruth{lmks_frame_id:02d}.vtk')
        )

    def load_pts(self, frame_id=0, obs=1):
        """Load predicted mesh and landmarks."""
        self.mesh_pred_mm = load_pts_mm(
            os.path.join(self.results_dir, 'MESH', f'finalMesh{frame_id:03d}.vtk')
        )
        self.lmks_pred_mm = load_pts_mm(
            os.path.join(self.results_dir, 'LMKS', f'obs{obs}_results{frame_id:03d}.vtk')
        )

    def load_nifti_3d(self, frame_id, set_affine=None):
        dicom_files = glob.glob(os.path.join(self.raw_dir, f'time_{frame_id + 1}', '*'))
        cine_arrs_2d = [pydicom.dcmread(file).pixel_array.T for file in dicom_files]
        cine_arr_3d = np.stack(cine_arrs_2d, -1)
        return nib.Nifti1Image(cine_arr_3d, set_affine if set_affine is not None else self.affine)

    def load_nifti_4d(self, n_frames=20, set_affine=None):
        shape = self.load_nifti_3d(frame_id=0).shape
        arr_4d = np.zeros(shape + (n_frames,))
        for frame_id in range(n_frames):
            arr_3d = self.load_nifti_3d(frame_id=frame_id).get_fdata()
            arr_4d[:, :, :, frame_id] = arr_3d
        
        affine = set_affine if set_affine is not None else self.affine
        return nib.Nifti1Image(arr_4d, affine)


def load_pts_mm(pts_filename, regionID=False, subpartID=False):
    """Load points (mm) in 3D space stored in vtk file."""
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(pts_filename)
    reader.Update()
    pts_mm = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
    output = [pts_mm]
    if regionID:
        output.append(vtk_to_numpy(reader.GetOutput().GetPointData().GetArray('regionID')))
    if subpartID:
        output.append(vtk_to_numpy(reader.GetOutput().GetPointData().GetArray('subpartID')))
    return output if len(output) > 1 else pts_mm

def pts_pix_to_mm(pts_pix, affine):
    """Apply affine transformation to pixel coordinates."""
    pts_mm = np.dot(np.c_[pts_pix, np.ones(len(pts_pix))], affine.T)[:, :3]
    return pts_mm

def pts_mm_to_pix(pts_mm, affine):
    """Apply inverse affine transformation to mm coordinates."""
    pts_pix = np.dot(np.c_[pts_mm, np.ones(len(pts_mm))], np.linalg.inv(affine).T)[:, :3]
    return pts_pix

def cine_pix_to_tag_pix(cine_pts_pix, cine_affine, tag_affine):
    """Convert cine pixel space points to tag pixel space points."""
    return pts_mm_to_pix(pts_pix_to_mm(cine_pts_pix, cine_affine), tag_affine)

def tag_pix_to_cine_pix(tag_pts_pix, tag_affine, cine_affine):
    """Convert tag pixel space points to cine pixel space points."""
    return pts_mm_to_pix(pts_pix_to_mm(tag_pts_pix, tag_affine), cine_affine)

def cine_cnts_pix_to_tag_cnts_pix(cine_cts_pix, cine_affine, tag_affine):
    """Convert contours in cine pixel space to points in tag pixel space."""
    tag_cts_pix = []
    for slice_contours in cine_cts_pix:
        slice_tag_cts = [cine_pix_to_tag_pix(contour, cine_affine, tag_affine) for contour in slice_contours]
        tag_cts_pix.append(slice_tag_cts)
    return tag_cts_pix

def estimate_affine(ins, outs):
    """Estimate affine transformation matrix."""
    l = len(ins)
    B = np.vstack([np.transpose(ins), np.ones(l)])
    D = 1.0 / np.linalg.det(B)
    entry = lambda r, d: np.linalg.det(np.delete(np.vstack([r, B]), (d + 1), axis=0))
    M = [[(-1) ** i * D * entry(R, i) for i in range(l)] for R in np.transpose(outs)]
    return np.array(M + [[0, 0, 0, 1]])

###############################
####### EXTRA FUNCTIONS #######
###############################

def export_pts_mm_to_slicer_fiducials(pts_mm, fiducials_filename_fscv):
    """Save points (mm) in 3D space as a slicer fiducial file (.fscv)."""
    with open(fiducials_filename_fscv, 'w') as file:
        file.write('# Markups fiducial file version = 4.6\n')
        file.write('# CoordinateSystem = 0\n')
        file.write('# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n')
        for idx, point in enumerate(pts_mm):
            file.write(f'vtkMRMLMarkupsFiducialNode_0,{point[0]},{point[1]},{point[2]},0,0,0,1,1,1,0,F-{idx},,vtkMRMLScalarVolumeNode1\n')

def import_pts_mm_from_slicer_fiducials(fiducials_filename_fscv):
    """Load points (mm) in 3D space from a slicer fiducial file (.fscv)."""
    return pd.read_csv(fiducials_filename_fscv, skiprows=3, header=None).iloc[:, 1:4].to_numpy()

def import_cnts_pix_from_label(label_nifti_filename, label_id=1):
    """Load and convert 3D label of myocardial contours to contour points in pixel space."""
    label = nib.load(label_nifti_filename).get_fdata()
    epicardium, endocardium = [], []
    for z in range(label.shape[-1]):
        contours = find_contours(label[:, :, z] == label_id, 0.8)
        if len(contours) == 2:
            epicardium.append(np.c_[contours[0], z * np.ones((len(contours[0]), 1))])
            endocardium.append(np.c_[contours[1], z * np.ones((len(contours[1]), 1))])
    return epicardium, endocardium
