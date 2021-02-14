# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

import os
import pydicom

EXTENSIONS = {}
EXTENSIONS['NIFTI'] = ['.nii.gz', '.nii']
EXTENSIONS['DICOM'] = ['SAX']
EXTENSIONS['H5PY']  = ['.h5']

def is_data_file(filename, dformat="NIFTI"):
    return any(filename.endswith(extension) for extension in EXTENSIONS[dformat])

def make_dataset(dir, max_dataset_size=float("inf"), dformat="NIFTI"):
    
    filenames = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if dformat == 'DICOM':
                try:
                    protocol_name = pydicom.read_file(os.path.join(root, fname)).ProtocolName
                except:
                    continue
                if any(protocol in protocol_name for protocol in EXTENSIONS[dformat]):
                    path = os.path.join(root, fname)
                    filenames.append(path)
            elif is_data_file(fname, dformat=dformat):
                path = os.path.join(root, fname)
                filenames.append(path)
    return filenames[:min(max_dataset_size, len(filenames))]