import os


EXTENSIONS = {}
EXTENSIONS['NIFTI'] = ['.nii.gz', '.nii']
EXTENSIONS['DICOM'] = ['.dcm']

def is_data_file(filename, dformat="NIFTI"):
    return any(filename.endswith(extension) for extension in EXTENSIONS[dformat])

def make_dataset(dir, max_dataset_size=float("inf"), dformat="NIFTI"):
    
    filenames = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_data_file(fname, dformat=dformat):
                path = os.path.join(root, fname)
                filenames.append(path)
    return filenames[:min(max_dataset_size, len(filenames))]

