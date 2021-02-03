MODEL_DIR_3D=./datasets/sample_nifti_3D/
MODEL_DIR_4D=./datasets/sample_nifti_4D/
mkdir $MODEL_DIR_3D
mkdir $MODEL_DIR_4D

gdown https://drive.google.com/uc?id=1QZNgRojYpYBLzUQJntWAmw1QwQMh4H50 -O $MODEL_DIR_3D
gdown https://drive.google.com/uc?id=1zFJM_qQKwz85xiYpX3XBRqhL0SQwy-Iw -O $MODEL_DIR_3D
gdown https://drive.google.com/uc?id=1FqTquCYhLD2-EKxmCR9A5zt5265AEPdQ -O $MODEL_DIR_4D