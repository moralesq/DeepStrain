set -ex

DATAROOT=$1
DATAFORMAT=$2
RESULTS_DIR=$3

CARSON_PATH='./pretrained_models/carson_Jan2021.h5'
CARMEN_PATH='./pretrained_models/carmen_Jan2021.h5'
PIPELINE='segmentation_motion'

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ${DATAROOT} \
  --dataformat ${DATAFORMAT} \
  --results_dir ${RESULTS_DIR} \
  --pretrained_models_netS ${CARSON_PATH} \
  --pretrained_models_netME ${CARMEN_PATH} \
  --pipeline ${PIPELINE}
  
  