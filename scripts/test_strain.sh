set -ex

RESULTS_DIR=$1
PIPELINE='strain'

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ${RESULTS_DIR} \
  --results_dir ${RESULTS_DIR} \
  --pipeline ${PIPELINE}