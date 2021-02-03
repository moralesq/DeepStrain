echo "Note: available models are carson_Jan2021, carmen_Jan2021"
echo "Downloading models ..."
MODEL_DIR=./pretrained_models/

gdown https://drive.google.com/uc?id=1rINpNPZ4_lT9XuFB6Q7gyna_L4O3AIY9 -O $MODEL_DIR
gdown https://drive.google.com/uc?id=10eMGoYYa4xFdwFuiwC7bwVSJ6b-bx7Ni -O $MODEL_DIR
