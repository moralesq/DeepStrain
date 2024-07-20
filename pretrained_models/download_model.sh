#!/bin/bash
echo "Note: available models are carson_Jan2021, carmen_Jan2021"
echo "Downloading weights from the Harvard Dataverse: https://doi.org/10.7910/DVN/XB6PEZ"

# Define the URLs of the files to download
url1="https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/XB6PEZ/19069226bf4-5bebf2cddd86?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27carson_Jan2021.h5&response-content-type=application%2Fx-hdf5&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240720T171428Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20240720%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=b4232fc17b53238fa71f8b0dfa85e6f700a045dbc7549f66cd6da52902efed71"
url2="https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/XB6PEZ/19069239dab-9fc74c404789?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27carmen_Jan2021.h5&response-content-type=application%2Fx-hdf5&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240720T173020Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20240720%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e4d61cdfb182049855717365479c57adf5dcd8166177b5c7d6db3ab7ebd81424"

# Define the directory to save the files
MODEL_DIR="./pretrained_models"

# Define the filenames
filename1="carson_Jan2021.h5"
filename2="carmen_Jan2021.h5"

# Create the directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Function to download a file
download_file() {
  local url=$1
  local filename=$2
  curl -L -o "$MODEL_DIR/$filename" "$url"
  
  # Check if the download was successful
  if [ $? -eq 0 ]; then
      echo "Download of $filename completed successfully. Saved to $MODEL_DIR/$filename"
  else
      echo "Download of $filename failed."
      exit 1
  fi
}

# Download the files
download_file "$url1" "$filename1"
download_file "$url2" "$filename2"

