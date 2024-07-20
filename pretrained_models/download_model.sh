#!/bin/bash
echo "Note: available models are carson_Jan2021, carmen_Jan2021"
echo "Downloading weights from the Harvard Dataverse: https://doi.org/10.7910/DVN/XB6PEZ"

# Define the URLs of the files to download
url1="https://dataverse.harvard.edu/api/access/datafile/10358815"
url2="https://dataverse.harvard.edu/api/access/datafile/10358816"

# Define the directory to save the files
MODEL_DIR="./pretrained_models"

# Define the filenames
filename1="carmen_Jan2021.h5"
filename2="carson_Jan2021.h5"

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

