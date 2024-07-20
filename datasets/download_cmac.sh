#!/bin/bash
# This script downloads the CMAC dataset from the Harvard Dataverse.
# Usage:
#   ./download_cmac_dataset.sh                  # Downloads the entire CMAC dataset
#   ./download_cmac_dataset.sh gt results -1    # Downloads GT, results, and all subjects
#   ./download_cmac_dataset.sh gt               # Downloads only the ground truth (GT) data
#   ./download_cmac_dataset.sh results          # Downloads only the results data
#   ./download_cmac_dataset.sh gt v1            # Downloads only the ground truth (GT) data and subject volunteer v1
#   ./download_cmac_dataset.sh results v1       # Downloads only the results data and subject volunteer v1
#   ./download_cmac_dataset.sh gt results v1    # Downloads GT, results, and subject volunteer v1
#
# Parameters:
#   gt        - Download ground truth data
#   results   - Download results data
#   v1, v2, etc. - Download specific subject volunteer
#   -1        - Download all subject volunteers

echo "Downloading CMAC dataset from the Harvard Dataverse: https://doi.org/10.7910/DVN/XB6PEZ"

# Define the URLs of the ground-truth and results files to download
url1="https://dataverse.harvard.edu/api/access/datafile/10394992"
url2="https://dataverse.harvard.edu/api/access/datafile/10394993"


# Define the URLs of the raw files to download
urls=(
    "https://dataverse.harvard.edu/api/access/datafile/10394994"
    "https://dataverse.harvard.edu/api/access/datafile/10394995"
    ""
    "https://dataverse.harvard.edu/api/access/datafile/10394996"
    "https://dataverse.harvard.edu/api/access/datafile/10395521"
    "https://dataverse.harvard.edu/api/access/datafile/10395522"
    "https://dataverse.harvard.edu/api/access/datafile/10395524"
    "https://dataverse.harvard.edu/api/access/datafile/10395525"
    "https://dataverse.harvard.edu/api/access/datafile/10395526"
    "https://dataverse.harvard.edu/api/access/datafile/10395527"
    "https://dataverse.harvard.edu/api/access/datafile/10395528"
    "https://dataverse.harvard.edu/api/access/datafile/10395529"
    "https://dataverse.harvard.edu/api/access/datafile/10395555"
    "https://dataverse.harvard.edu/api/access/datafile/10395558"
    "https://dataverse.harvard.edu/api/access/datafile/10395560"
    "https://dataverse.harvard.edu/api/access/datafile/10395561"
)

# Define the directory to save the files
MODEL_DIR="./datasets/CMAC"
RAW_DIR="$MODEL_DIR/raw"

# Create the directory if it doesn't exist
mkdir -p "$MODEL_DIR"
mkdir -p "$RAW_DIR"

# Function to download a file
download_file() {
  local url=$1
  local filename=$2
  local directory=$3
  echo "-L -o $directory/$filename '$url'"
  curl -L -o "$directory/$filename" "$url"
  
  # Check if the download was successful
  if [ $? -eq 0 ]; then
      echo "Download of $filename completed successfully. Saved to $directory/$filename"
  else
      echo "Download of $filename failed."
      exit 1
  fi
}

# Function to unzip a file and delete the zip file and __MACOSX folder
unzip_and_cleanup() {
  local filename=$1
  local directory=$2
  unzip "$directory/$filename" -d "$directory"
  
  # Check if the unzip was successful
  if [ $? -eq 0 ]; then
      echo "Unzipping of $filename completed successfully."
      # Delete the zip file
      rm "$directory/$filename"
      echo "$filename has been deleted."
      # Delete the __MACOSX folder if it exists
      if [ -d "$directory/__MACOSX" ]; then
          rm -rf "$directory/__MACOSX"
          echo "__MACOSX folder has been deleted."
      fi
  else
      echo "Unzipping of $filename failed."
      exit 1
  fi
}

# Function to process arguments and call the appropriate download functions
process_arguments() {
  local download_gt=false
  local download_results=false
  local download_subjects=false
  local subject_volunteer=-1

  for arg in "$@"; do
    case $arg in
      gt)
        download_gt=true
        ;;
      results)
        download_results=true
        ;;
      v*)
        subject_volunteer=${arg:1}
        download_subjects=true
        ;;
      -1)
        subject_volunteer=-1
        download_subjects=true
        ;;
    esac
  done

  # Download results if requested
  if $download_results; then
    download_file "$url1" "CMAC_RESULTS.zip" "$MODEL_DIR"
    unzip_and_cleanup "CMAC_RESULTS.zip" "$MODEL_DIR"
  fi

  # Download gt if requested
  if $download_gt; then
    download_file "$url2" "CMAC_GT.zip" "$MODEL_DIR"
    unzip_and_cleanup "CMAC_GT.zip" "$MODEL_DIR"
  fi

  # Download subjects if requested
  if $download_subjects; then
    if [ "$subject_volunteer" == "-1" ]; then
      for i in "${!urls[@]}"; do
        download_and_unzip_volunteer "$i"
      done
    else
      if (( subject_volunteer >= 1 && subject_volunteer <= ${#urls[@]} )); then
        download_and_unzip_volunteer "$((subject_volunteer - 1))"
      else
        echo "Invalid subject volunteer: v$subject_volunteer"
        exit 1
      fi
    fi
  fi
}

# Function to download and unzip a specific volunteer
download_and_unzip_volunteer() {
  local volunteer_index=$1
  local url=${urls[$volunteer_index]}
  local filename="CMAC_raw_v$((volunteer_index + 1)).zip"
  # Check if the URL is not empty
  if [ -n "$url" ]; then
    download_file "$url" "$filename" "$RAW_DIR"
    unzip_and_cleanup "$filename" "$RAW_DIR"
  else
    echo "Skipping download for volunteer v$((volunteer_index + 1)) because this was not provided in the dataset."
  fi

}

# Main script execution
if [ $# -eq 0 ]; then
  process_arguments gt results -1
else 
  process_arguments "$@"
fi
