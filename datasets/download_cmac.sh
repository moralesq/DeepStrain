#!/bin/bash
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
  local subject_version=-1

  for arg in "$@"; do
    case $arg in
      gt)
        download_gt=true
        ;;
      results)
        download_results=true
        ;;
      v*)
        subject_version=${arg:1}
        download_subjects=true
        ;;
      -1)
        subject_version=-1
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
    if [ "$subject_version" == "-1" ]; then
      for i in "${!urls[@]}"; do
        download_and_unzip_version "$i"
      done
    else
      if (( subject_version >= 1 && subject_version <= ${#urls[@]} )); then
        download_and_unzip_version "$((subject_version - 1))"
      else
        echo "Invalid subject version: v$subject_version"
        exit 1
      fi
    fi
  fi
}

# Function to download and unzip a specific version
download_and_unzip_version() {
  local version_index=$1
  local url=${urls[$version_index]}
  local filename="CMAC_raw_v$((version_index + 1)).zip"
  download_file "$url" "$filename" "$RAW_DIR"
  unzip_and_cleanup "$filename" "$RAW_DIR"
}

# Main script execution
if [ $# -eq 0 ]; then
  process_arguments gt results -1
else 
  process_arguments "$@"
fi
