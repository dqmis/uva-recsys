#!/bin/bash

# Create a folder called data and move into it
mkdir -p data
cd data

# List of files to download
files=(
    "ebnerd_demo.zip"
    "ebnerd_small.zip"
    "ebnerd_large.zip"
    "articles_large_only.zip"
    "ebnerd_testset.zip"
    "predictions_large_random.zip"
    "artifacts/Ekstra_Bladet_word2vec.zip"
    "artifacts/Ekstra_Bladet_image_embeddings.zip"
    "artifacts/Ekstra_Bladet_contrastive_vector.zip"
    "artifacts/google_bert_base_multilingual_cased.zip"
    "artifacts/FacebookAI_xlm_roberta_base.zip"
)

# Download and unzip the files
for file in "${files[@]}"; do
    # Create subdir path
    subdir=$(dirname "$file")
    mkdir -p "$subdir"

    # Change to subdir for downloading
    cd "$subdir"

    # Extract filename and directory name
    filename=$(basename "$file")
    dirname="${filename%.zip}"

    # Check if the file already exists
    if [ ! -f "$filename" ]; then
        # Download the file
        wget "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/$file"

        # Create directory for extraction and unzip there
        mkdir -p "$dirname"
        unzip "$filename" -d "$dirname"
    else
        echo "$filename already exists in $subdir"
    fi

    # Move back to the data directory
    cd - > /dev/null
done