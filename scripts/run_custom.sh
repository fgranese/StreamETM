#!/bin/bash

####### important : chmod +x run.sh to make it executable ########
####### ./run.sh to run the script ########

# Bash script to process all CSV files in the /documents folder with the topic modeling script

# Set the directory containing the documents
DOCUMENTS_DIR="/home/fgranese/StreamETM/data/custom"

# Set the language parameter
CONFIG="/home/fgranese/StreamETM/config/config_custom.yaml"

# Path to the Python script
SCRIPT="/home/fgranese/StreamETM/main.py"

# Loop over all CSV files in the documents directory
for FILE in "$DOCUMENTS_DIR"/chunk_*.csv; do
    if [ -f "$FILE" ]; then
        echo "Processing $FILE..."
        python "$SCRIPT"  --config "$CONFIG" --documents "$FILE"
    else
        echo "No CSV files found in $DOCUMENTS_DIR."
    fi
done
