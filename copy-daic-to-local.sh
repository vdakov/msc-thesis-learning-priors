#!/bin/bash

HPC_USER="vdakov"
HPC_HOST="login.daic.tudelft.nl"
REMOTE_DIR="/home/nfs/vdakov/msc-thesis-learning-priors/results"
LOCAL_DIR="./daic_results"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

echo "Connecting to $HPC_HOST to fetch results..."

# --- The SCP Command ---
# -r: recursive (for folders)
# -p: preserves modification times and modes
scp -i -r "${HPC_USER}@${HPC_HOST}:${REMOTE_DIR}/*" "$LOCAL_DIR"

if [ $? -eq 0 ]; then
    echo "Successfully moved files to $LOCAL_DIR"
else
    echo "Transfer failed. Check your connection or paths."
fi