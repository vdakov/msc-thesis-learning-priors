#!/bin/sh
#SBATCH --job-name="PFN Prior Learning Training"
#SBATCH --account="ewi-insy-prb"
#SBATCH --partition="insy"      # Request partition.
#SBATCH --time=01:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --tasks-per-node=1         # Set one task per node
#SBATCH --cpus-per-task=1          # Request number of CPUs (threads) per task.
#SBATCH --gres=gpu                 # Request 1 GPU
#SBATCH --mem=8GB                  # Request 4 GB of RAM in total
#SBATCH --mail-type=END            # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=slurm-%x-%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm-%x-%j.err    # Set name of error log. %j is the Slurm jobId

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=================================================="
echo ""

# Print allocated resources
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo ""

export APPTAINER_IMAGE="../containers/msc-thesis-learning-priors_latest.sif"

# If you use GPUs
module use /opt/insy/modulefiles
module load cuda/12.1

# Run script
srun  apptainer exec --nv  ../containers/msc-thesis-learning-priors_latest.sif python src/training_script.py --config src/configs/one_epoch_prior_learning.yaml

# --nv binds NVIDIA libraries from the host (only if you use CUDA)
# --env-file source additional environment variables from e.g. .env file (optional)
# -B /$HOME:/$HOME/ mounts host file-sytem inside container
# The home folder should be mounted by default, but sometimes it is not
# -B can be used several times, change this to match your cluster file-system
# APPTAINER_IMAGE is the full path to the container.sif file
# python script.py is the command that you want to use inside the container

~                                                                                                 