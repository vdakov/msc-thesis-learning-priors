#SBATCH --job-name="test-container-jobs"
#SBATCH --account=ewi-insy-reit
#SBATCH --partition=general
#SBATCH --ntasks=4 –-cpus-per-task=4
#SBATCH --mem-per-cpu=1000
#SBATCH --gres=gpu
#SBATCH --time=10:00
#SBATCH --qos=short

export APPTAINER_PATH="/tudelft.net/staff-umbrella/reit/apptainer/pytorch2.2.1-cuda12.1.sif"