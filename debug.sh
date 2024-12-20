#!/bin/bash
#SBATCH --job-name=train-nn-gpu
#SBATCH -t 05:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:shared        # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G  A100:1                 #3g.20gb:1              # requesting GPU slices, see https://docs.hpc.gwdg.de/usage_guide/slurm/gpu_usage/index.html for more options
#SBATCH --nodes=1                   # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCh --mem=100GB
#SBATCH --cpus-per-task 4            # number of CPU cores per task
#SBATCH --constraint='inet'         #require internet access
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load anaconda3
module load cuda/11.8

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#module load miniconda3/22.11.1
#module load cuda/12.2.1
source activate sensorium_env # Or whatever you called your environment.


# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"


source .bashrc
sleep infinity
