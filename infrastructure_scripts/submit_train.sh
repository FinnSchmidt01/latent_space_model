#!/bin/bash
#SBATCH --job-name=train-nn-gpu
#SBATCH -t 12:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:shared     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH --gres=gpu:A100:1                          #3g.20gb:1                # requesting GPU slices, see https://docs.hpc.gwdg.de/usage_guide/slurm/gpu_usage/index.html for more options
#SBATCH --nodes=1                   # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task 5            # number of CPU cores per task
#SBATCH --mem=110GB
#SBATCH --constraint='inet'         #require internet access
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#module load miniconda3/22.11.1
#module load cuda/12.2.1
module load anaconda3
module load cuda/11.8

module --ignore_cache load "anaconda3"
#source activate sensorium_env # Or whatever you called your environment.
source /sw/tools/python/anaconda3/2020.11/skl/etc/profile.d/conda.sh
conda activate /mnt/lustre-grete/usr/u11302/.conda/envs/sensorium_env

#source activate /mnt/lustre-grete/usr/u11302/.conda/envs/sensorium_env

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Run the script:
#python -u latent_space_model.py
python -u eval.py
#python -u heat_maps.py
#python -u cca_plots.py
# Run the script with logger:
#python -u train_with_logger.py -l ~/${SLURM_JOB_NAME}_${SLURM_JOB_ID}  -t True -p True -d True -s True -f True
