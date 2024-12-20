#!/bin/bash
#SBATCH --job-name=singularity
#SBATCH -t 05:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:interactive              # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G 3g.20gb:1                   # requesting GPU slices, see https://docs.hpc.gwdg.de/usage_guide/slurm/gpu_usage/index.html for more options
#SBATCH --nodes=1                   # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task 4            # number of CPU cores per task
#SBATCH --constraint='inet'         #require internet access
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

%environment
    export HTTP_PROXY="http://www-cache.gwdg.de:3128"
    export HTTPS_PROXY="http://www-cache.gwdg.de:3128"

module load singularity
# note: if you start your instance from your home directory the whole home directory will always be mounted into your instance (even with --no-home) 
singularity instance start --env https_proxy="http://www-cache.gwdg.de:3128/" --nv --bind /home/USER/.vscode−server:/.vscode−server,/home/USER/.vscode-server:/.vscode-server,/home/USER/hlrn-access/example_data:/example_data singularity/singularity_example.sif example_instance
# keep the instance and job running since you want to connect to it with ssh and develop in the container
sleep infinity