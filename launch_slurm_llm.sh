#!/bin/sh -l
#SBATCH --partition=gpu
#SBATCH --gpus-per-node $$n_gpus$$ # <---- number of gpus per node
#SBATCH -c 24
#SBATCH -t 40
#SBATCH -N 1  # <------ number of nodes. Keep it to '1' because it does not work. "AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'".
#SBATCH --export=ALL

# get host name
hosts_file="hosts.txt"
scontrol show hostname $SLURM_JOB_NODELIST > $hosts_file

# Collect public key and accept them
while read -r node; do
    ssh-keyscan "$node" >> ~/.ssh/known_hosts
done < "$hosts_file"

# Create the host file containing node names and the number of GPUs
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=4 if $slots==0;
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile

source /work/projects/ulhpc-tutorials/PS10-Horovod/env_ds.sh

# Launch HuggingFace+DeepSpeed code by varying the number of GPUs
deepspeed --num_gpus $$n_gpus$$ --num_nodes 1  --hostfile hostfile LLM.py





