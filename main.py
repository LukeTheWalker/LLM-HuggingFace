import os
import shutil
import subprocess

# Define the array and the files to be copied
max_local_batch = 15
n_gpus = [1, 2, 3, 4]
files_to_copy = ['LLM.py', 'launch_slurm_llm.sh']
batch_script = 'launch_slurm_llm.sh'

# Base directory
base_dir = '/home/users/lgreco/Development/ML4HPC/proj5'

for gpu in n_gpus:
    # Create a folder for each gpu
    gpu_dir = os.path.join(base_dir, "data/gpu_" + str(gpu))
    os.makedirs(gpu_dir, exist_ok=True)
    for lb in range(1, max_local_batch):
        batch_size = lb * gpu
        
        # Create a folder for each batch_size
        batch_size_dir = os.path.join(gpu_dir, "batch_" + str(batch_size))
        os.makedirs(batch_size_dir, exist_ok=True)
        
        # Read and format the files
        with open(os.path.join(base_dir, 'LLM.py'), 'r') as file:
            file_content = file.read()

        # formatted_content = file_content.format(bs_size_key=str(batch_size))
        formatted_content = file_content.replace('$$bs_size_key$$', str(batch_size))
        formatted_content = formatted_content.replace('$$n_rows$$', str(60))
        
        with open(os.path.join(batch_size_dir, 'LLM.py'), 'w') as file:
            file.write(formatted_content)

        with open(os.path.join(base_dir, 'launch_slurm_llm.sh'), 'r') as file:
            file_content = file.read()

        formatted_content = file_content.replace('$$n_gpus$$', str(gpu))

        with open(os.path.join(batch_size_dir, 'launch_slurm_llm.sh'), 'w') as file:
            file.write(formatted_content)

        # cd into the batch_size directory
        os.chdir(batch_size_dir)

        # Launch the batch script
        subprocess.run(['sbatch', os.path.join(batch_size_dir, batch_script)])

        # cd back to the base directory
        os.chdir(base_dir)