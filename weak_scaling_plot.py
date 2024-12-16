import os
import re

import matplotlib.pyplot as plt

def read_training_time(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if "Training time:" in line:
                return float(line.split("Training time:")[1].strip().split()[0])
    return None

def collect_data(base_dir):
    data = {}
    base_gpu_1_dir = os.path.join(base_dir, 'base_gpu_1')

    for gpu_folder in os.listdir(base_dir):
        if gpu_folder.startswith('gpu_') and gpu_folder != 'base_gpu_1':
            gpu_path = os.path.join(base_dir, gpu_folder)
            for row_folder in os.listdir(gpu_path):
                if row_folder.startswith('nrows'):
                    row_path = os.path.join(gpu_path, row_folder)
                    timings_file = os.path.join(row_path, 'timings.txt')
                    training_time = read_training_time(timings_file)
                    if training_time is not None:
                        num_gpus = int(gpu_folder.split('_')[1])
                        num_rows = int(row_folder.split('nrows_')[1])

                        base_row_path = os.path.join(base_gpu_1_dir, f'nrows_{num_rows}')
                        base_timings_file = os.path.join(base_row_path, 'timings.txt')
                        base_training_time = read_training_time(base_timings_file)
                        
                        if base_training_time is not None:
                            speedup = base_training_time / training_time
                            data[num_gpus] = speedup
                        else:
                            print(f"Base training time not found for num_rows: {num_rows}")
                    else:
                        print(f"Training time not found in {timings_file}")

    return data

def plot_strong_scalability(data):
    plt.figure(figsize=(10, 6))
    
    num_gpus = sorted(data.keys())
    speedups = [data[gpu] for gpu in num_gpus]
    
    plt.plot(num_gpus, speedups, marker='o', linestyle='-', label='Speedup')
    
    plt.xlabel('Number of GPUs')
    plt.ylabel('Speedup')
    plt.title('Weak Scalability')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig('weak_scaling.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    base_dir = 'data_weak'
    data = collect_data(base_dir)
    plot_strong_scalability(data)