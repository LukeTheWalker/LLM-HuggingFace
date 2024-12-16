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
    for gpu_dir in os.listdir(base_dir):
        if gpu_dir.startswith('gpu_'):
            n = int(gpu_dir.split('_')[1])
            for batch_dir in os.listdir(os.path.join(base_dir, gpu_dir)):
                if batch_dir.startswith('batch_'):
                    b = int(batch_dir.split('_')[1]) // n
                    timings_file = os.path.join(base_dir, gpu_dir, batch_dir, 'timings.txt')
                    training_time = read_training_time(timings_file)
                    if training_time is not None:
                        if b not in data:
                            data[b] = []
                        data[b].append((n, training_time))
    return data

def plot_strong_scalability(data):
    plt.figure(figsize=(10, 6))
    for batch_size in sorted(data.keys()):
        values = data[batch_size]
        values.sort()
        n_values, times = zip(*values)
        plt.plot(n_values, times, marker='o', label=f'Batch size {batch_size}')
    
    plt.xlabel('Number of GPUs')
    plt.ylabel('Training Time (s)')
    plt.yscale('log')
    plt.title('Strong Scalability')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig('strong_scaling.png', bbox_inches='tight')

if __name__ == "__main__":
    base_dir = 'data'
    data = collect_data(base_dir)
    plot_strong_scalability(data)