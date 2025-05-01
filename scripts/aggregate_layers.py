import os
import torch
import random
import argparse
import numpy as np

# Set random seeds for reproducibility
random.seed(42)          # Python's built-in random module
np.random.seed(42)       # Numpy module
torch.manual_seed(42)    # PyTorch

# Set up the argument parser
parser = argparse.ArgumentParser(description='Load PyTorch data from a specified file path.')
parser.add_argument('hs_folder', type=str, help='The file path to the base folder containing layers and where aggregated data will be saved.')
parser.add_argument('num_layers', type=int, help='The number of layers.')

# Parse the arguments
args = parser.parse_args()

# Example usage:
print("Start processing")
base_folder = args.hs_folder
layers_folder = os.path.join(base_folder, "layers")
num_layers = args.num_layers

# Ensure the 'aggregated' directory exists
aggregated_folder = os.path.join(base_folder, "aggregated")
os.makedirs(aggregated_folder, exist_ok=True)

for i in range(num_layers + 1):  # Loop through each layer
    folder_name = f"layer_{i}"
    print(f"Processing {folder_name}...")
    folder_path = os.path.join(layers_folder, folder_name)
    output_file = os.path.join(aggregated_folder, f'layer_{i}.pth')

    aggregated_data = {
        'hs': [],
        'ids': [],
        'labels': [],
        'cont_st_indices': []
    }
    
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            print("Processing file: ", file_name)
            if file_name.endswith('.pth'):
                file_path = os.path.join(folder_path, file_name)
                data = torch.load(file_path, map_location=torch.device('cpu'))
                
                if isinstance(data, dict):
                    for key in aggregated_data.keys():
                        if key in data and isinstance(data[key], list):
                            aggregated_data[key].extend(data[key])
                        else:
                            print(f"Key {key} not found or not a list in file {file_path}")
                else:
                    print(f"File {file_path} does not contain a dictionary.")

    # Save the aggregated data
    torch.save(aggregated_data, output_file)
    del aggregated_data
    
    print(f"Aggregated data saved to {output_file}")

print("Data processing completed.")
