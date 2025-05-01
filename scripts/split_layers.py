import os
import torch
import argparse

def save_layer_data(layer_data, output_path):
    """Saves the specified layer data to a given output path."""
    torch.save(layer_data, output_path)

def process_files(hs_folder, num_layers):
    """
    Processes all .pth files in the hs_folder directory, extracting specified layers,
    and saves them in the layers folder within the hs_folder.
    
    Parameters:
    hs_folder (str): Directory containing the .pth files.
    num_layers (int): Number of layers to process (inclusive).
    """
    # Create the layers folder within the hs_folder
    layers_folder = os.path.join(hs_folder, 'layers')
    os.makedirs(layers_folder, exist_ok=True)

    for filename in os.listdir(hs_folder):
        if filename.endswith('.pth'):
            file_path = os.path.join(hs_folder, filename)
            print("Processing file: ", file_path)
            
            # Load the .pth file
            data = torch.load(file_path, map_location=torch.device('cpu'))
            
            # Process each layer from 0 to num_layers (inclusive)
            for i in range(num_layers + 1):
                layer_key = f'hs_{i}'
                if layer_key in data:
                    layer_data = {
                        'hs': data[layer_key],
                        'ids': data.get('ids'),
                        'labels': data.get('labels'),
                        'cont_st_indices': data.get('cont_st_indices')
                    }
                    layer_dir = os.path.join(layers_folder, f'layer_{i}')
                    os.makedirs(layer_dir, exist_ok=True)
                    
                    # Save the file without timestamp
                    output_path = os.path.join(layer_dir, f'{filename}_layer_{i}.pth')
                    save_layer_data(layer_data, output_path)
                    print(f"Layer {layer_key} saved to {output_path}.")
            
            # Clean up
            del data
            print(f"File {filename} processed.")
    
    print("Data processing completed.")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process .pth files to extract specified layers.")
    parser.add_argument('hs_folder', type=str, help="Path to the hs folder containing .pth files.")
    parser.add_argument('num_layers', type=int, help="Number of layers to process (inclusive).")
    
    args = parser.parse_args()
    
    # Process the files with the given arguments
    process_files(args.hs_folder, args.num_layers)
