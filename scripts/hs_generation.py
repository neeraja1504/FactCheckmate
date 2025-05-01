import os
import json
import argparse
import torch
import argparse
import numpy as np
import random
from utils import *
import tensor_parallel as tp
#python datasets/hs_generation.py -m llama2_7b -d nq_open 
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

f = open(os.path.dirname(__file__) + "/config.json")
config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LARGE_MODELS  = ["llama3_70b", "llama3.1_70b"]


def main():
    
    parser = argparse.ArgumentParser(description='Extract LLM hs and save the output.')
    parser.add_argument('-m', '--model_id', type=str, required=True, help='Model ID')
    parser.add_argument('-d', '--dataset_name', type=str, required=True, help='Name of the dataset')
    
    args = parser.parse_args()
    model_id = config['models'][args.model_id]
    dataset_path = config['datasets'][args.dataset_name]['balanced'][args.model_id]
    hs_path = config['datasets'][args.dataset_name]['hs_path'][args.model_id]
    dataset_name = args.dataset_name

    print(f'model: {model_id}\ndata: {dataset_name}\ndata path: {dataset_path}')

    # Load model and tokenizer
    model, tokenizer = load_auto_model_and_tokenizer(model_id)
    if args.model_id in LARGE_MODELS:
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {n_gpus}")
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

    print("reading data from: ", dataset_path)
    data = load_json_file(dataset_path)

    tensor_dict = {}
    ids = []
    cont_st_indices = []
    labels = []
    
    stop = ["\n", ".", ","]
    eos = tokenizer.decode(tokenizer.eos_token_id, skip_special_tokens=False)
    stop.append(eos)
    
    for i, row in enumerate(data):
        # if i < 7201:
        #     continue
        print(f"row {i}")
        
        ids.append(row["id"]) 
        hs, cont_st_idx = generate_hs(model, tokenizer, row['question'], row['response'], stop, device)
        cont_st_indices.append(cont_st_idx)
        labels.append(row['label'])

        for layer in range(len(hs)):
            if f"hs_{layer}" not in tensor_dict:
                tensor_dict[f"hs_{layer}"] = []
            tensor_dict[f"hs_{layer}"].append(hs[layer])

        if i > 0 and i%20 == 0:
            tensor_dict["ids"] = ids
            tensor_dict["cont_st_indices"] = cont_st_indices
            tensor_dict["labels"] = labels

            torch.save(tensor_dict, f'{hs_path}/test_ends_at_{i+1}.pth')
            del ids
            del cont_st_indices
            del labels

            ids = []
            cont_st_indices = []
            labels = []
            
            del tensor_dict
            tensor_dict = {}
            torch.cuda.empty_cache()
            print("saved")

    if ids is not None:
        tensor_dict["ids"] = ids
        tensor_dict["cont_st_indices"] = cont_st_indices
        tensor_dict["labels"] = labels
        torch.save(tensor_dict, f'{hs_path}/test_ends_at_{i+1}.pth')


if __name__ == '__main__':
    main()