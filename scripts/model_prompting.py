import os
import json
import argparse
import torch
import argparse
import numpy as np
import random
from utils import *
import tensor_parallel as tp
from metrics import is_exact_match

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
    parser = argparse.ArgumentParser(description='Prompt LLM and save the output.')
    parser.add_argument('-m', '--model_id', type=str, required=True, help='Model ID')
    parser.add_argument('-d', '--dataset_name', type=str, required=True, help='Name of the dataset')

    args = parser.parse_args()
    model_id = config['models'][args.model_id]
    dataset_name = args.dataset_name

    print(f'model: {model_id}\ndataset: {dataset_name}')

    model, tokenizer = load_auto_model_and_tokenizer(model_id)
    if args.model_id in LARGE_MODELS:
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {n_gpus}")
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

    stop = ["\n", ".", ","]
    eos = tokenizer.decode(tokenizer.eos_token_id, skip_special_tokens=False)
    stop.append(eos)
    
    # Path to the JSONL file
    train_path = config['datasets'][dataset_name]['data']
    print("train_path: ", train_path)
    
    train_dataset = load_json_file(train_path)
    train_responses = []

    i = 0
    label_0, label_1 = 0 , 0
    for i, item in enumerate(train_dataset):
        prompt = item['question']
        response = prompt_model(model, tokenizer, prompt, stop)
        label = is_exact_match(response, prompt, item['answer'], "A")
        
        if label == 1:
             label_1 += 1
        else:
            label_0 += 1
        
        # Initialize the dictionary to append
        response_data = {
            'question': prompt,
            'answer': item['answer'],  # Assume the correct answer is always provided
            'response': response, 
            'label': label
        }

        # Append the dictionary to the list of responses
        train_responses.append(response_data)
        
        if label_0 >= 6000 and label_1 >= 6000:
            break
        
    output_path = f'datasets/{dataset_name}/{model_id.split("/")[-1]}/test_responses_{model_id.split("/")[-1]}_ends_{i}.json'
    # Save responses to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(train_responses, outfile, indent=4)

    print(f"Responses saved to {output_path}")


if __name__ == '__main__':
    main()
