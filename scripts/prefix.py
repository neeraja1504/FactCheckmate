import torch
# python /projects/bcky/deema/research/FactCheckMate/prefix_expt/pre.py -m Llama-2-13b-hf -mode 0 -layer_used 14 -hs 5120
##########################################################################
import random
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from sklearn.metrics import f1_score  
import torch.optim as optim
import argparse
# import wandb

# Set a seed value
seed = 42  # You can choose any seed number

# Set the random seed for various libraries
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser(description="A simple argument parsing example")

# Add arguments
parser.add_argument('-m', '--model', type=str, help='Enter the model id', required=True)
parser.add_argument('-mode', '--mode', type=str, help='Enter the mode', required=True)
parser.add_argument('-layer_used', '--layer', type=str, help='Enter the layer number', required=True)
parser.add_argument('-hs', '--hs_size', type=int, help='Enter the hidden state size', required=True)

# Parse the arguments
args = parser.parse_args()
hs_size = args.hs_size
# hs_size = 4096
mode = int(args.mode)
# mode = 0
# projects/bcky/deema/research/fact_checkmate/datasets/nq_open/Meta-Llama-3-8B/hs/test/layer_15.pth
layer_used = args.layer
model_id = args.model
# file_path = f"/projects/bcky/deema/research/fact_checkmate/datasets/nq_open/{model_id}/hs/test/layer_{layer_used}.pth"
# best_model_path = f"/projects/bcky/deema/research/fact_checkmate/clss/nq_open/{model_id}/m_{mode}_l_{layer_used}_b_128.pth"

file_path = f'/projects/bcky/deema/research/FactCheckMate/datasets/medmcqa/{model_id}/processed/hs/cls_split/layer_{layer_used}/test_data_layer_{layer_used}.pth'
best_model_path = f'/projects/bcky/deema/research/FactCheckMate/clss/mmlu/{model_id}/cls_m_{mode}_l_{layer_used}_b_128.pth'
print("Getting results for the model",model_id)
print(file_path)
print(best_model_path)
print("Details are:",layer_used,mode)

if(torch.cuda.is_available()):
    gpu=0
    device='cuda:{}'.format(gpu)
else:
  device='cpu' 

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden to Output Layer
        self.sigmoid = nn.Sigmoid()  # Since it's a binary classification

    def forward(self, x):
        out = self.relu(self.fc(x))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out
    

def map_selected_mode(hs, mode):
    # Convert list of tensors into a single tensor
    # Each tensor is of shape [1, 4096], we concatenate along dim=0
    # tensor = torch.cat(hs, dim=0).float()  # Resulting tensor shape will be [len(hs), 1, 4096]

    # # Squeeze the middle dimension to make the shape [len(hs), 4096]
    # tensor = tensor.squeeze(1)  # Now tensor shape is [len(hs), 4096]
    tensor = hs
    if mode == 0:
        # Mean pooling across the new concatenated dimension
        mean_pooled = torch.mean(tensor, dim=0)
        assert mean_pooled.shape[-1] == hs_size
        return mean_pooled
    
    elif mode == 1:
        # Max pooling across the new concatenated dimension
        max_pooled, _ = torch.max(tensor, dim=0)
        assert max_pooled.shape[-1] == hs_size
        return max_pooled
    
    else:
        # Return the last sequence element of the last tensor if mode is neither 0 nor 1
        last = hs[-1].squeeze(0)
        assert last.shape[-1] == hs_size
        return last  # Ensure it returns a tensor of shape [4096]
    

# (tensor_id, tensor, label)
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    

def collate_fn(batch):
    data, labels = zip(*batch)
    if all(isinstance(d, torch.Tensor) for d in data):
        data_tensor = torch.stack(data)
    else:
        data_tensor = torch.tensor(data)
    labels_tensor = torch.tensor(labels)
    return data_tensor, labels_tensor



def accuracy(outputs, labels):
    rounded_preds = torch.round(outputs)
    correct = (rounded_preds == labels).float()  # Convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def f1(outputs, labels):
    predictions = torch.round(outputs.detach())
    f1_score_value = f1_score(labels.numpy(), predictions.numpy(), average='macro', zero_division=0)
    return f1_score_value

def false_positives(outputs, labels):
    rounded_preds = torch.round(outputs)
    false_positives = ((rounded_preds == 1) & (labels == 0)).float().sum()
    return false_positives.item()

def false_negatives(outputs, labels):
    rounded_preds = torch.round(outputs)
    false_negatives = ((rounded_preds == 0) & (labels == 1)).float().sum()
    return false_negatives.item()


#####################################################################################################
#####################################################################################################

def load_tensor(file_path):
    """Load a tensor from a .pth file."""
    data = torch.load(file_path)
    return data

def extract_prefixes(data):
    """Extract prefixes up to the specified index and create variations by subtracting indices."""
    hss_data = data['hs']
    conts_st_index = data['cont_st_indices']
    prefix_data = []
    prefix_data_plus_one = []
    prefix_minus_one_data = []
    prefix_minus_two_data = []
    prefix_minus_three_data = []
    prefix_minus_four_data = []
    prefix_minus_five_data = []
    prefix_minus_six_data = []
    prefix_first_data = []
    
    for hs_data, cont_st_index in zip(hss_data, conts_st_index):
        # print(len(hs_data))
        # Create new keys for prefix and variations
        prefix_data_plus_one.append(hs_data[:cont_st_index+1])
        prefix_data.append(hs_data[:cont_st_index])
        # prefix_minus_one_data.append(hs_data[:cont_st_index-7])
        # prefix_minus_two_data.append(hs_data[:cont_st_index-8])
        # prefix_minus_three_data.append(hs_data[:cont_st_index-9])
        # prefix_minus_four_data.append(hs_data[:cont_st_index-10])
        # prefix_minus_five_data.append(hs_data[:cont_st_index-11])
        # prefix_minus_six_data.append(hs_data[:cont_st_index-12])
        # prefix_first_data.append(hs_data[:1])
        prefix_minus_one_data.append(hs_data[:cont_st_index-1])
        prefix_minus_two_data.append(hs_data[:cont_st_index-2])
        prefix_minus_three_data.append(hs_data[:cont_st_index-3])
        prefix_minus_four_data.append(hs_data[:cont_st_index-4])
        prefix_minus_five_data.append(hs_data[:cont_st_index-5])
        prefix_minus_six_data.append(hs_data[:cont_st_index-6])
        prefix_first_data.append(hs_data[:1])
        
    return prefix_data_plus_one, prefix_data, prefix_minus_one_data, prefix_minus_two_data, prefix_minus_three_data, prefix_minus_four_data, prefix_minus_five_data, prefix_minus_six_data, prefix_first_data
def save_tensor(updated_data, output_path):
    """Save the updated tensor to a .pth file."""
    torch.save(updated_data, output_path)

# Path to the .pth file %%%
# file_path = '/projects/bcky/deema/research/FactCheckMate/Deema_Dont_Give_UP/datasets/nq_open/Llama-2-7b-hf/processed/hs/test/layer_14.pth'
# # file_path = '/projects/bcky/deema/research/FactCheckMate/datasets/nq_open/gemma-7b/hs/test/layer_17.pth'
# # file_path = '/projects/bcky/deema/research/FactCheckMate/datasets/nq_open/Meta-Llama-3.1-8B/hs/test/layer_15.pth'
# # file_path = '/projects/bcky/deema/research/FactCheckMate/datasets/nq_open_old/Llama-2-7b-hf/processed/hs/cls_split/layer_14/test_data_layer_14.pth'
# # file_path = '/projects/bcky/deema/research/FactCheckMate/Deema_Dont_Give_UP/clss/nq_open/Llama-2-7b-hf/cls_m_0_l_14_b_128.pth'
# # file_path = '/projects/bcky/deema/research/FactCheckMate/datasets/nq_open/Llama-2-7b-hf/hs/test/layer_14.pth'
# file_path = '/projects/bcky/deema/research/fact_checkmate/datasets/nq_open/gemma-7b/hs/test/layer_18.pth'
# file_path = '/projects/bcky/deema/research/fact_checkmate/datasets/nq_open/Llama-2-7b-hf/hs/test/layer_21.pth'
# file_path = '/projects/bcky/deema/research/fact_checkmate/datasets/nq_open/Meta-Llama-3-8B/hs/test/layer_15.pth'
# Load the tensor
tensor_data = load_tensor(file_path)

# Extract prefixes and update the tensor data
prefix_data_plus_one, prefix_data, prefix_minus_one_data, prefix_minus_two_data, prefix_minus_three_data, prefix_minus_four_data, prefix_minus_five_data, prefix_minus_six_data, prefix_first_data = extract_prefixes(tensor_data)

print("Updated tensor with prefixes successfully.")


def test_model(model, test_loader, criterion):
    model.eval()
    test_loss, test_acc, test_f1, fp, fn = 0, 0, 0, 0 , 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())

            test_loss += criterion(outputs.squeeze(-1), labels.float()).item()
            test_acc += accuracy(outputs.squeeze(-1), labels.float())
            test_f1 += f1(outputs.squeeze(-1), labels.float())
            fp += false_positives(outputs.squeeze(-1), labels.float())
            fn += false_negatives(outputs.squeeze(-1), labels.float())
            
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    test_f1 /= len(test_loader)
    fp /= len(test_loader)
    fn /= len(test_loader)

    # test_metrics = {
    #     'test_loss':test_loss,
    #     'test_acc': test_acc,
    #     'test_f1': test_f1
    # }
    # wandb.log(test_metrics)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1}, FP: {fp}, FN: {fn}')


hss =  [prefix_data_plus_one, prefix_data, prefix_minus_one_data, prefix_minus_two_data, prefix_minus_three_data, prefix_minus_four_data, prefix_minus_five_data, prefix_minus_six_data, prefix_first_data]

for hs in hss:
   
    t_data = []
    
    for i, j in zip(hs, tensor_data['labels']):
        i = torch.stack(i)
        
        if i != []:
            i =  map_selected_mode(i, mode)
            i = i.view(-1)
            t_data.append((i,j))

    test_data = t_data
    test_dataset = CustomDataset([t[0] for t in test_data], [t[1] for t in test_data])

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    # run_name = f"cls_ner_smp_m_mean_b_10_test"
    # wandb.init(project=f"classifers", entity="deema2", name=run_name, reinit=True)

    # best_model_path = "/projects/bcky/deema/research/FactCheckMate/clss/nq_open/gemma-7b/cls_m_0_l_15_b_128.pth"
    # best_model_path = "/projects/bcky/deema/research/FactCheckMate/clss/nq_open/Meta-Llama-3.1-8B/cls_m_0_l_15_b_128.pth"
    # best_model_path = "/projects/bcky/deema/research/FactCheckMate/clss/nq_open/Meta-Llama-3-8B/cls_m_0_l_17_b_128.pth"
    # best_model_path = "/projects/bcky/deema/research/FactCheckMate/clss/nq_open_pre/llama2_7b_nq_open/cls_3_shots_m_0_l_14_b_128.pth"
    # best_model_path = "/projects/bcky/deema/research/FactCheckMate/Deema_Dont_Give_UP/clss/nq_open/Llama-2-7b-hf/cls_m_0_l_14_b_128.pth"
    # best_model_path = '/projects/bcky/deema/research/FactCheckMate/clss/nq_open/Llama-2-7b-hf/cls_m_0_l_14_b_128.pth'
    # best_model_path = '/projects/bcky/deema/research/fact_checkmate/clss/nq_open/gemma-7b/m_0_l_18_b_128.pth'
    # best_model_path = '/projects/bcky/deema/research/fact_checkmate/clss/nq_open/Llama-2-7b-hf/m_0_l_21_b_128.pth'
    # best_model_path = "/projects/bcky/deema/research/fact_checkmate/clss/nq_open/Meta-Llama-3-8B/m_-1_l_15_b_128.pth"

    model = SimpleMLP(hs_size, 1024, 1)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # train_model(model, train_loader, val_loader, epochs=50, criterion=criterion, optimizer=optimizer, best_model_path = best_model_path)
    model.load_state_dict(torch.load(best_model_path,map_location=torch.device('cpu')))
    test_model(model, test_loader, criterion)