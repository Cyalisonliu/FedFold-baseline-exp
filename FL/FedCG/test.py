import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from data import get_data_loaders, set_parameters  # Assuming these functions are defined in your data module

import argparse


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='Federated Learning with FedLamp')

# Add arguments that can be passed via the command line
parser.add_argument('--dataset', type=str, default='CIFAR10', help='Specify training dataset')
parser.add_argument('--model', type=str, default='CNN', help='Specify training model')
parser.add_argument('--n_split', type=int, default=2, help='noniid split values (e.g., "2,4,6")')
parser.add_argument('--strong', type=int, default=20, help='Number of strong clients')
parser.add_argument('--weak', type=int, default=80, help='Number of weak clients')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--checkpoint', type=str, default='1st')

args = parser.parse_args()

# Initialize configuration
cfg = {
    'model_checkpoint': f'./output/{args.dataset}/{args.model}/{args.strong}_{args.weak}/{args.checkpoint}_run_noniid_{args.n_split}/best_model.pth',  # Path to the saved model
    'dataset': args.dataset,
    'model': args.model,
    'batch_size': args.batch_size,
    'n_device': 100,
    'n_split': args.n_split,
    'val_ratio': 0.2,
    'selected_device': 10,  # M = 10 clients selected per round
    'non_iid_type': 'devices_per_class_partition',  # Use skewed label partitioning
    'psi': 20,  # Number of classes each client lacks
    'seed': 2,
}

# for reproducibility
fix_seed(cfg['seed'])

# Set the device to GPU
device = f'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Load the saved model weights
if not os.path.exists(cfg['model_checkpoint']):
    print(f"Model checkpoint not found at {cfg['model_checkpoint']}")
    exit(0)

model_fn = set_parameters(cfg)
model = model_fn()
model = torch.nn.DataParallel(model).to(device)
# Load the state_dict
state_dict = torch.load(cfg['model_checkpoint'], map_location=device)
model.load_state_dict(state_dict)

criterion = torch.nn.CrossEntropyLoss()

# Get data loaders
train_loaders, val_loader, test_loader, _ = get_data_loaders(cfg)

print("Starting testing...")
model.eval()
total_correct = 0
total_samples = 0
total_loss = 0.0

with torch.no_grad():  # Disable gradient calculation for testing
    for img, label in test_loader:
        img, label = img.to(device), label.to(device)
        logits = model(img)
        loss = criterion(logits, label)
        total_loss += loss.item()

        # Compute the number of correct predictions
        _, predicted = logits.max(1)
        total_correct += predicted.eq(label).sum().item()
        total_samples += label.size(0)

        # Release memory
        del img, label, logits
        torch.cuda.empty_cache()

avg_loss = total_loss / len(test_loader)
accuracy = 100.0 * total_correct / total_samples

print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")

# Save the results to a file
result_dir = f"./evaluation/{cfg['dataset']}/{cfg['model']}/{args.strong}_{args.weak}/{args.checkpoint}"
os.makedirs(result_dir, exist_ok=True)
result_file = f"{result_dir}/test_results_noniid={cfg['n_split']}.txt"
with open(result_file, 'a+') as f:
    f.write(f"Test Accuracy: {accuracy:.2f}%\n")