import os
import random
import time
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import shutil

import wandb

from fedcg import FedCG
from data import set_parameters, get_data_loaders
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning with FedCG')

    # Add arguments that can be passed via the command line
    parser.add_argument('--model', type=str, default='ResNet18', help='Specify training model')
    parser.add_argument('--n_split', type=int, default=2, help='Number of data splits per device')
    # parser.add_argument('--n_split', type=str, default='2,4,6', help='List of split values (e.g., "2,4,6")')
    parser.add_argument('--global_epochs', type=int, default=300, help='Number of global epochs')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs')
    parser.add_argument('--selected_device', type=int, default=10, help='Number of clients selected per round')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--time_budget', type=float, default=30, help='Time budget in seconds')
    parser.add_argument('--n_device', type=int, default=100, help='Number of clients/devices')
    parser.add_argument('--strong', type=int, default=20, help='Number of strong clients')
    parser.add_argument('--weak', type=int, default=80, help='Number of weak clients')
    parser.add_argument('--strong_comp', type=float, default=2.18, help='Computation wall time of strong clients')
    parser.add_argument('--weak_comp', type=float, default=5.89, help='Computation wall time of weak clients')
    # parser.add_argument('--log', action='store_true', help='Enable logging with wandb')

    return parser.parse_args()


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_device_capabilities(n_device=100, n_strong=20, n_weak=80, strong_comp=2.18, weak_comp=5.89):
    assert n_strong + n_weak == n_device, "The total number of devices must equal n_strong + n_weak."

    # Simulate computational heterogeneity
    # Strong devices: resnet18: 6.62, cnn: 2.18
    strong_computation_times = n_strong * [strong_comp]
    strong_bandwidth = n_strong * [3 * 1_000_000 / 8]

    # Weak devices: resnet18: 17.29, 5.89
    weak_computation_times = n_weak * [weak_comp]
    weak_bandwidth = n_weak * [30 * 1_000_000 / 8]

    # Combine strong and weak device computation times
    computation_times = np.concatenate([strong_computation_times, weak_computation_times])
    bandwidths = np.concatenate([strong_bandwidth, weak_bandwidth])

    client_capabilities = []
    for i in range(n_device):
        client_capabilities.append({
            'computation_time': computation_times[i],  # Computation time for each device
            'outbound_bandwidth': bandwidths[i],  # Bandwidth in bytes per second
        })

    return client_capabilities


def update_device_bandwidth(client_capabilities):
    """
    Randomly update each client's bandwidth (inbound and outbound) before each round.
    """
    for client in client_capabilities:
        outbound_bandwidth = np.random.uniform(1, 5) * 1_000_000 / 8  # In bytes per second

        # Recalculate communication times
        client['outbound_bandwidth'] = outbound_bandwidth


def validate(model, val_loader, criterion, device='cpu'):
    """
    Validate all models on the validation set.
    """
    model.to(device)  # Move model to the correct device
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            logits = model(img)
            loss = criterion(logits, label)
            val_loss += loss.item()
            _, predicted = logits.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            del logits, loss, img, label
            torch.cuda.empty_cache()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_selected_clients(model, train_loader, criterion, compression_rate, local_epochs, cfg, device='cpu'):

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    total_loss = 0.0
    correct = 0
    total = 0

    train_start = time.time()
    for local_epoch in range(local_epochs):
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            model.train()
            # Forward pass
            logits = model(img)
            # Calculate loss
            loss = criterion(logits, label)
            # Clear the gradients computed at the previous step
            optimizer.zero_grad()
            # Backward pass (calculate gradients)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Update parameters with computed gradient
            optimizer.step()

            # Accumulate loss, acc
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(label).sum().item()
            total += label.size(0)

    wall_time = time.time() - train_start

    # Calculate the average loss and accuracy
    avg_loss = total_loss / len(train_loader.dataset)  # total loss divided by the total number of examples
    accuracy = 100. * correct / total  # accuracy as percentage
    # scheduler.step(avg_loss)

    # Apply Top-K sparsification to gradients after all local epochs (communication step)
    k_ratio = compression_rate
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Skip None or empty parameters
            if param is not None and param.numel() > 0:
                # Flatten the parameter tensor to a 1D vector for easier manipulation
                param_flat = param.view(-1)

                # Calculate the number of elements to retain based on the k_ratio
                k = int(len(param_flat) * k_ratio)
                if k == 0:
                    continue

                # Get the indices of the top-K elements based on absolute values
                _, topk_indices = torch.topk(param_flat.abs(), k, sorted=False)

                # Retain the top-K values
                compressed_param = torch.zeros_like(param_flat)
                compressed_param[topk_indices] = param_flat[topk_indices]
                # Update the parameter with the compressed values, and reshape to the original shape
                param.copy_(compressed_param.view_as(param))

    torch.cuda.empty_cache()

    return avg_loss, accuracy, wall_time


def main():
    args = parse_args()

    cfg = {
        'log': True,
        'dataset': 'CIFAR10',
        'model': args.model,
        'batch_size': args.batch_size,
        'n_device': args.n_device,
        'n_split': args.n_split,
        'val_ratio': 0.2,
        'lr': args.lr,
        'global_epochs': args.global_epochs,
        'local_epochs': args.local_epochs,
        'selected_device': args.selected_device,
        'time_budget': args.time_budget,
        'strong': args.strong,
        'weak': args.weak,
        'strong_comp': args.strong_comp,
        'weak_comp': args.weak_comp,
        'non_iid_type': 'devices_per_class_partition',
        'psi': 20,
        'seed': 2,
    }

    # for reproducibility
    fix_seed(cfg['seed'])

    # Set the device to GPU
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(device)

    # Set model and parameters
    model_fn = set_parameters(cfg)
    criterion = nn.CrossEntropyLoss()

    # Initialize client capabilities
    client_capabilities = initialize_device_capabilities(cfg['n_device'], cfg['strong'], cfg['weak'], cfg['strong_comp'],
                                                         cfg['weak_comp'])
    # Set up directories for logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"./log/{cfg['dataset']}/{cfg['model']}/{cfg['strong']}_{cfg['weak']}/run_{timestamp}_{cfg['n_split']}/"
    out_dir = f"./output/{cfg['dataset']}/{cfg['model']}/{cfg['strong']}_{cfg['weak']}/_run_{timestamp}_{cfg['n_split']}/"
    # backup_dir = f"./script_backups/{cfg['dataset']}/{cfg['model']}_run_{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    # os.makedirs(backup_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_log.txt")
    # script_name = os.path.basename(__file__)  # Get the name of the current script
    # backup_file = os.path.join(backup_dir, f"{script_name}_{timestamp}.py")
    # shutil.copy(__file__, backup_file)

    wandb.init(group=log_dir, project='FedCG')
    wandb.config.update(cfg)

    log_f = open(log_file, 'w')

    # Wrap each model with DataParallel for multiple GPUs
    models = [torch.nn.DataParallel(model_fn()).to(device) for _ in range(cfg['n_device'])]
    global_model = torch.nn.DataParallel(model_fn()).to(device)

    # Get data loaders
    train_loaders, val_loader, test_loader, data_sizes = get_data_loaders(cfg)

    fedcg_server = FedCG(data_sizes, global_model, cfg['selected_device'], cfg['n_device'])

    # To record the time taken
    total_comp_time = 0
    total_comm_cost = 0
    best_accuracy = 0
    best_model_path = os.path.join(out_dir, "best_model.pth")

    # Training loop
    start = time.time()
    for epoch in range(cfg['global_epochs']):

        # Step 1: Joint optimization
        # selection_start_time = time.time()
        # Joint optimization to select clients and determine compression rates
        selected_clients, compression_rates = fedcg_server.joint_optimization_sub(client_capabilities, cfg['time_budget'])
        total_comm_cost += np.average(compression_rates)

        # Step 2: Train selected clients
        wall_times = []
        total_train_loss = 0.0
        total_train_accuracy = 0.0
        # Train selected clients
        for i, client_id in enumerate(selected_clients):
            print(f"Device {client_id} is training...")
            # download
            fedcg_server.download(models[client_id])

            client_train_loss, client_train_accuracy, wall_time = train_selected_clients(
                models[client_id], train_loaders[client_id], criterion,
                compression_rates[i], cfg['local_epochs'], cfg, device)
            wall_times.append(wall_time)

            # Accumulate training loss and accuracy
            total_train_loss += client_train_loss
            total_train_accuracy += client_train_accuracy

            # upload update
            fedcg_server.upload(models[client_id].state_dict(), client_id)

        # Calculate average training loss and accuracy for this round
        avg_train_loss = total_train_loss / cfg['selected_device']
        avg_train_accuracy = total_train_accuracy / cfg['selected_device']

        # Aggregate updates from selected clients
        fedcg_server.aggregate()

        # Validation step: Validate the global model after aggregation
        avg_loss, accuracy = validate(fedcg_server.global_model, val_loader, criterion, device)

        max_wall_time = max(wall_times)
        total_comp_time += max_wall_time

        # logging
        log_output = (
            f"[ Train | {epoch:03d}/{cfg['global_epochs']:03d} ] "
            f"Loss: {avg_train_loss:.4f} | Accuracy: {avg_train_accuracy:.4f}\n"
            f"Max Wall Time: {max_wall_time:.3f}s | \n"
            f"[ Val   | {epoch:03d}/{cfg['global_epochs']:03d} ] "
            f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}\n"
        )
        print(log_output, file=log_f)
        log_f.flush()

        # if log:
        wandb.log({
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'validation_loss': avg_loss,
            'validation_accuracy': accuracy,
            'max_wall_time': max_wall_time,
        })

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(fedcg_server.global_model.state_dict(), best_model_path)

    total_training_time = time.time() - start
    print(f"WALL TIME per round: {total_comp_time / cfg['global_epochs']:.2f} seconds.", file=log_f)
    print(f"Communication cost per round: {total_comm_cost / cfg['global_epochs']:.2f} seconds.", file=log_f)
    print(f"Total training time: {total_training_time:.2f} seconds.", file=log_f)
    log_f.flush()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
