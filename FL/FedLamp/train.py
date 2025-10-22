import os
import random
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from data import set_parameters, get_data_loaders, get_dataset
from worker import Worker
from server import Server
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning with FedCG')

    # Add arguments that can be passed via the command line
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='CNN', help='Specify training model')
    parser.add_argument('--global_epochs', type=int, default=300, help='Number of global epochs')
    parser.add_argument('--n_device', type=int, default=100, help='Number of clients/devices')
    parser.add_argument('--n_selected_device', type=int, default=10, help='Number of clients selected per round')
    parser.add_argument('--n_split', type=int, default=2, help='List of split values (e.g., "2,4,6")')
    parser.add_argument('--strong', type=int, default=20, help='Number of strong clients')
    # parser.add_argument('--strong', type=str, default=20, help='Number of strong clients')
    parser.add_argument('--weak', type=int, default=80, help='Number of weak clients')
    # parser.add_argument('--weak', type=str, default=80, help='Number of weak clients')
    parser.add_argument('--max_tau', type=int, default=5, help='Maximum tau')
    parser.add_argument('--min_tau', type=int, default=4, help='Minimum tau')
    parser.add_argument('--max_gamma', type=float, default=1.0, help='Maximum gamma')
    parser.add_argument('--min_gamma', type=float, default=0.6, help='Minimum gamma')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.993, help='Learning rate decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--strong_comp', type=float, default=0.436, help='Computation wall time of strong clients')
    parser.add_argument('--weak_comp', type=float, default=1.178, help='Computation wall time of weak clients')

    return parser.parse_args()


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simulate_dynamic_bandwidth(device_bandwidth_limit):
    """Simulate dynamic bandwidth within the device's bandwidth limit."""
    outbound_bw = np.random.normal(device_bandwidth_limit[0], device_bandwidth_limit[1])

    return outbound_bw * 1e6 / 8  # Convert to bytes per second


def validate(val_model, val_loader, criterion, device):
    val_model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            outputs = val_model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_val_loss, accuracy


def main():
    args = parse_args()
    cfg = {
        # general
        'seed': 2,
        'log': True,

        # data
        'dataset': args.dataset,
        'model': args.model,
        'val_ratio': 0.2,
        'n_split': args.n_split,
        'p': 0.8,
        'non_iid_type': 'PWM p partition',

        # FL
        'n_device': args.n_device,
        'n_selected_device': args.n_selected_device,
        'strong': args.strong,
        'weak': args.weak,
        'max_tau': args.max_tau,
        'min_tau': args.min_tau,
        'max_gamma': args.max_gamma,
        'min_gamma': args.min_gamma,
        'strong_comp': args.strong_comp,
        'weak_comp': args.weak_comp,

        # training
        'batch_size': args.batch_size,
        'lr': args.lr,
        'learning_rate_decay': args.lr_decay,
        'global_epochs': args.global_epochs,
        'momentum': args.momentum,  # ours: momentum=0.9,  weight_decay=5e-4
        'weight_decay': args.weight_decay,  # 0.001
    }
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # for reproducibility
    fix_seed(cfg['seed'])

    # Initialize the model and loss function
    model_fn = set_parameters(cfg)
    total_parameters = sum(p.numel() for p in model_fn(cfg['n_class']).parameters())
    bytes_per_parameter = 4
    model_size_in_bytes = total_parameters * bytes_per_parameter
    print(f"Model size: {model_size_in_bytes} bytes")
    criterion = nn.CrossEntropyLoss()

    # Define the computing parameters and bandwidth limits (mean, std_dev) for each device type
    # cnn: 2.18/5=0.436, 5.89/5=1.178
    # resnet18: 6.62/5=1.324, 17.29/5=3.458
    device_gaussian_params = {
        'S': (cfg['strong_comp'], 0.3),  # strong device
        'W': (cfg['weak_comp'], 0.5),  # weak device
    }
    device_bandwidth_limits = {
        'S': (3, 0.5),  # strong device
        'W': (30, 2),  # weak device
    }

    # Define the types of devices
    device_types = ['S', 'W']

    device_counts = {
        'S': cfg['strong'],
        'W': cfg['weak'],
    }
    devices = []
    for device_type in device_types:
        devices.extend([device_type] * device_counts[device_type])

    # Get data loaders
    train_loaders, val_loader, test_loader = get_data_loaders(cfg)

    # Set up directories for logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"./log/{cfg['dataset']}/{cfg['model']}/{cfg['strong']}_{cfg['weak']}/run_{timestamp}_noniid={cfg['n_split']}/"
    out_dir = f"./output/{cfg['dataset']}/{cfg['model']}/{cfg['strong']}_{cfg['weak']}/run_{timestamp}_noniid={cfg['n_split']}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_log.txt")
    log_f = open(log_file, 'w')

    # wandb
    wandb.init(group=log_dir, project='FedLamp')
    wandb.config.update(cfg)

    # Initialize workers
    workers = []
    for i in range(cfg['n_device']):
        device_type = devices[i]
        worker_model = torch.nn.DataParallel(model_fn(cfg['n_class']).to(device))  # A separate model for each worker

        worker = Worker(
            worker_id=i,
            device_type=device_type,
            model=worker_model,
            data_loader=train_loaders[i],
            val_loader=val_loader,
            resource_limits=(device_gaussian_params[device_type][0], device_bandwidth_limits[device_type][0]),
            loss_fn=criterion,
            computing_params=device_gaussian_params[device_type],
            bandwidth_params=device_bandwidth_limits[device_type],
            learning_rate=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            learning_rate_decay=cfg['learning_rate_decay'],
            device=device
        )
        workers.append(worker)

    # Initialize the server with the workers
    global_model = torch.nn.DataParallel(model_fn(cfg['n_class']).to(device))
    server = Server(
        global_model=global_model,
        workers=workers,
        resource_limits=(np.inf, np.inf),
        tau_e=cfg['max_tau'],
        tau_s=cfg['min_tau'],
        max_g=cfg['max_gamma'],
        min_g=cfg['min_gamma'],
        mu_s=device_gaussian_params['S'][0],
        mu_w=device_gaussian_params['W'][0],
        beta_s=device_bandwidth_limits['S'][0],
        beta_w=device_bandwidth_limits['W'][0],
    )
    stop_flag = server.stop_flag

    # h = 0
    best_val_acc = 0.0
    total_wall_time = 0
    comm_cost = 0
    start_time = time.time()
    # while not stop_flag and h <= cfg['global_epochs']:
    for h in range(cfg['global_epochs']):
        # Reset loss and accuracy for the round
        train_loss = [0.0] * cfg['n_selected_device']
        train_acc = [0.0] * cfg['n_selected_device']

        # training
        wall_times = []
        communication_cost = []
        print(f"start communication round {h}:\n")
        selected_workers = server.select_workers(cfg['n_selected_device'], cfg['weak']/cfg['n_device'])
        for worker in selected_workers:
            w_id = worker.id
            # Simulate dynamic network conditions within the device's bandwidth limits
            # outbound_bw = simulate_dynamic_bandwidth(device_bandwidth_limits[worker.device_type])
            simulated_compute_time, outbound_bw = worker.simulate_time()
            print(f"Worker {w_id} - Dynamic Outbound bandwidth: {outbound_bw * 8 / 1e6:.2f} Mb/s")

            # record real world computation and communication time
            mu_h, beta_h = 0, 0
            # Receive updates from PS
            tau_h, gamma_h = server.download(worker)

            # start training
            start_wall_time = time.time()
            train_accuracies, train_losses = worker.local_update(int(tau_h))
            wall_time = time.time() - start_wall_time

            # Accumulate loss, accuracy, and total samples for each worker
            train_loss.append(sum(train_losses)/len(train_losses))
            train_acc.append(sum(train_accuracies)/len(train_accuracies))

            # Apply model compression and update error compensation
            worker.compress_model_top_k(gamma_h)

            beta_h += gamma_h * model_size_in_bytes / outbound_bw
            mu_h += simulated_compute_time
            server.upload(worker.model.state_dict(), mu_h, beta_h, w_id)

            wall_times.append(wall_time)
            communication_cost.append(gamma_h)

            print(f"Worker {w_id} - Computation time: {mu_h}, Communication time: {beta_h}")

        avg_train_loss = sum(train_loss) / cfg['n_selected_device']
        avg_train_acc = sum(train_acc) / cfg['n_selected_device']

        max_wall_time = max(wall_times)

        total_wall_time += max_wall_time
        comm_cost += np.average(communication_cost)

        stop_flag = server.run(h)

        # validation
        val_model = server.global_model
        avg_val_loss, val_acc = validate(val_model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(out_dir, f"best_model.pt")
            torch.save(val_model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {h} with accuracy {val_acc:.4f}")

        log_output = (
            f"[ Train | {h:03d}/{cfg['global_epochs']:03d} ] "
            f"loss = {avg_train_loss:.4f}, acc = {avg_train_acc:.4f}\n"
            f"Max Wall Time: {max_wall_time:.3f}s\n" 
            f"[ Val | {h:03d}/{cfg['global_epochs']:03d} ] "
            f"loss = {avg_val_loss:.4f}, acc = {val_acc:.4f}\n"
        )
        print(log_output, file=log_f)
        log_f.flush()

        wandb.log({
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_acc,
            'validation_loss': avg_val_loss,
            'validation_accuracy': val_acc,
            'max_wall_time': max_wall_time,
        })

    end_time = time.time()
    print(f"Max Wall Time Per Round: {total_wall_time/cfg['global_epochs']:.3f} seconds", file=log_f)
    print(f"Avg Communication cost: {comm_cost:.4f}", file=log_f)
    print(f"Training ended, completed in {end_time - start_time:.2f} seconds", file=log_f)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
