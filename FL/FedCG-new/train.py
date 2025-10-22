import os
import random
import numpy as np
import time
import argparse
import math
import random

import torch
from torch.utils.data import DataLoader
# import wandb

from data import SplitDataset, get_dataset, split_dataset, set_parameters
from model import Conv, LocalMaskCrossEntropyLoss, MLP, ResNet18, ResNet152
from fed import Federation
from utils import Compressor


def fixSeed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_device_capabilities(_random=False, n_device=100, n_strong=20, n_weak=80, strong_comp=1.01, weak_comp=5.89, min_bandwidth_mbps=3, max_bandwidth_mbps=30):
    assert n_strong + n_weak == n_device, "The total number of devices must equal n_strong + n_weak."
    min_bandwidth_bytes = min_bandwidth_mbps * 1_000_000 / 8  # Convert to bytes per second
    max_bandwidth_bytes = max_bandwidth_mbps * 1_000_000 / 8  # Convert to bytes per second

    # Simulate computational heterogeneity
    strong_computation_times = n_strong * [strong_comp]
    if _random:
        strong_bandwidth = n_strong * [np.random.uniform(min_bandwidth_bytes, max_bandwidth_bytes)]
    else:
        strong_bandwidth = n_strong * [min_bandwidth_mbps * 1_000_000 / 8]

    weak_computation_times = n_weak * [weak_comp]
    if _random:
        weak_bandwidth = n_weak * [np.random.uniform(min_bandwidth_bytes, max_bandwidth_bytes)]
    else:
        weak_bandwidth = n_weak * [max_bandwidth_mbps * 1_000_000 / 8]

    # Combine strong and weak device computation times
    computation_times = np.concatenate([strong_computation_times, weak_computation_times])
    bandwidths = np.concatenate([strong_bandwidth, weak_bandwidth])

    capabilities = []
    for i in range(n_device):
        capabilities.append({
            'computation_time': computation_times[i],  # Computation time for each device
            'outbound_bandwidth': bandwidths[i],  # Bandwidth in bytes per second
        })

    return capabilities


def update_device_bandwidth(capabilities, min_bandwidth_mbps=3, max_bandwidth_mbps=30):
    min_bandwidth_bytes = min_bandwidth_mbps * 1_000_000 / 8  # Convert to bytes per second
    max_bandwidth_bytes = max_bandwidth_mbps * 1_000_000 / 8  # Convert to bytes per second

    for client in capabilities:
        client['outbound_bandwidth'] = np.random.uniform(min_bandwidth_bytes, max_bandwidth_bytes)

cfg = {
    # general
    'seed': 2,
    'log': True,

    # data
    'dataset': 'CIFAR100',  # 'TinyImageNet', 'Otto', 'SVHN'
    'n_split': 2,
    'val_ratio': 0.2,

    # FL
    'n_device': 100,
    'selected_device': 10,
    'device_ratio': 'S2-W8',

    # training
    'batch_size': 32,
    'lr': 1e-2,
    'global_epochs': 10,  # change from 300 to 100
    'local_epochs': 5,

    # device dropout
    'dropout_rate': 0.5,
}

parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true', help='enable wandb log')
parser.add_argument('--model_structure', type=str, default='S10', help='model_structure')
parser.add_argument('--device_ratio', type=str, default='S2-W8', help='device ratio')
parser.add_argument('--train_ratio', type=str, default='16-1', help='trainable width ratio')
parser.add_argument('--n_split', type=int, default=2, help='number of splits for non-iid data')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
parser.add_argument('--model', type=str, default='CNN', help='dataset')
parser.add_argument('--strong', type=int, default=20, help='Number of strong clients')
parser.add_argument('--weak', type=int, default=80, help='Number of weak clients')
parser.add_argument('--random', action='store_true')
parser.add_argument('--no-random', dest='random', action='store_false')
parser.set_defaults(random=False)
args = parser.parse_args()

cfg['log'] = args.log
cfg['model_structure'] = args.model_structure
cfg['device_ratio'] = args.device_ratio
cfg['strong'] = args.strong
cfg['weak'] = args.weak
cfg['train_ratio'] = args.train_ratio
cfg['n_split'] = args.n_split
cfg['TBW'] = int(args.train_ratio.split('-')[0])
cfg['medium_BW'] = int(args.train_ratio.split('-')[1])  # for three types of devices
cfg['dataset'] = args.dataset
cfg['model'] = args.model
cfg['random'] = args.random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())


# get dataset
dataset, labels = get_dataset(cfg['dataset'], cfg['val_ratio'])

# get model and hidden size for each width of model
model_fn, hidden_size = set_parameters(cfg)
if model_fn == Conv:
    cfg['strong_comp'] = 1.01
    cfg['weak_comp'] = 5.89
    cfg['time_budget'] = 5.99
elif model_fn == ResNet18:
    cfg['strong_comp'] = 1.54
    cfg['weak_comp'] = 12.62
    cfg['time_budget'] = 16.2
else:
    cfg['strong_comp'] = 3.175
    cfg['weak_comp'] = 64.75
    cfg['time_budget'] = 83.28
print(cfg['time_budget'])

# split train dataset for Non-IID
fixSeed(cfg['seed'])
data_split = {}
data_split['train'], label_split = split_dataset(labels['train'], cfg['n_device'], cfg['n_class'], cfg['n_split'])

# Open a log file for writing the classes of each device
class_log_file = open("device_classes_log.txt", "w")
for i in range(cfg['n_device']):
    class_log_file.write(f"Classes {label_split[i]}\n")
class_log_file.close()

# make dataloader
train_loader = []
data_sizes = []
for i in range(cfg['n_device']):
    train_set = SplitDataset(dataset['train'], data_split['train'][i])
    train_loader.append(
        DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=1, pin_memory=True))
    data_sizes.append(len(train_set))
val_loader = DataLoader(dataset['val'], batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

# create model
BW_cnt = 0
TBW = cfg['TBW']
model_list = []
model_log = []

model_list.append(model_fn(hidden_size[str(TBW)], n_class=cfg['n_class']).to(device))
model_log.append(TBW)
total_bytes = 0
for param in model_list[0].parameters():
    total_bytes += param.data.nelement() * param.data.element_size()
print(f"total bytes of model[{0}]: {total_bytes}")

# calculate the number of BW
for i in model_log:
    if i == 1:
        BW_cnt += 1

model_list = model_list[::-1]
model_log = model_log[::-1]

print(f'BW_cnt: {BW_cnt}')
print(f'model_list: {model_log}')
n_model = len(model_list)

# for federation learning
fed = Federation([model_list[i].state_dict() for i in range(n_model)], data_sizes)

# create loss function for FL training
loss_fn = LocalMaskCrossEntropyLoss(cfg['n_class'])

# setting device type according to device ratio
device_type = []
device_cnt = []
select = []
for ratio in cfg['device_ratio'].split('-'):
    device_type += [ratio[0]] * int(ratio[1:]) * cfg['selected_device']
    device_cnt.append(int(ratio[1:]) * cfg['selected_device'])
    select.append(int(ratio[1:]))
print(f'device_type: {device_type}')
print(f'device_cnt: {device_cnt}')

client_capabilities = initialize_device_capabilities(cfg['random'], cfg['n_device'], cfg['strong'], cfg['weak'], cfg['strong_comp'], cfg['weak_comp'])
# create model tag for saving model
time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
model_tag = f"{cfg['dataset']}_non-iid-{cfg['n_split']}_train-ratio-{cfg['train_ratio']}"
model_name = 'Conv'
model_tag += '_FedCG'
if model_fn == ResNet18:
    model_tag += '_ResNet18'
    model_name = 'ResNet18'
if model_fn == ResNet152:
    model_tag += '_ResNet152'
    model_name = 'ResNet152'
print(f'model tag: {model_tag}')

if cfg['random']:
    output_dir = f"./output/random/{model_tag}/{cfg['device_ratio']}_{time_stamp}"
else:
    output_dir = f"./output/{model_tag}/{cfg['device_ratio']}_{time_stamp}"
os.makedirs(output_dir, exist_ok=True)

if cfg['random']:
    # wandb.init(group=output_dir, project='fedcg-rebuild-random')
    os.makedirs('./log_random', exist_ok=True)
    log_f = open(f"./log_random/{model_tag}_{cfg['device_ratio']}-{time_stamp}.txt", 'w')
else:
    # wandb.init(group=output_dir, project='fedcg-rebuild')
    os.makedirs('./log', exist_ok=True)
    log_f = open(f"./log/{model_tag}_{cfg['device_ratio']}-{time_stamp}.txt", 'w')

# for fixed selection
# S2-W8: 20, 80 => 0, 20, 100
device_cnt.insert(0, 0)
for i in range(len(select)):
    device_cnt[i + 1] += device_cnt[i]

# start training
BW_idx = 0
best_acc = [0.0] * n_model
model_cnt = [0] * n_model

# calculate total training time
total_comp_time = 0.0
total_cpu_comp_time = 0.0
start = time.time()
total_communication_cost = 0
original_communication = 0

for epoch in range(1, cfg['global_epochs'] + 1):
    if cfg['random']:
        update_device_bandwidth(client_capabilities)
    device_idx, compression_rates = fed.joint_optimization_sub(client_capabilities, cfg['time_budget'])
    selected_device_types = [device_type[idx] for idx in device_idx]
    selected_device_bw = [client_capabilities[idx] for idx in device_idx]

    print(f"Epoch {epoch}: {cfg['device_ratio']}, {device_idx}"
          f", device types: {selected_device_types}"
          f", device bw: {selected_device_bw}", file=log_f)

    # training
    train_loss = []
    train_acc = []
    model_0_train_acc = []
    model_0_train_loss = []
    model_1_train_acc = []
    model_2_train_acc = []
    model_3_train_acc = []
    model_4_train_acc = []

    MAX_WALL_TIME = 0.0
    MAX_CPU_TIME = 0.0
    for i, d_id in enumerate(device_idx):
        # decide the model idx to train
        model_idx = [0]
        optimizers = {}
        for idx in model_idx:
            fed.download(model_list[idx].state_dict(), idx)
            optimizers[idx] = torch.optim.SGD(model_list[idx].parameters(), lr=cfg['lr'], momentum=0.9,
                                              weight_decay=5e-4)
        WALL_TIME = 0.0
        cuda_time = 0.0
        CPU_TIME = 0.0
        for local_epoch in range(cfg['local_epochs']):
            for img, label in train_loader[d_id]:
                img, label = img.to(device), label.to(device)

                for idx in model_idx:
                    model = model_list[idx]
                    model.train()

                    TIME_START = time.time()
                    cpu_start = time.process_time()
                    cuda_start = torch.cuda.Event(enable_timing=True)
                    cuda_end = torch.cuda.Event(enable_timing=True)
                    cuda_start.record()

                    CPU_TIME_START = time.process_time()
                    # forward data
                    logits = model(img)

                    # calcuate loss
                    loss = loss_fn(logits, label)

                    # clear gradient computed at previous step
                    optimizers[idx].zero_grad()

                    # calculate gradient
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    # update parameters with computed gradient
                    optimizers[idx].step()
                    WALL_TIME += time.time() - TIME_START
                    cuda_end.record()
                    torch.cuda.synchronize()
                    cuda_time += cuda_start.elapsed_time(cuda_end)
                    CPU_TIME += time.process_time() - CPU_TIME_START

                    # log training loss and accuracy
                    acc = (logits.argmax(dim=1) == label).float().mean()
                    train_loss.append(loss.item())
                    train_acc.append(acc)
                    if idx == 0:
                        model_0_train_acc.append(acc.item())
                        model_0_train_loss.append(loss.item())
                    elif idx == 1:
                        model_1_train_acc.append(acc.item())
                    elif idx == 2:
                        model_2_train_acc.append(acc.item())
                    elif idx == 3:
                        model_3_train_acc.append(acc.item())
                    elif idx == 4:
                        model_4_train_acc.append(acc.item())

        # upload local model parameter to server
        total_communication_cost += compression_rates[i] * total_bytes / client_capabilities[d_id]['outbound_bandwidth']
        original_communication += total_bytes / client_capabilities[d_id]['outbound_bandwidth']
        k_ratio = compression_rates[i]
        for idx in model_idx:
            # apply top-K compression
            Compressor.apply_top_k(model_list[idx], k_ratio)

            fed.upload(model_list[idx].state_dict(), idx)

        print(
            f'Device {d_id} train {model_idx}, Wall time: {WALL_TIME:.3f}s, CPU time: {CPU_TIME:.3f}s, cuda time: {cuda_time:.3f}s')
        MAX_WALL_TIME = max(MAX_WALL_TIME, WALL_TIME)
        MAX_CPU_TIME = max(MAX_CPU_TIME, CPU_TIME)
        total_comp_time += MAX_WALL_TIME
        total_cpu_comp_time += MAX_CPU_TIME

    # after local training, server aggregates model parameter
    step = epoch
    fed.aggregate()
    if model_0_train_acc:
        avg_train_acc = sum(model_0_train_acc) / len(model_0_train_acc)
        avg_train_loss = sum(model_0_train_loss) / len(model_0_train_loss)
        # wandb.log({'Model 0 Train Accuracy': avg_train_acc, 'Model 0 Train Loss': avg_train_loss, 'epoch': epoch}, step=step)
    if model_1_train_acc:
        avg_train_acc = sum(model_1_train_acc) / len(model_1_train_acc)
        # wandb.log({'Model 1 Train Accuracy': avg_train_acc}, step=step)
    if model_2_train_acc:
        avg_train_acc = sum(model_2_train_acc) / len(model_2_train_acc)
        # wandb.log({'Model 2 Train Accuracy': avg_train_acc}, step=step)
    if model_3_train_acc:
        avg_train_acc = sum(model_3_train_acc) / len(model_3_train_acc)
        # wandb.log({'Model 3 Train Accuracy': avg_train_acc}, step=step)
    if model_4_train_acc:
        avg_train_acc = sum(model_4_train_acc) / len(model_4_train_acc)
        # wandb.log({'Model 4 Train Accuracy': avg_train_acc}, step=step)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)
    print(f"[ Train | {epoch :03d}/{cfg['global_epochs']:03d} ] loss = {train_loss:.4f}, acc = {train_acc:.4f}")
    print(f"Max wall time: {MAX_WALL_TIME:.3f}s")
    print(f"Max CPU time: {MAX_CPU_TIME:.3f}s")

    # validation
    for idx in range(n_model):
        fed.download(model_list[idx].state_dict(), idx)

    val_loss = []
    val_acc = [0.0] * n_model
    prediction = [None] * n_model  # for each split model
    prediction_ens = None  # for ensemble
    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            pred_ens = None
            for idx in range(n_model):
                model = model_list[idx]
                model_list[idx].eval()

                logits = model(img)
                logits = torch.nn.functional.softmax(logits, dim=1)

                loss = loss_fn(logits, label)
                val_loss.append(loss.item())

                if prediction[idx] == None:
                    prediction[idx] = logits
                else:
                    prediction[idx] = torch.cat([prediction[idx], logits], dim=0)

                # ensemble for plotting comvergence time
                if pred_ens == None:
                    pred_ens = logits
                else:
                    pred_ens += logits
            if prediction_ens == None:
                prediction_ens = pred_ens
            else:
                prediction_ens = torch.cat([prediction_ens, pred_ens], dim=0)

    # save each model according to their individual validation accuracy
    for idx in range(n_model):
        val_acc[idx] = (prediction[idx].argmax(dim=1) == torch.tensor(labels['val']).to(device)).float().mean()
        if val_acc[idx] > best_acc[idx]:
            best_acc[idx] = val_acc[idx]
            torch.save(model_list[idx].state_dict(), os.path.join(output_dir, f'{idx}_{model_log[idx]}.pth'))

    val_loss = sum(val_loss) / len(val_loss)
    val_acc = (prediction_ens.argmax(dim=1) == torch.tensor(labels['val']).to(device)).float().mean()
    print(f"[ Val | {epoch :03d}/{cfg['global_epochs']:03d} ] loss = {val_loss:.4f}, acc = {val_acc:.4f}")

    # if cfg['log']:
    # wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
    # for ploting convergence time
    print(f"{val_acc}", file=log_f)
    log_f.flush()
end = time.time()
cpu_end = time.process_time()
torch.cuda.empty_cache()

print(f"Cuda time: {cuda_start.elapsed_time(cuda_end)}ms")
print(f"Parallel Wall time: {total_comp_time:.3f}ds", file=log_f)
print(f"Parallel CPU time: {total_cpu_comp_time:.3f}ds")
print(f"Total time: {(end - start):.2f}s", file=log_f)
print(f"Total CPU time: {(cpu_end - cpu_start):.2f}s")
print(f"Total communication time: {total_communication_cost:.2f}", file=log_f)
print(f"Total communication saving: {(1-total_communication_cost/original_communication):.2f}", file=log_f)
