import os
import random
import numpy as np
import time
import argparse
import math
import random
import json

import torch
from torch.utils.data import DataLoader
import wandb

from data import SplitDataset, get_dataset, split_dataset, set_parameters
from model import Conv, LocalMaskCrossEntropyLoss, MLP, ResNet
from fed import Federation
from utils import Compressor, Utils

def fixSeed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
cfg = {
    # general
    'seed': 2,
    'log': True,

    # data
    'dataset': 'CIFAR10', #'Otto', 'SVHN'
    'n_split': 20,
    'val_ratio': 0.2,

    # FL
    'n_device': 100,
    'selected_device': 10,
    'device_ratio': 'S2-W8',

    # training
    'batch_size': 32,
    'lr': 1e-2,
    'global_epochs': 10, #change from 300 to 100
    'local_epochs': 5,  

    #device dropout
    'dropout_rate' : 0.5,
}


parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true', help='enable wandb log')
parser.add_argument('--device_ratio', type=str, default='S2-W8', help='device ratio')
parser.add_argument('--train_ratio', type=str, default='16-1', help='trainable width ratio')
parser.add_argument('--fix_split', type=int, default=-1, help='fixed split ratio')
parser.add_argument('--only_strong', action='store_true', help='train only strong devices')
parser.add_argument('--only_weak', action='store_true', help='train only weak devices')
parser.add_argument('--n_split', type=int, default=2, help='number of splits for non-iid data')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
parser.add_argument('--quantize', action='store_true', help='Apply quantization to uploaded model weights.')
parser.add_argument('--quant_bits', type=int, default=-1, 
                    help='Fixed number of bits for quantization (e.g., 4, 8, 16). If -1, a random rate from [1, 2, 4, 8, 16, 32] is selected.')
parser.add_argument('--n_device', type=int, default=100, help='total number of devices (clients) to simulate')
parser.add_argument('--selected_device', type=int, default=cfg['selected_device'], help='(fixed) number of devices to select each round')
parser.add_argument('--selection_mode', type=str, choices=['fixed','proportional'], default='fixed', help='selection scaling: fixed keeps selected_device constant; proportional scales selected_device by participation_rate')
parser.add_argument('--participation_rate', type=float, default=0.1, help='(used when selection_mode=proportional) fraction of population to select each round')
parser.add_argument('--sim_mode', type=str, choices=['none','replicate','synthetic'], default='none', help='simulation mode for large n_device')
parser.add_argument('--replicate_splits', type=int, default=100, help='number of base splits to create when sim_mode=replicate')
parser.add_argument('--global_epochs', type=int, default=cfg['global_epochs'], help='number of global epochs to run')
parser.add_argument('--local_epochs', type=int, default=cfg['local_epochs'], help='number of local epochs per selected client')
args = parser.parse_args()

cfg['log'] = args.log
cfg['device_ratio'] = args.device_ratio
cfg['only_strong'] = args.only_strong
cfg['only_weak'] = args.only_weak
cfg['train_ratio'] = args.train_ratio
cfg['fix_split'] = args.fix_split
cfg['n_split'] = args.n_split
cfg['TBW'] = int(args.train_ratio.split('-')[0])
cfg['medium_BW'] = int(args.train_ratio.split('-')[1]) # for three types of devices
cfg['dataset'] = args.dataset
cfg['quantize'] = args.quantize
cfg['quant_bits'] = args.quant_bits
cfg['n_device'] = args.n_device
cfg['selected_device'] = args.selected_device
cfg['selection_mode'] = args.selection_mode
cfg['participation_rate'] = args.participation_rate
cfg['sim_mode'] = args.sim_mode
cfg['replicate_splits'] = args.replicate_splits
cfg['global_epochs'] = args.global_epochs
cfg['local_epochs'] = args.local_epochs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(torch.cuda.is_available())

# get dataset
dataset, labels = get_dataset(cfg['dataset'], cfg['val_ratio'])

# get model and hidden size for each width of model
model_fn, hidden_size, cfg['n_class']= set_parameters(cfg)

# split train dataset for Non-IID
fixSeed(cfg['seed'])
data_split = {}
# If replicate mode, create K base splits and keep them; mapping to virtual clients will be
# created per-epoch (randomized) so each epoch virtual clients can be assigned different base splits.
base_split_dict = None
base_label_split = None
if cfg['sim_mode'] == 'replicate':
    K = max(1, int(cfg['replicate_splits']))
    print(f"Sim mode=replicate: creating {K} base splits (will map to {cfg['n_device']} virtual clients per-epoch)")
    base_split_dict, base_label_split = split_dataset(labels['train'], K, cfg['n_class'], cfg['n_split'])
    # base_split_dict: dict {0: [idxs], ...}
    # We'll generate a randomized mapping (length n_device) per epoch later.
else:
    data_split['train'], label_split = split_dataset(labels['train'], cfg['n_device'], cfg['n_class'], cfg['n_split'])

# Open a log file for writing the classes of each device
class_log_file = open("device_classes_log.txt", "w")
if cfg['sim_mode'] == 'replicate':
    # label_split for replicate refers to base_label_split; write base splits only
    for j in range(len(base_label_split)):
        class_log_file.write(f"BaseSplit {j} Classes {base_label_split[j]}\n")
    # for virtual clients we will map at runtime; optionally note mapping later
else:
    for i in range(cfg['n_device']):
        class_log_file.write(f"Classes {label_split[i]}\n")
class_log_file.close()

# # make dataloader
# train_loader = []
# for i in range(cfg['n_device']):
#     train_set = SplitDataset(dataset['train'], data_split['train'][i])
#     # train_loader.append(DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=1, pin_memory=True))

# NOTE: to support very large `n_device` we do NOT pre-create a DataLoader per client
# (that would use a lot of memory). Instead we create DataLoaders on-the-fly for
# the selected devices inside the training loop. This makes it possible to set
# `--n_device` to a very large number and only materialize loaders for the
# few clients participating each round.
val_loader = DataLoader(dataset['val'], batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

# create model
BW_cnt = 0
TBW = cfg['TBW']
model_list = []
model_log = []

if cfg['device_ratio'] == 'W10':
    model_list.append(model_fn(hidden_size['1'], n_class=cfg['n_class']).to(device))
    model_log.append(1)
    total_bytes=0
    for param in model_list[0].parameters():
        total_bytes += param.data.nelement() * param.data.element_size()
    print(f"total bytes of model[{0}]: {total_bytes}")
elif cfg['device_ratio'] == 'S10':
    model_list.append(model_fn(hidden_size[str(TBW)], n_class=cfg['n_class']).to(device))
    model_log.append(TBW)
    total_bytes=0
    for param in model_list[0].parameters():
        total_bytes += param.data.nelement() * param.data.element_size()
    print(f"total bytes of model[{0}]: {total_bytes}")
else:
    if cfg['fix_split'] == -1: # progressive splitting
        i=0
        while TBW != 1:
            target = math.ceil(TBW/2)
            model_list.append(model_fn(hidden_size[str(target)], n_class=cfg['n_class']).to(device))
            model_log.append(target)
            TBW -= target
            total_bytes = 0
            for param in model_list[i].parameters():
                total_bytes += param.data.nelement() * param.data.element_size()
            print(f"total bytes of model[{TBW}]: {total_bytes}")
            i+=1
        model_list.append(model_fn(hidden_size['1'], n_class=cfg['n_class']).to(device))
        model_log.append(1)
        print(f"model list has {len(model_list)} elements")
    else: # fixed splitting
        target = math.ceil(TBW/2)
        remain = TBW - target
        while target >= cfg['fix_split']:
            model_list.append(model_fn(hidden_size[str(cfg['fix_split'])], n_class=cfg['n_class']).to(device))
            model_log.append(cfg['fix_split'])
            target -= cfg['fix_split']
        for _ in range(remain + target):         
            model_list.append(model_fn(hidden_size['1'], n_class=cfg['n_class']).to(device))
            model_log.append(1)

# calculate the number of BW
for i in model_log:
    if i == 1:
        BW_cnt += 1

model_list = model_list[::-1]
model_log = model_log[::-1]
if cfg['only_weak']:
    model_list = model_list[:BW_cnt]
    model_log = model_log[:BW_cnt]
print(f'BW_cnt: {BW_cnt}')
print(f'model_list: {model_log}')
n_model = len(model_list)

# for federation learning
fed = Federation([model_list[i].state_dict() for i in range(n_model)], n_class=cfg['n_class'], model_fn=model_fn)

# create loss function for FL training
loss_fn = LocalMaskCrossEntropyLoss(cfg['n_class'])

# setting device type according to device ratio
# Interpret `device_ratio` as relative weights (e.g., 'S2-W8') and allocate
# device counts proportionally so total devices == cfg['n_device'].
device_type = []
device_cnt = []
select = []
parts = []
for part in cfg['device_ratio'].split('-'):
    kind = part[0]
    try:
        weight = int(part[1:])
    except Exception:
        weight = 1
    parts.append((kind, weight))

# compute counts per kind proportional to n_device
total_weight = sum(w for _, w in parts)
remaining = cfg['n_device']
counts = []
for i, (kind, weight) in enumerate(parts):
    if i == len(parts) - 1:
        cnt = remaining
    else:
        cnt = int(round(cfg['n_device'] * (weight / total_weight)))
        remaining -= cnt
    counts.append((kind, cnt))

for kind, cnt in counts:
    device_type += [kind] * cnt
    device_cnt.append(cnt)

print(f'device_type counts: {device_cnt} (sum={sum(device_cnt)})')

# create model tag for saving model
time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
model_tag = f"{cfg['dataset']}_non-iid-{cfg['n_split']}_train-ratio-{cfg['train_ratio']}"
model_name = 'Conv'
if cfg['device_ratio'] == 'W10':
    model_tag += '_FedAvg_small'
elif cfg['device_ratio'] == 'S10':
    model_tag += '_FedAvg_large'
else:
    if cfg['fix_split'] != -1:
        model_tag += f"_fix-split-{cfg['fix_split']}"
    if cfg['only_strong']:
        model_tag += '_only-strong'
    elif cfg['only_weak']:
        model_tag += '_only-weak'
if model_fn == ResNet:
    model_tag += '_ResNet'
    model_name = 'ResNet'
print(f'model tag: {model_tag}')

output_dir = f"./output/{model_tag}/{cfg['device_ratio']}_{time_stamp}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs('./log', exist_ok=True)


# if cfg['log']:
print("inititating wandb...")
if cfg['log']:
    wandb.init(group=output_dir, project='CIFAR100_ResNet152_fedfold')
log_f = open(f"./log/{model_tag}_{cfg['device_ratio']}.txt", 'w')


# Build cumulative device boundaries and compute per-kind selection counts (`select`).
# device_cnt currently holds counts per device kind, e.g. [20, 80].
# device_bounds will be [0, 20, 100] (start/end indices for each kind).
device_bounds = [0]
for cnt in device_cnt:
    device_bounds.append(device_bounds[-1] + cnt)

# Compute how many devices to select per kind depending on selection_mode.
# Two modes supported: 'fixed' -> cfg['selected_device'] total; 'proportional' -> participation_rate * n_device.
if cfg['selection_mode'] == 'fixed':
    total_selected = int(cfg['selected_device'])
else:
    total_selected = int(round(cfg['participation_rate'] * cfg['n_device']))

# Ensure at least one client is selected to avoid empty selections
total_selected = max(1, total_selected)

# Distribute total_selected across kinds proportionally to their counts.
select = []
selected_so_far = 0
for i, cnt in enumerate(device_cnt):
    if i == len(device_cnt) - 1:
        # ensure sum(select) == total_selected
        s = max(0, total_selected - selected_so_far)
    else:
        s = int(round(total_selected * (cnt / max(1, cfg['n_device']))))
        selected_so_far += s
    select.append(s)

print(f"selection mode: {cfg['selection_mode']}, total_selected: {total_selected}, per-kind select: {select}")

# start training
BW_idx = 0
best_acc = [0.0]*n_model
model_cnt = [0]*n_model

# write experiment config to metadata file
meta_path = os.path.join(output_dir, 'run_meta.json')
with open(meta_path, 'w') as mf:
    json.dump({
        'cfg': cfg,
        'time_stamp': time_stamp,
        'model_tag': model_tag
    }, mf, indent=2)

# open per-epoch metrics file (json lines)
metrics_path = os.path.join(output_dir, 'epoch_metrics.jsonl')
metrics_f = open(metrics_path, 'w')

#calculate total training time
total_comp_time = 0.0
total_cpu_comp_time = 0.0
start = time.time()

for epoch in range(1, cfg['global_epochs'] + 1):
    # If replicate mode, generate a randomized mapping from virtual clients -> base splits
    if cfg['sim_mode'] == 'replicate':
        K = len(base_split_dict)
        # create balanced mapping by tiling base indices then shuffling
        tile = np.tile(np.arange(K), int(np.ceil(cfg['n_device'] / K)))[:cfg['n_device']]
        mapping = np.random.permutation(tile)
        # mapping[i] gives which base split client i uses this epoch
    else:
        mapping = None

    # per-epoch instrumentation collectors
    epoch_record = {
        'epoch': epoch,
        'selected_clients': [],
        'client_uploads': [],
        'total_wall_time': 0.0,
        'total_cpu_time': 0.0,
        'total_cuda_time': 0.0
    }
    device_idx = []
    if cfg['only_strong']:
        # strong devices are in device_bounds[0:2]
        device_idx += np.random.permutation(np.arange(device_bounds[0], device_bounds[1])).tolist()[:select[0]]
    elif cfg['only_weak']:
        # weak devices are in device_bounds[1:3] (assuming two kinds)
        device_idx += np.random.permutation(np.arange(device_bounds[1], device_bounds[2])).tolist()[:select[1]]
    else:
        for i in range(len(select)):
            device_idx += np.random.permutation(np.arange(device_bounds[i], device_bounds[i+1])).tolist()[:select[i]]
  
    print(f"Epoch {epoch}: {cfg['device_ratio']}, {device_idx}")

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
    for i in device_idx:
        # decide the model idx to train
        model_idx = []
        if cfg['device_ratio'] in ['W10', 'S10']:
            model_idx = [0]
        else:
            if device_type[i] == 'S':
                model_idx = [i for i in range(n_model)]
            elif device_type[i] == 'M':
                medium_BW = cfg['medium_BW']
                for j in range(len(model_log) - 1, -1, -1):
                    if medium_BW >= model_log[j]:
                        if model_log[j] == 1:
                            model_idx.append(BW_idx)
                            BW_idx = (BW_idx + 1) % BW_cnt
                        else:
                            model_idx.append(j)
                        medium_BW -= model_log[j]
            elif device_type[i] == 'W':
                model_idx = [BW_idx]
                BW_idx = (BW_idx + 1) % BW_cnt

            
        # download global model parameter from server and create optimizer
        optimizers= {}
        for idx in model_idx:
            fed.download(model_list[idx].state_dict(), idx)
            # global_param_dict = fed.global_params[idx] 
    
            # Ensure that you have the correct model parameters
            # for i, local_param in enumerate(model_list[idx].parameters()):
            #     # Extract the corresponding global parameter from the global_param_dict
            #     global_param_tensor = global_param_dict[list(global_param_dict.keys())[i]]

            #     print(f"Local Param {i} Data Ptr: {local_param.data.data_ptr()}")
            #     print(f"Global Param {i} Data Ptr: {global_param_tensor.data.data_ptr()}")
            optimizers[idx] = torch.optim.SGD(model_list[idx].parameters(), lr=cfg['lr'], momentum=0.9,  weight_decay=5e-4)
            # optimizers[idx] = torch.optim.SGD(model_list[idx].parameters(), lr=cfg['lr'],  weight_decay=5e-4)
        
        WALL_TIME = 0.0
        cuda_time = 0.0
        CPU_TIME = 0.0
        # create DataLoader for this device on-the-fly to save memory when n_device is large
        if cfg['sim_mode'] == 'replicate':
            src = int(mapping[i])
            train_indices = base_split_dict[src]
            client_label_split = base_label_split[src]
        else:
            train_indices = data_split['train'][i]
            client_label_split = label_split[i]
        train_set = SplitDataset(dataset['train'], train_indices)
        loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=1, pin_memory=True)
        for local_epoch in range(cfg['local_epochs']):
            for img, label in loader:
                img, label = img.to(device), label.to(device)

                for idx in model_idx:
                    model = model_list[idx]
                    model.train()

                    TIME_START = time.time()
                    cpu_start = time.process_time()
                    if torch.cuda.is_available():
                        cuda_start = torch.cuda.Event(enable_timing=True)
                        cuda_end = torch.cuda.Event(enable_timing=True)
                    else:
                        cuda_start = None
                        cuda_end = None
                    if cuda_start is not None:
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
                    if cuda_end is not None:
                        cuda_end.record()
                        torch.cuda.synchronize()
                        cuda_time += cuda_start.elapsed_time(cuda_end)
                    else:
                        cuda_time = 0.0

                    CPU_TIME += time.process_time()-CPU_TIME_START
                    
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

        # select quantize rate
        current_bits = 32
        if cfg['quantize']:
            bits_list = [1, 2, 4, 8, 16, 32]
            if cfg['quant_bits'] == -1: # default: random
                current_bits = bits_list[random.randint(0, len(bits_list) - 1)] 
                print(f"Random quantize rate with current bit: {current_bits}")
            elif cfg['quant_bits'] in bits_list:
                current_bits = cfg['quant_bits']
                print(f"Fix quantize rate with current bit: {current_bits}")
            else:
                print(f"Warning: Invalid quantization bits ({cfg['quant_bits']}). Defaulting to random.")
                current_bits = bits_list[random.randint(0, len(bits_list) - 1)]
            random_number = random.randint(0, 5)

        # upload local model parameter to server
        if device_type[i] == 'S'or device_type[i] == 'M':                
            models = []
            split_size = 1

            reversed_model_idx = model_idx.copy()
            reversed_model_idx.reverse()
            for idx in reversed_model_idx:
                models.append(model_list[idx].state_dict())

            local_model = Utils.accum_model(models)
            split_models = Utils.split_model(local_model, split_size, model_name, 1, cfg['n_class'])
            aggregate_model = Utils.accum_model(split_models)
            total_bytes = 0
            for param_name, param in aggregate_model.items():
                total_bytes += param.nelement() * param.element_size()
            print(f"total bytes: {total_bytes}") 
            # quantize models
            if cfg['quantize']:
                quantized_dict = {k: Compressor.quantize(v, current_bits) for k, v in aggregate_model.items()}  
                total_bytes_quantized = 0
                for k, v in quantized_dict.items():
                    total_bytes_quantized += v.nelement() * (current_bits / 8)  # bits to bytes
                print(f"Total bytes of model[{idx}] after quantization: {total_bytes_quantized}")
            else:
                quantized_dict = aggregate_model
            # time the upload to collect a simple upload duration metric
            up_start = time.time()
            fed.upload(quantized_dict, 0, 0, 1)
            up_time = time.time() - up_start
            uploaded_bytes = 0
            if cfg['quantize']:
                for k, v in quantized_dict.items():
                    uploaded_bytes += v.nelement() * (current_bits / 8)
            else:
                for k, v in aggregate_model.items():
                    uploaded_bytes += v.nelement() * v.element_size()
            epoch_record['client_uploads'].append({'client': i, 'bytes': uploaded_bytes, 'upload_time': up_time, 'model_idx': 0})
            fed.upload(quantized_dict, 0, 0, 1)           
        else:
            for idx in model_idx:
                if cfg['quantize']:
                    quantized_dict = {k: Compressor.quantize(v, current_bits) for k, v in model_list[idx].state_dict().items()} 
                    up_start = time.time()
                    fed.upload(quantized_dict, idx, 0, 0)
                    up_time = time.time() - up_start
                    uploaded_bytes = 0
                    for k, v in quantized_dict.items():
                        uploaded_bytes += v.nelement() * (current_bits / 8)
                    epoch_record['client_uploads'].append({'client': i, 'bytes': uploaded_bytes, 'upload_time': up_time, 'model_idx': idx})
                else:
                    up_start = time.time()
                    fed.upload(model_list[idx].state_dict(), idx,0,0)
                    up_time = time.time() - up_start
                    uploaded_bytes = 0
                    for k, v in model_list[idx].state_dict().items():
                        uploaded_bytes += v.nelement() * v.element_size()
                    epoch_record['client_uploads'].append({'client': i, 'bytes': uploaded_bytes, 'upload_time': up_time, 'model_idx': idx})
        print(f'Device {i} train {model_idx}, Wall time: {WALL_TIME:.3f}s, CPU time: {CPU_TIME:.3f}s, cuda time: {cuda_time:.3f}s')       
        MAX_WALL_TIME = max(MAX_WALL_TIME, WALL_TIME)
        MAX_CPU_TIME = max(MAX_CPU_TIME, CPU_TIME)
        total_comp_time+=MAX_WALL_TIME
        total_cpu_comp_time += MAX_CPU_TIME
        

    # after local training, server aggregates model parameter
    step = epoch
    fed.aggregate()
    # finalize epoch_record timing
    epoch_record['total_wall_time'] = total_comp_time
    epoch_record['total_cpu_time'] = total_cpu_comp_time
    epoch_record['total_cuda_time'] = 0.0
    # list selected clients
    epoch_record['selected_clients'] = device_idx
    # write epoch_record to jsonl
    metrics_f.write(json.dumps(epoch_record) + "\n")
    metrics_f.flush()
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
    val_acc = [0.0]*n_model
    prediction = [None]*n_model # for each split model
    prediction_ens = None # for ensemble
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


    if cfg['log']:
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
    # for ploting convergence time
    print(f"{val_acc}", file=log_f)
    log_f.flush()
end = time.time()
cpu_end = time.process_time()

metrics_f.close()

print(f"Cuda time: {cuda_start.elapsed_time(cuda_end)}ms")
print(f"Parallel Wall time: {total_comp_time:.3f}ds")
print(f"Parallel CPU time: {total_cpu_comp_time:.3f}ds")
print(f"Total time: {(end - start):.2f}s")
print(f"Total CPU time: {(cpu_end - cpu_start):.2f}s")
