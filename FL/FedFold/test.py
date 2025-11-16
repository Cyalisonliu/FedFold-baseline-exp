import os
import random
import numpy as np
import time
import argparse
import copy
import math
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from data import SplitDataset, get_dataset, split_dataset, set_parameters
from model import Conv, LocalMaskCrossEntropyLoss, MLP, ResNet
from fed import Federation

cfg = {
    # general
    'seed': 2,
    'log': False,

    # data
    'dataset': 'CIFAR10', #'Otto', 'SVHN'
    'n_split': 2,
    'val_ratio': 0.2,

    # FL
    'n_device': 100,
    'selected_device': 10,
    'device_ratio': 'S2-W8',

    #  training
    'batch_size': 32,
    'lr': 1e-2,
    'global_epochs': 300,
    'local_epochs': 5,
}
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--weight', action='store_true', help='enable weighted ensemble')
parser.add_argument('--device_ratio', type=str, default='S2-W8', help='device ratio')
parser.add_argument('--train_ratio', type=str, default='16-1', help='trainable width ratio')
parser.add_argument('--fix_split', type=int, default=-1, help='fixed split ratio')
parser.add_argument('--only_strong', action='store_true', help='train only strong devices')
parser.add_argument('--only_weak', action='store_true', help='train only weak devices')
parser.add_argument('--n_split', type=int, default=2, help='number of splits for non-iid data')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
args = parser.parse_args()

cfg['device_ratio'] = args.device_ratio
cfg['weight'] = args.weight
cfg['only_strong'] = args.only_strong
cfg['only_weak'] = args.only_weak
cfg['train_ratio'] = args.train_ratio
cfg['fix_split'] = args.fix_split
cfg['n_split'] = args.n_split
cfg['TBW'] = int(args.train_ratio.split('-')[0])
cfg['medium_BW'] = int(args.train_ratio.split('-')[1]) # for three types of devices
cfg['dataset'] = args.dataset

# get model function and hidden size according to dataset
model_fn, hidden_size = set_parameters(cfg)

# get model path
model_tag = f"{cfg['dataset']}_non-iid-{cfg['n_split']}_train-ratio-{cfg['train_ratio']}"
if cfg['device_ratio'] == 'W10':
    model_tag += '_FedAvg_small'
elif cfg['device_ratio'] == 'S10':
    model_tag += '_FedAvg_large'
else:
    if cfg['fix_split'] != -1:
        model_tag += f"_fix-split-{cfg['fix_split']}"
    if cfg['only_strong']:
        model_tag += '_only-strong'
    if cfg['only_weak']:
        model_tag += '_only-weak'
if model_fn == ResNet:
    model_tag += '_ResNet'
print(f'model tag: {model_tag}')

model_path = f'./output/{model_tag}'
model_group = None
for i, dir_name in enumerate(sorted(os.listdir(model_path), reverse=True)):
    # only inference on the lastest model
    if cfg['device_ratio'] in dir_name: #and i%3 == 0:
        print(i, dir_name)
        # dir_name = 'S9-W1_2024-11-14 15:06:39'
        model_group = dir_name.split('_')[0]
        print(f'Found {dir_name} !')
        break

# if model_group == None:
#     print('Not Found!')
#     exit(0)

# get dataset
dataset, labels = get_dataset(cfg['dataset'], cfg['val_ratio'])

# make dataloader
test_loader = DataLoader(dataset['test'], batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# get device weight from device ratio
device_weight = []
for ratio in cfg['device_ratio'].split('-'):
    device_weight.append(int(ratio[1:]))
print(device_weight)

# create model
BW_cnt = 0
TBW = cfg['TBW']
model_list = []
model_log = []
if cfg['device_ratio'] == 'W10':
    model_list.append(model_fn(hidden_size['1']).to(device))
    model_log.append(1)
elif cfg['device_ratio'] == 'S10':
    model_list.append(model_fn(hidden_size[str(TBW)]).to(device))
    model_log.append(TBW)
else:
    if cfg['fix_split'] == -1: # progressive splitting
        while TBW != 1:
            target = math.ceil(TBW/2)
            model_list.append(model_fn(hidden_size[str(target)]).to(device))
            model_log.append(target)
            TBW -= target
        model_list.append(model_fn(hidden_size['1']).to(device))
        model_log.append(1)
    else: # fixed splitting
        target = math.ceil(TBW/2)
        remain = TBW - target
        while target >= cfg['fix_split']:
            model_list.append(model_fn(hidden_size[str(cfg['fix_split'])]).to(device))
            model_log.append(cfg['fix_split'])
            target -= cfg['fix_split']
        for _ in range(remain + target):         
            model_list.append(model_fn(hidden_size['1']).to(device))
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

# load model weight
for idx in range(n_model):
    model_list[idx].load_state_dict(torch.load(f'{model_path}/{dir_name}/{idx}_{model_log[idx]}.pth'))


# calculate weight
if cfg['weight']:
    device_sum = sum(device_weight)
    nonBW_weight = device_weight[0]/device_sum
    BW_weight = device_weight[-1]/device_sum

    # for three types of devices
    medium_idx = []
    medium_weight = 1 - nonBW_weight - BW_weight
    if len(device_weight) == 3:
        medium_BW = cfg['medium_BW']
        for j in range(len(model_log) - 1, -1, -1):
            if medium_BW >= model_log[j]:
                medium_idx.append(model_log[j])
                medium_BW -= model_log[j]
        # print( medium_idx)
    # print(nonBW_weight, BW_weight, medium_weight)

#print each small model's test acc
model_logits = [None for _ in range(n_model)]
with torch.no_grad():
    for img, label in test_loader:
        img, label = img.to(device), label.to(device)
        for idx in range(n_model):
            model = model_list[idx]
            model.eval()

            logits = model(img)
            if model_logits[idx] == None:
                model_logits[idx] = logits
            else:
                model_logits[idx] = torch.cat([model_logits[idx], logits], dim=0)

model_acc = []
for idx in range(n_model):
    test_acc = (model_logits[idx].argmax(dim=1) == torch.tensor(labels['test']).to(device)).float().mean()
    # test_acc = round(100*test_acc.item(), 2)
    model_acc.append(test_acc)
    # print(f'model {model_log[idx]} test acc: {test_acc}')
    # print(f'{test_acc*100}')

model_weight = []
for idx in range(n_model):
    model_weight.append(model_acc[idx]/sum(model_acc))

# start testing
prediction = None
with torch.no_grad():
    for img, label in test_loader:
        img, label = img.to(device), label.to(device)
        pred = None
        for idx in range(n_model):
            model = model_list[idx]
            model.eval()

            logits = model(img)
            logits = torch.nn.functional.softmax(logits, dim=1)

            if cfg['weight']:
                # weight = model_weight[idx]
                if idx < BW_cnt:
                    weight = BW_weight
                else:
                    weight = nonBW_weight
                
                if model_log[idx] in medium_idx:
                    weight += medium_weight
            else:
                weight = 1

            if pred == None:
                pred = logits*weight
            else:
                pred += logits*weight
        if prediction == None:
            prediction = pred
        else:
            prediction = torch.cat([prediction, pred], dim=0)

test_acc = (prediction.argmax(dim=1) == torch.tensor(labels['test']).to(device)).float().mean()
test_acc = round(100*test_acc.item(), 2)
# print(f"{cfg['device_ratio']} test acc: {test_acc}")
print(test_acc)

# save testing result
result_dir = f"../evaluation/{cfg['dataset']}"
os.makedirs(result_dir, exist_ok=True)
file_name = f"{result_dir}/{model_tag}{'_weight' if cfg['weight'] else ''}.txt"

with open(file_name, 'a+') as f:
    print(f"{cfg['device_ratio']} {test_acc}", file=f)
    f.flush()
