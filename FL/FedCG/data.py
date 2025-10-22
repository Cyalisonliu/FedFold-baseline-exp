import numpy as np
import torch
import os
from collections import defaultdict
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, CIFAR100, EMNIST
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import VGG16, ResNet9, resnet18, resnet152, VGG9, CNN


def set_parameters(cfg):
    if cfg['dataset'] == 'CIFAR10':
        if cfg['model'] == 'ResNet18':
            model_fn = resnet18
        elif cfg['model'] == 'CNN':
            model_fn = CNN
        else:
            model_fn = ResNet9
        cfg['n_class'] = 10

    elif cfg['dataset'] == 'CIFAR100':
        if cfg['model'] == 'ResNet18':
            model_fn = resnet18
        elif cfg['model'] == 'CNN':
            model_fn = CNN
        else:
            model_fn = ResNet9
        cfg['n_class'] = 100

    elif cfg['dataset'] == 'EMNIST':
        model_fn = VGG9
        cfg['n_class'] = 47
    else:
        model_fn = VGG16
    return model_fn


class SplitDataset(Dataset):
    def __init__(self, dataset, data_idx):
        super().__init__()
        self.dataset = dataset
        self.data_idx = data_idx

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        img, label = self.dataset[self.data_idx[idx]]
        return img, label


def get_dataset(dataset_name, val_ratio=0.2):
    print(f'fetching dataset: {dataset_name}')
    # os.makedirs('/home/b09705024/FL/dataset/', exist_ok=True)
    root = f'../dataset/{dataset_name}'
    dataset = {}
    labels = {}
    if dataset_name == 'CIFAR10':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        # train_data = datasets.load_dataset(path='cifar10', split='train')
        # test_data = datasets.load_dataset(path='cifar10', split='test')
        dataset['train'] = CIFAR10(root, train=True, download=True, transform=train_transforms)
        dataset['test'] = CIFAR10(root, train=False, download=True, transform=test_transforms)
        val_size = int(len(dataset['train']) * val_ratio)
        train_size = len(dataset['train']) - val_size
        dataset['train'], dataset['val'] = torch.utils.data.random_split(dataset['train'], [train_size, val_size])
        labels['train'] = [dataset['train'].dataset.targets[i] for i in dataset['train'].indices]
        labels['val'] = [dataset['val'].dataset.targets[i] for i in dataset['val'].indices]
        labels['test'] = dataset['test'].targets
    elif dataset_name == 'CIFAR100':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        dataset['train'] = CIFAR100(root, train=True, download=True, transform=train_transforms)
        dataset['test'] = CIFAR100(root, train=False, download=True, transform=test_transforms)
        val_size = int(len(dataset['train']) * val_ratio)
        train_size = len(dataset['train']) - val_size
        dataset['train'], dataset['val'] = torch.utils.data.random_split(dataset['train'], [train_size, val_size])
        labels['train'] = [dataset['train'].dataset.targets[i] for i in dataset['train'].indices]
        labels['val'] = [dataset['val'].dataset.targets[i] for i in dataset['val'].indices]
        labels['test'] = dataset['test'].targets
    elif dataset_name == 'EMNIST':
        # Example for 'balanced' split. Adjust if you want a different split.
        split_type = 'balanced'
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3331,))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3331,))
        ])
        dataset['train'] = EMNIST(root, split=split_type, train=True, download=True, transform=train_transforms)
        dataset['test'] = EMNIST(root, split=split_type, train=False, download=True, transform=test_transforms)
        val_size = int(len(dataset['train']) * val_ratio)
        train_size = len(dataset['train']) - val_size
        dataset['train'], dataset['val'] = torch.utils.data.random_split(dataset['train'], [train_size, val_size])
        labels['train'] = [dataset['train'].dataset.targets[i] for i in dataset['train'].indices]
        labels['val'] = [dataset['val'].dataset.targets[i] for i in dataset['val'].indices]
        labels['test'] = dataset['test'].targets

    print(f"Train: {len(labels['train'])}, Val: {len(labels['val'])}, Test: {len(labels['test'])}")
    print('data ready')
    return dataset, labels


def process_df(df):
    data_y = []
    target_labels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    for i in df['target']:
        data_y.append(target_labels.index(i))
    data_x = torch.tensor(np.array(df)[:, 1:-1].astype(np.float32))
    return list(zip(data_x, data_y)), data_y


def iid_partition(labels, n_device, n_class, n_split):
    """
    Split the dataset labels into IID partitions.
    Each client gets an approximately equal share of the data samples.

    Args:
    - labels: The dataset labels.
    - n_device: Number of devices (clients).
    - n_class: Number of classes (not used in IID partitioning, but included for consistency).
    - n_split: Number of data splits per class (not used in IID partitioning, but included for consistency).

    Returns:
    - data_split: Dictionary mapping client IDs to indices of data samples.
    """
    data_split = {i: [] for i in range(n_device)}

    # Get all the indices for the dataset
    all_indices = np.arange(len(labels))

    # Shuffle the indices to ensure random distribution
    np.random.shuffle(all_indices)

    # Partition the indices equally among clients
    split_size = len(all_indices) // n_device
    for i in range(n_device):
        start_index = i * split_size
        end_index = (i + 1) * split_size if i != n_device - 1 else len(all_indices)
        data_split[i] = all_indices[start_index:end_index].tolist()

    return data_split, None


def lda_partition(labels, n_device, n_class, alpha):
    label_indices = defaultdict(list)
    class_distribution = {i: {c: 0 for c in range(n_class)} for i in range(n_device)}

    # Collect indices for each class
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)

    # Shuffle the indices to randomize
    for k in label_indices.keys():
        np.random.shuffle(label_indices[k])

    # Dirichlet distribution for each client
    client_class_distribution = np.random.dirichlet([alpha] * n_class, n_device)

    data_split = {i: [] for i in range(n_device)}

    for c in range(n_class):
        class_indices = label_indices[c]
        np.random.shuffle(class_indices)
        class_size = len(class_indices)
        class_indices_split = np.split(class_indices, np.cumsum(client_class_distribution[:, c])[:-1] * class_size)

        for i in range(n_device):
            data_split[i].extend(class_indices_split[i].tolist())
            class_distribution[i][c] += len(class_indices_split[i])

    return data_split, class_distribution


def skewed_label_partition(labels, n_device, n_class, psi):
    data_split = {i: [] for i in range(n_device)}
    class_distribution = {i: {c: 0 for c in range(n_class)} for i in range(n_device)}
    all_classes = np.arange(n_class)

    for i in range(n_device):
        # Randomly select a subset of classes for this client
        selected_classes = np.random.choice(all_classes, n_class - psi, replace=False)
        for c in selected_classes:
            indices = np.where(np.array(labels) == c)[0]
            data_split[i].extend(indices.tolist())
            class_distribution[i][c] += len(indices)

    return data_split, class_distribution


def iid_partition(labels, n_device, n_class, n_split):
    data_split = {i: [] for i in range(n_device)}
    class_distribution = {i: {c: 0 for c in range(n_class)} for i in range(n_device)}

    # Get all the indices for the dataset
    all_indices = np.arange(len(labels))

    # Shuffle the indices to ensure random distribution
    np.random.shuffle(all_indices)

    # Partition the indices equally among clients
    split_size = len(all_indices) // n_device
    for i in range(n_device):
        start_index = i * split_size
        end_index = (i + 1) * split_size if i != n_device - 1 else len(all_indices)
        data_split[i] = all_indices[start_index:end_index].tolist()

        # Track the class distribution
        for idx in data_split[i]:
            label = labels[idx]
            class_distribution[i][label] += 1

    return data_split, class_distribution


def devices_per_class_partition(labels, n_device, n_class, n_split):
    """
    Partition the dataset labels such that a fixed number of devices receive data from each class.
    This function ensures that data from each class is split across `n_split` devices.

    Args:
    - labels: The dataset labels.
    - n_device: Number of devices (clients).
    - n_class: Number of classes in the dataset.
    - n_split: Number of devices that will receive data from each class.

    Returns:
    - data_split: Dictionary mapping device IDs to indices of data samples.
    - class_distribution: Dictionary mapping device IDs to class distributions.
    """
    data_split = {i: [] for i in range(n_device)}  # {client: [data idx]}
    class_distribution = {i: {c: 0 for c in range(n_class)} for i in range(n_device)}

    label_data_idx = defaultdict(list)  # {class: [data idx]}

    # Collect all data indices for each class
    for idx, label in enumerate(labels):
        label_data_idx[label].append(idx)

    # For each class, distribute its data to n_split devices
    for c in range(n_class):
        np.random.shuffle(label_data_idx[c])  # Shuffle the data for each class
        class_data = np.array_split(label_data_idx[c], n_split)  # Split data for this class into n_split parts

        # Assign each split to a set of devices
        for i, data_indices in enumerate(class_data):
            device_idx = i % n_device  # Assign to devices in a round-robin manner
            data_split[device_idx].extend(data_indices.tolist())  # Add class data to the selected device
            class_distribution[device_idx][c] += len(data_indices)  # Update class distribution

    return data_split, class_distribution


# def split_dataset(labels, n_device, n_class, non_iid_type=None, alpha=0.5, psi=0, n_split=2):
#     if non_iid_type == "lda":
#         data_split, class_distribution = lda_partition(labels, n_device, n_class, alpha)
#     elif non_iid_type == "skewed_label":
#         data_split, class_distribution = skewed_label_partition(labels, n_device, n_class, psi)
#     else:
#         data_split, class_distribution = devices_per_class_partition(labels, n_device, n_class, n_split)
#
#     return data_split, class_distribution

def split_dataset(labels, n_device, n_class, n_split, label_split=None):
    data_split = {i: [] for i in range(n_device)}  # {client: [data idx]}

    label_data_idx = defaultdict(list)  # {class: [data idx]}
    for i in range(len(labels)):
        label_data_idx[labels[i]].append(i)

    device_per_label = n_split * n_device // n_class  # device per label: 20
    device_per_label_list = [device_per_label for _ in range(n_class)]
    remain = np.random.choice(n_class, n_split * n_device % n_class, replace=False)
    for i in remain:
        device_per_label_list[i] += 1

    # Debugging output
    print(f"Device per label list: {device_per_label_list}")
    # print(f"Label data index lengths: {[len(label_data_idx[label]) for label in range(n_class)]}")

    # split label_data_idx to number of device_per_label
    for label, data_idx in label_data_idx.items():
        if device_per_label_list[label] == 0:
            raise ValueError(f"Invalid configuration: device_per_label_list[{label}] is 0.")
        num_leftover = len(data_idx) % device_per_label_list[label]
        leftover = data_idx[-num_leftover:] if num_leftover > 0 else []
        tmp = np.array(data_idx[:-num_leftover]) if num_leftover > 0 else np.array(data_idx)
        tmp = tmp.reshape((device_per_label_list[label], -1)).tolist()
        for i, leftover_data_idx in enumerate(leftover):
            tmp[i] = np.concatenate([tmp[i], [leftover_data_idx]])
        label_data_idx[label] = tmp

    # split label to number of n_split
    if label_split == None:
        label_split = []
        for _ in range(device_per_label):
            tmp = list(range(n_class))  # [0,1, ... , 9]
            tmp = torch.tensor(tmp)[torch.randperm(len(tmp))].tolist()
            label_split.append(tmp)
        label_split = np.array(label_split).reshape(-1).tolist()
        for i in remain:
            label_split.append(i)
        label_split = np.array(label_split).reshape((n_device, -1)).tolist()
        label_split = torch.tensor(label_split)[torch.randperm(len(label_split))].tolist()
        print(label_split)

    # split data idx to each client
    for i in range(n_device):
        for label in label_split[i]:  # [[0, 1], [2, 3]...], len=100
            idx = torch.arange(len(label_data_idx[label]))[torch.randperm(len(label_data_idx[label]))[0]].item()
            data_split[i].extend(label_data_idx[label].pop(idx))
        print(len(data_split[i]))
    return data_split, label_split


def get_data_loaders(cfg):
    dataset, labels = get_dataset(cfg['dataset'], val_ratio=cfg.get('val_ratio', 0.2))

    data_split, class_distribution = split_dataset(labels['train'], cfg['n_device'], cfg['n_class'], cfg['n_split'])

    train_loaders = []
    data_sizes = []
    for i in range(cfg['n_device']):
        train_set = SplitDataset(dataset['train'], data_split[i])
        # print(f"Number of samples in train_set: {len(train_set)}")
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=1, pin_memory=True)
        train_loaders.append(train_loader)
        data_sizes.append(len(train_set))

    val_loader = DataLoader(dataset['val'], batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    return train_loaders, val_loader, test_loader, data_sizes

