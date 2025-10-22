import numpy as np
import torch
import os
from collections import defaultdict
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, CIFAR100, EMNIST
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import CNN, resnet18, ResNet9, VGG16


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
        model_fn = resnet18
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


def split_dataset(labels, n_device, n_class, n_split, label_split=None):
    data_split = {i: [] for i in range(n_device)}  # {client: [data idx]}

    label_data_idx = defaultdict(list)  # {class: [data idx]}
    for i in range(len(labels)):
        label_data_idx[labels[i]].append(i)

    # number of devices can be assigned by each class
    device_per_label = n_split * n_device // n_class  # device per label: 20
    device_per_label_list = [device_per_label for _ in range(n_class)]
    remain = np.random.choice(n_class, n_split * n_device % n_class, replace=False)
    # print(remain)
    for i in remain:
        device_per_label_list[i] += 1

    # split label_data_idx to number of device_per_label
    for label, data_idx in label_data_idx.items():
        # label_data_idx[label] = np.array(data_idx).reshape((device_per_label, -1)).tolist()
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

    data_split, label_split = split_dataset(labels['train'], cfg['n_device'], cfg['n_class'], cfg['n_split'])
    # data_split, label_split = split_dataset_noniid(labels['train'], cfg['n_device'], cfg['n_class'], cfg['p'])

    # Print class distribution for each device
    print("Class distribution for each device:")
    for device_id, classes in enumerate(label_split):
        print(f"Device {device_id} has classes: {sorted(classes)}")

    train_loaders = []
    for i in range(cfg['n_device']):
        train_set = SplitDataset(dataset['train'], data_split[i])
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=1, pin_memory=True)
        train_loaders.append(train_loader)

    val_loader = DataLoader(dataset['val'], batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=cfg['batch_size'], shuffle=False, num_workers=1,
                             pin_memory=True)

    return train_loaders, val_loader, test_loader


def split_dataset_noniid(labels, n_device, n_class, p, label_split=None):
    """
    Split dataset in a non-IID manner where each worker gets p proportion of a unique class
    and the remaining data is distributed uniformly across other workers.

    Args:
    - labels (list): The list of labels in the dataset.
    - n_device (int): The number of devices/workers.
    - n_class (int): The number of classes in the dataset.
    - p (float): The non-IID level, proportion of data each worker gets from a unique class.
    - label_split (list): Optional predefined label splits for workers.

    Returns:
    - data_split (dict): A dictionary mapping each worker to its list of data indices.
    - label_split (list): A list of label assignments for each worker.
    """
    # Initialize data splits for each worker
    data_split = {i: [] for i in range(n_device)}

    # Group indices by label
    label_data_idx = defaultdict(list)
    for i, label in enumerate(labels):
        label_data_idx[label].append(i)

    assert 0 <= p <= 1, "Proportion p must be between 0 and 1"

    # Initialize label_split if not provided
    if label_split is None:
        label_split = [[] for _ in range(n_device)]

    # Step 1: Assign p portion of a unique class to each worker
    for i in range(n_device):
        unique_class = i % n_class
        np.random.shuffle(label_data_idx[unique_class])

        # Calculate how much data to give this worker from their unique class
        # print(len(label_data_idx[unique_class]))
        unique_class_size = int(len(label_data_idx[unique_class]) * p)
        # print(unique_class_size)
        data_split[i].extend(label_data_idx[unique_class][:unique_class_size])
        label_split[i].append(unique_class)  # Keep track of the class

        # Remove the assigned data from the unique class list
        label_data_idx[unique_class] = label_data_idx[unique_class][unique_class_size:]

    # Step 2: Distribute remaining data uniformly across all workers
    for label, data_idx in label_data_idx.items():
        np.random.shuffle(data_idx)

        # Distribute the remaining data across workers uniformly
        split_data = np.array_split(data_idx, n_device)
        for i in range(n_device):
            data_split[i].extend(split_data[i])
            if label not in label_split[i]:
                label_split[i].append(label)

    # Shuffle each worker's data
    for i in range(n_device):
        np.random.shuffle(data_split[i])

    return data_split, label_split


# import numpy as np
# import torch
# import os
# from collections import defaultdict
# from torchvision import transforms
# from torchvision.datasets import CIFAR10, SVHN, CIFAR100, EMNIST
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from model import VGG16, ResNet9, ResNet18, VGG9
#
#
# def set_parameters(cfg):
#     if cfg['dataset'] == 'CIFAR10':
#         model_fn = ResNet18
#         cfg['n_class'] = 10
#
#     elif cfg['dataset'] == 'CIFAR100':
#         model_fn = ResNet9
#         cfg['n_class'] = 100
#
#     elif cfg['dataset'] == 'EMNIST':
#         model_fn = VGG9
#         cfg['n_class'] = 47
#     else:
#         model_fn = VGG16
#     return model_fn
#
#
# class SplitDataset(Dataset):
#     def __init__(self, dataset, data_idx):
#         super().__init__()
#         self.dataset = dataset
#         self.data_idx = data_idx
#
#     def __len__(self):
#         return len(self.data_idx)
#
#     def __getitem__(self, idx):
#         img, label = self.dataset[self.data_idx[idx]]
#         return img, label
#
#
# def get_dataset(dataset_name, val_ratio=0.2):
#     print(f'fetching dataset: {dataset_name}')
#     # os.makedirs('/home/b09705024/FL/dataset/', exist_ok=True)
#     root = f'../dataset/{dataset_name}'
#     dataset = {}
#     labels = {}
#     if dataset_name == 'CIFAR10':
#         train_transforms = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         ])
#         test_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         ])
#         # train_data = datasets.load_dataset(path='cifar10', split='train')
#         # test_data = datasets.load_dataset(path='cifar10', split='test')
#         dataset['train'] = CIFAR10(root, train=True, download=True, transform=train_transforms)
#         dataset['test'] = CIFAR10(root, train=False, download=True, transform=test_transforms)
#         val_size = int(len(dataset['train']) * val_ratio)
#         train_size = len(dataset['train']) - val_size
#         dataset['train'], dataset['val'] = torch.utils.data.random_split(dataset['train'], [train_size, val_size])
#         labels['train'] = [dataset['train'].dataset.targets[i] for i in dataset['train'].indices]
#         labels['val'] = [dataset['val'].dataset.targets[i] for i in dataset['val'].indices]
#         labels['test'] = dataset['test'].targets
#     elif dataset_name == 'CIFAR100':
#         train_transforms = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
#         ])
#         test_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
#         ])
#         dataset['train'] = CIFAR100(root, train=True, download=True, transform=train_transforms)
#         dataset['test'] = CIFAR100(root, train=False, download=True, transform=test_transforms)
#         val_size = int(len(dataset['train']) * val_ratio)
#         train_size = len(dataset['train']) - val_size
#         dataset['train'], dataset['val'] = torch.utils.data.random_split(dataset['train'], [train_size, val_size])
#         labels['train'] = [dataset['train'].dataset.targets[i] for i in dataset['train'].indices]
#         labels['val'] = [dataset['val'].dataset.targets[i] for i in dataset['val'].indices]
#         labels['test'] = dataset['test'].targets
#     elif dataset_name == 'EMNIST':
#         # Example for 'balanced' split. Adjust if you want a different split.
#         split_type = 'balanced'
#         train_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1751,), (0.3331,))
#         ])
#         test_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1751,), (0.3331,))
#         ])
#         dataset['train'] = EMNIST(root, split=split_type, train=True, download=True, transform=train_transforms)
#         dataset['test'] = EMNIST(root, split=split_type, train=False, download=True, transform=test_transforms)
#         val_size = int(len(dataset['train']) * val_ratio)
#         train_size = len(dataset['train']) - val_size
#         dataset['train'], dataset['val'] = torch.utils.data.random_split(dataset['train'], [train_size, val_size])
#         labels['train'] = [dataset['train'].dataset.targets[i] for i in dataset['train'].indices]
#         labels['val'] = [dataset['val'].dataset.targets[i] for i in dataset['val'].indices]
#         labels['test'] = dataset['test'].targets
#
#     print(f"Train: {len(labels['train'])}, Val: {len(labels['val'])}, Test: {len(labels['test'])}")
#     print('data ready')
#     return dataset, labels
#
#
# def iid_partition(labels, n_device, n_class, n_split):
#     """
#     Split the dataset labels into IID partitions.
#     Each client gets an approximately equal share of the data samples.
#
#     Args:
#     - labels: The dataset labels.
#     - n_device: Number of devices (clients).
#     - n_class: Number of classes (not used in IID partitioning, but included for consistency).
#     - n_split: Number of data splits per class (not used in IID partitioning, but included for consistency).
#
#     Returns:
#     - data_split: Dictionary mapping client IDs to indices of data samples.
#     """
#     data_split = {i: [] for i in range(n_device)}
#
#     # Get all the indices for the dataset
#     all_indices = np.arange(len(labels))
#
#     # Shuffle the indices to ensure random distribution
#     np.random.shuffle(all_indices)
#
#     # Partition the indices equally among clients
#     split_size = len(all_indices) // n_device
#     for i in range(n_device):
#         start_index = i * split_size
#         end_index = (i + 1) * split_size if i != n_device - 1 else len(all_indices)
#         data_split[i] = all_indices[start_index:end_index].tolist()
#
#     return data_split, None
#
#
# def lda_partition(labels, n_device, n_class, alpha):
#     label_indices = defaultdict(list)
#     class_distribution = {i: {c: 0 for c in range(n_class)} for i in range(n_device)}
#
#     # Collect indices for each class
#     for idx, label in enumerate(labels):
#         label_indices[label].append(idx)
#
#     # Shuffle the indices to randomize
#     for k in label_indices.keys():
#         np.random.shuffle(label_indices[k])
#
#     # Dirichlet distribution for each client
#     client_class_distribution = np.random.dirichlet([alpha] * n_class, n_device)
#
#     data_split = {i: [] for i in range(n_device)}
#
#     for c in range(n_class):
#         class_indices = label_indices[c]
#         np.random.shuffle(class_indices)
#         class_size = len(class_indices)
#         class_indices_split = np.split(class_indices, np.cumsum(client_class_distribution[:, c])[:-1] * class_size)
#
#         for i in range(n_device):
#             data_split[i].extend(class_indices_split[i].tolist())
#             class_distribution[i][c] += len(class_indices_split[i])
#
#     return data_split, class_distribution
#
#
# def skewed_label_partition(labels, n_device, n_class, psi):
#     data_split = {i: [] for i in range(n_device)}
#     class_distribution = {i: {c: 0 for c in range(n_class)} for i in range(n_device)}
#     all_classes = np.arange(n_class)
#
#     for i in range(n_device):
#         # Randomly select a subset of classes for this client
#         selected_classes = np.random.choice(all_classes, n_class - psi, replace=False)
#         for c in selected_classes:
#             indices = np.where(np.array(labels) == c)[0]
#             data_split[i].extend(indices.tolist())
#             class_distribution[i][c] += len(indices)
#
#     return data_split, class_distribution
#
#
# def iid_partition(labels, n_device, n_class):
#     data_split = {i: [] for i in range(n_device)}
#     class_distribution = {i: {c: 0 for c in range(n_class)} for i in range(n_device)}
#
#     # Get all the indices for the dataset
#     all_indices = np.arange(len(labels))
#
#     # Shuffle the indices to ensure random distribution
#     np.random.shuffle(all_indices)
#
#     # Partition the indices equally among clients
#     split_size = len(all_indices) // n_device
#     for i in range(n_device):
#         start_index = i * split_size
#         end_index = (i + 1) * split_size if i != n_device - 1 else len(all_indices)
#         data_split[i] = all_indices[start_index:end_index].tolist()
#
#         # Track the class distribution
#         for idx in data_split[i]:
#             label = labels[idx]
#             class_distribution[i][label] += 1
#
#     return data_split, class_distribution
#
# import numpy as np
# from collections import defaultdict
#
#
# def balanced_class_partition(labels, n_device, n_class, max_classes_per_device=90):
#     """
#     Partition data such that each device receives data from a subset of classes,
#     and all classes are assigned to devices.
#
#     Args:
#     - labels: List of labels.
#     - n_device: Number of devices.
#     - n_class: Number of classes.
#     - max_classes_per_device: Maximum number of classes per device.
#
#     Returns:
#     - data_split: Dictionary mapping device IDs to data indices.
#     - class_distribution: Dictionary mapping device IDs to class counts.
#     """
#     # Initialize data structures
#     data_split = {i: [] for i in range(n_device)}
#     class_distribution = {i: {c: 0 for c in range(n_class)} for i in range(n_device)}
#     label_indices = defaultdict(list)
#
#     # Collect indices for each class
#     for idx, label in enumerate(labels):
#         label_indices[label].append(idx)
#
#     # Shuffle the class order and device order for randomness
#     class_list = list(range(n_class))
#     np.random.shuffle(class_list)
#     device_list = list(range(n_device))
#     np.random.shuffle(device_list)
#
#     # Determine the number of devices each class should be assigned to
#     min_devices_per_class = max(1, n_device * max_classes_per_device // n_class)
#
#     # Assign each class to devices
#     for c in class_list:
#         # Shuffle devices for this class
#         np.random.shuffle(device_list)
#         # Select devices to assign this class
#         devices_for_class = device_list[:min_devices_per_class]
#
#         # Shuffle the data indices for this class
#         np.random.shuffle(label_indices[c])
#
#         # Split the class data among selected devices
#         splits = np.array_split(label_indices[c], len(devices_for_class))
#
#         for device_idx, split_indices in zip(devices_for_class, splits):
#             data_split[device_idx].extend(split_indices.tolist())
#             class_distribution[device_idx][c] += len(split_indices)
#
#     return data_split, class_distribution
#
#
# def devices_per_class_partition(labels, n_device, n_class, n_split):
#     data_split = {i: [] for i in range(n_device)}  # {client: [data idx]}
#     class_distribution = {i: {c: 0 for c in range(n_class)} for i in range(n_device)}
#
#     label_data_idx = defaultdict(list)  # {class: [data idx]}
#
#     # Collect all data indices for each class
#     for idx, label in enumerate(labels):
#         label_data_idx[label].append(idx)
#
#     # For each class, distribute its data to devices
#     for c in range(n_class):
#         np.random.shuffle(label_data_idx[c])  # Shuffle the data for each class
#
#         # Split data into n_device parts to distribute across all devices
#         class_data_splits = np.array_split(label_data_idx[c], n_device)
#
#         # Shuffle device indices for randomness
#         device_indices = list(range(n_device))
#         np.random.shuffle(device_indices)
#
#         # Assign each split to a device
#         for device_idx, data_indices in zip(device_indices, class_data_splits):
#             data_split[device_idx].extend(data_indices.tolist())
#             class_distribution[device_idx][c] += len(data_indices)
#
#     return data_split, class_distribution
#
#
# def split_dataset(labels, n_device, n_class, non_iid_type=None, alpha=0.5, psi=0, n_split=2):
#     if non_iid_type == "lda":
#         data_split, class_distribution = lda_partition(labels, n_device, n_class, alpha)
#     elif non_iid_type == "skewed_label":
#         data_split, class_distribution = skewed_label_partition(labels, n_device, n_class, psi)
#     else:
#         data_split, class_distribution = balanced_class_partition(labels, n_device, n_class)
#
#     return data_split, class_distribution
#
#
# def get_data_loaders(cfg):
#     dataset, labels = get_dataset(cfg['dataset'], val_ratio=cfg.get('val_ratio', 0.2))
#
#     data_split, class_distribution = split_dataset(labels['train'], cfg['n_device'], cfg['n_class'], cfg['non_iid_type'], cfg['n_split'])
#
#     train_loaders = []
#     for i in range(cfg['n_device']):
#         train_set = SplitDataset(dataset['train'], data_split[i])
#         classes = [c for c, count in class_distribution[i].items() if count > 0]
#         num_classes = len(classes)
#         print(f"Device {i}: Number of classes: {num_classes}, Classes: {classes}")
#
#         train_loader = DataLoader(
#             train_set,
#             batch_size=cfg['batch_size'],
#             shuffle=True,
#             num_workers=1,
#             pin_memory=True
#         )
#         train_loaders.append(train_loader)
#
#     val_loader = DataLoader(dataset['val'], batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
#     test_loader = DataLoader(dataset['test'], batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
#
#     return train_loaders, val_loader, test_loader
#
