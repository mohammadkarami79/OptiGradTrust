import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
from federated_learning.config.config import *

class LabelFlippingDataset(Dataset):
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        flipped_target = self.num_classes - target - 1
        return data, flipped_target

class BackdoorDataset(Dataset):
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        self.trigger = self.create_trigger()

    def create_trigger(self):
        trigger = torch.zeros((1, 28, 28))
        trigger[:, 24:, 24:] = 1.0
        return trigger

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        data = data + self.trigger
        data = torch.clamp(data, 0, 1)
        target = 0
        return data, target

class AdaptiveAttackDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target

def load_dataset():
    if DATASET == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_channels = 1
    else:
        raise ValueError("Unknown dataset")
    return full_dataset, test_dataset, num_classes, input_channels

def split_dataset_non_iid(dataset, num_classes):
    client_datasets = [[] for _ in range(NUM_CLIENTS)]
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    for c in range(num_classes):
        random.shuffle(class_indices[c])
    clients_per_group = NUM_CLIENTS // num_classes
    for c in range(num_classes):
        group_clients = list(range(c * clients_per_group, (c + 1) * clients_per_group))
        other_clients = [i for i in range(NUM_CLIENTS) if i not in group_clients]
        for idx in class_indices[c]:
            if random.random() < Q:
                client_id = random.choice(group_clients)
            else:
                client_id = random.choice(other_clients)
            client_datasets[client_id].append(idx)
    client_datasets = [torch.utils.data.Subset(dataset, indices) for indices in client_datasets]
    return client_datasets

def create_root_dataset(full_dataset, num_classes):
    if BIAS_PROBABILITY == 1.0:
        biased_indices = [i for i, (_, label) in enumerate(full_dataset) if label == BIAS_CLASS]
        root_indices = random.sample(biased_indices, ROOT_DATASET_SIZE)
    else:
        biased_size = int(ROOT_DATASET_SIZE * BIAS_PROBABILITY)
        unbiased_size = ROOT_DATASET_SIZE - biased_size
        biased_indices = [i for i, (_, label) in enumerate(full_dataset) if label == BIAS_CLASS]
        unbiased_indices = [i for i, (_, label) in enumerate(full_dataset) if label != BIAS_CLASS]
        root_indices = random.sample(biased_indices, min(biased_size, len(biased_indices))) + \
                       random.sample(unbiased_indices, min(unbiased_size, len(unbiased_indices)))
    root_dataset = torch.utils.data.Subset(full_dataset, root_indices)
    return root_dataset 