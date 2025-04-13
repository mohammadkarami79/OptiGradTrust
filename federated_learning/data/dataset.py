import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import numpy as np
from federated_learning.config.config import *

class LabelFlippingDataset(Dataset):
    """
    Dataset wrapper that implements label flipping attack.
    Flips each label l to (num_classes - l - 1).
    
    For example, in a 10-class dataset:
    0 -> 9, 1 -> 8, 2 -> 7, etc.
    """
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
    """
    Dataset wrapper that implements backdoor attack.
    Adds a trigger pattern to the data and sets all labels to a target label (0 by default).
    
    This creates poisoned examples that the model will associate with the target label.
    """
    def __init__(self, dataset, num_classes, target_label=0):
        self.dataset = dataset
        self.num_classes = num_classes
        self.target_label = target_label
        self.trigger = self.create_trigger()

    def create_trigger(self):
        """
        Creates a trigger pattern to add to images.
        Currently creates a small white square in the bottom right corner.
        """
        trigger = torch.zeros((1, 28, 28))
        trigger[:, 24:, 24:] = 1.0
        return trigger

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, _ = self.dataset[idx]  # Original target is ignored
        # Add trigger to the image
        data = data + self.trigger
        data = torch.clamp(data, 0, 1)  # Ensure pixel values stay in valid range
        # Set to target label
        return data, self.target_label

class AdaptiveAttackDataset(Dataset):
    """
    Dataset wrapper for adaptive attacks.
    
    Can be extended with specific modifications to evade defensive measures.
    Currently acts as a pass-through wrapper.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Can implement sophisticated adaptive data poisoning here
        return data, target

class MinMaxAttackDataset(Dataset):
    """
    Dataset wrapper that implements the min-max attack as described in FLTrust paper.
    
    This attack tries to maximize negative impact on accuracy while minimizing 
    the chance of being detected. It specifically targets the most confusing class
    for each true class.
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        # Create a confusion matrix to determine most confusing classes
        # For simplicity, we use a predefined pattern: target = (original + 1) % num_classes
        self.confusion_map = [(i + 1) % num_classes for i in range(num_classes)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Map to the confusing class
        confused_target = self.confusion_map[target]
        return data, confused_target

class MinSumAttackDataset(Dataset):
    """
    Dataset wrapper that implements the min-sum attack.
    
    This attack aims to minimize the sum of cosine similarities with benign updates.
    In practice, it creates a consistent but subtle pattern across all samples
    to gradually push the model in a specific wrong direction.
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        # Create a probability map - instead of just flipping, we introduce uncertainty
        self.prob_map = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            # Invert the probability distribution - highest prob for opposite class
            for j in range(num_classes):
                if i == j:
                    self.prob_map[i, j] = 0.1  # Small prob for correct class
                else:
                    # Higher prob for classes further away (in circular sense)
                    distance = min((j - i) % num_classes, (i - j) % num_classes)
                    self.prob_map[i, j] = 1.0 / distance if distance > 0 else 0.1
            # Normalize to create a probability distribution
            self.prob_map[i] = self.prob_map[i] / self.prob_map[i].sum()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Sample from the probability distribution for this class
        new_target = torch.multinomial(self.prob_map[target], 1).item()
        return data, new_target

class AlternatingAttackDataset(Dataset):
    """
    Dataset wrapper that implements the alternating attack.
    
    This attack alternates between correct and incorrect labels in a pattern
    that's difficult to detect but creates systematic bias.
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        # Create alternating patterns for different samples
        self.alternating_offset = np.random.randint(0, num_classes, size=len(dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Use sample index to determine if we should alter the label
        if idx % 2 == 0:
            # Even indices: use offset to create deterministic but varied pattern
            altered_target = (target + self.alternating_offset[idx]) % self.num_classes
            return data, altered_target
        else:
            # Odd indices: keep original
            return data, target

class TargetedAttackDataset(Dataset):
    """
    Dataset wrapper that implements a targeted attack.
    
    This attack focuses on a specific subset of the data (e.g., a particular class)
    and makes targeted modifications to mislead the model specifically on that subset.
    """
    def __init__(self, dataset, num_classes, target_class=0, target_output=1):
        self.dataset = dataset
        self.num_classes = num_classes
        self.target_class = target_class  # The class to attack
        self.target_output = target_output  # The incorrect output to produce

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Only modify samples from the target class
        if target == self.target_class:
            return data, self.target_output
        return data, target

class GradientInversionAttackDataset(Dataset):
    """
    Dataset wrapper that implements a gradient inversion attack.
    
    This sophisticated attack creates a pattern of label modifications
    that result in gradient updates that have varying effects on different
    parts of the model, making detection more difficult.
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        # Create patterns for different sections of the dataset
        dataset_size = len(dataset)
        quarter_size = dataset_size // 4
        
        # Different strategies for different quarters
        self.strategies = [
            lambda x: (x + 1) % num_classes,                  # First quarter: simple shift
            lambda x: (x + num_classes // 2) % num_classes,   # Second quarter: maximum distance
            lambda x: x,                                      # Third quarter: unchanged
            lambda x: num_classes - x - 1                     # Fourth quarter: inversion
        ]
        
        # Map each index to its quarter
        self.quarter_map = np.zeros(dataset_size, dtype=int)
        for i in range(4):
            start = i * quarter_size
            end = (i + 1) * quarter_size if i < 3 else dataset_size
            self.quarter_map[start:end] = i

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Apply the strategy for this sample's quarter
        quarter = self.quarter_map[idx]
        modified_target = self.strategies[quarter](target)
        return data, modified_target

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