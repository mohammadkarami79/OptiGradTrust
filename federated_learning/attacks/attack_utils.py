import torch
from federated_learning.config.config import *

def simulate_attack(raw_grad, attack_type):
    if attack_type == 'label_flipping':
        return -raw_grad
    elif attack_type == 'scaling_attack':
        return raw_grad * NUM_CLIENTS
    elif attack_type == 'partial_scaling_attack':
        scaling_factor = NUM_CLIENTS
        mask = (torch.rand(raw_grad.shape, device=raw_grad.device) < 0.66).float()
        return raw_grad * (mask * scaling_factor + (1 - mask))
    elif attack_type == 'backdoor_attack':
        return raw_grad + torch.ones_like(raw_grad) * NUM_CLIENTS
    elif attack_type == 'adaptive_attack':
        return raw_grad + 0.1 * torch.randn_like(raw_grad)
    else:
        return raw_grad 