import torch
import numpy as np


def load_data(token_ids, batch_size, context_length, device):
    max_start_idx = len(token_ids) - context_length
    sampled_start_idx = np.random.randint(0, max_start_idx, batch_size)
    
    inputs = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    targets = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    for idx, start in enumerate(sampled_start_idx):
        inputs[idx] = torch.tensor(token_ids[start:start + context_length], device=device)
        targets[idx] = torch.tensor(token_ids[start + 1:start + 1 + context_length], device=device)
    
    return inputs, targets
