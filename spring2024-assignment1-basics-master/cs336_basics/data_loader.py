import torch
import numpy as np

def load_data(token_ids, batch_size, context_length, device):
    token_ids = torch.tensor(token_ids, dtype=torch.int16, device=device)
    max_start_idx = len(token_ids) - context_length
    sampled_start_idx = np.random.choice(max_start_idx, size=batch_size, replace=False)
    return (torch.stack([token_ids[i:i+context_length] for i in sampled_start_idx]), torch.stack([token_ids[i+1:i+1+context_length] for i in sampled_start_idx]))