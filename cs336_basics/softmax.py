import torch 

def softmax(v, dimension):
    largest_val = torch.max(v, dim=dimension, keepdim=True).values
    v = v - largest_val
    exp_v = torch.exp(v)
    sum_v = torch.sum(exp_v, dimension, keepdim=True)
    return exp_v / sum_v




