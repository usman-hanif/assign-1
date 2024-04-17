import torch

def gradient_clipping(params, max_norm, eps=1e-6):
    for p in params:
        grad_norm = torch.norm(p.grad.data)
        if grad_norm > max_norm:
            p.grad.data = p.grad.data / (grad_norm + eps) * max_norm
