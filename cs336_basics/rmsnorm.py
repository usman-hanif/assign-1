import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-5):
        super(RMSNorm, self).__init__()
        self.epsilon = epsilon
        self.g = nn.Parameter(torch.rand(d_model,))

    def forward(self, a): 
        rms = torch.sqrt(torch.mean(a**2, dim=-1, keepdim=True) + self.epsilon)
        return a / rms * self.g




