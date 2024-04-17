import torch
import torch.nn as nn
from gelu import gelu


class my_ffn(nn.Module):
    def __init__(self, d_model, d_ff):
        super(my_ffn, self).__init__()
        self.w1 = nn.Parameter(torch.randn(d_ff, d_model))
        self.w2 = nn.Parameter(torch.randn(d_model, d_ff))

    def forward(self, x): 
        x_w1 = torch.matmul(x, self.w1.T)
        gelu_x = gelu(x_w1)
        return torch.matmul(gelu_x, self.w2.T)


        




        
