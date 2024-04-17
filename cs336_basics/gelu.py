import torch 

def gelu(x): 
    return (x * 0.5)*(1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))