import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable] = None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
            
                state = self.state[p]

                if len(state) == 0: 
                    state['step'] = 1
                    state['m'] = torch.zeros_like(p.data)
                    print(state['m'])
                    state['v'] = torch.zeros_like(p.data)
                
                m = state['m']
                v = state['v']
     
                b1, b2 = group['betas']

                m = torch.mul(b1, m) + torch.mul(1 - b1, grad)
                v = torch.mul(b2, v) + torch.mul(1 - b2, torch.mul(grad, grad))

            
                state['m'] = m
                state['v'] = v


                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = math.sqrt(1 - b2 ** state['step'])
                corrected_lr = group['lr'] * (bias_correction2 / bias_correction1)

               
                denom = torch.sqrt(v) + group['eps']


                p.data = p.data - (corrected_lr * (m/denom))
                p.data = p.data - (group['lr']*group['weight_decay']*p.data)
                

                state['step'] += 1

        


        return loss


                


