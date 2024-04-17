import torch
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss




# Assume weights and SGD class definition from the previous example are available

learning_rates = [1e1, 1e2, 1e3]
losses = []

for lr in learning_rates:
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    optimizer = SGD([weights], lr=lr)
    lr_losses = []

    for _ in range(10):  # Run for 10 iterations
        optimizer.zero_grad()
        loss = (weights**2).mean()  # A simple loss for demonstration
        loss.backward()
        optimizer.step()
        lr_losses.append(loss.item())

    losses.append(lr_losses)

# Now `losses` holds the loss values for each learning rate across iterations

# Print or log the loss behavior for each learning rate
for i, lr in enumerate(learning_rates):
    print(f"Learning rate: {lr}, Losses: {losses[i]}")
    # Based on the observed losses, you can write a one-two sentence response
