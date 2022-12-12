import torch
from torch.optim import Adam
import torch.nn as nn


a = nn.Parameter(torch.ones((1, 1)))
print(a.data.requires_grad)
# optimizer = Adam(list(a))
# optimizer.zero_grad()
b = a.repeat(3, 1)
print(a.requires_grad, b.requires_grad)
# b.retain_grad()

b = torch.ones(3, 1).requires_grad_() # shape = [3, 1]
y = torch.arange(1,4).reshape(3, 1) # shape = [3, 1]
loss = ((y - b) ** 2).mean()
loss.backward()
# optimizer.step()
print(b.grad)
print(a.grad)
