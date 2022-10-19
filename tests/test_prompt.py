import context

import torch

import src

p = src.model.NodePrompt()

a = torch.ones([1, 2])
print(a)
a.requires_grad = True
b = a.clone()
print(b.requires_grad)
a = a * 2
print(b)
print(a)
