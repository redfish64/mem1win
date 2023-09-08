#from https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
_ = torch.manual_seed(0)

def predict(weight, bias, x):
    return F.linear(x, weight, bias).tanh()

D = 16
weight = torch.randn(D, D)
bias = torch.randn(D)
x = torch.randn(D) # feature vector

def compute_jac(xp):
    jacobian_rows = [torch.autograd.grad(predict(weight, bias, xp), xp, vec)[0]
                     for vec in unit_vectors]
    return torch.stack(jacobian_rows)

xp = x.clone().requires_grad_()
unit_vectors = torch.eye(D)

jacobian = compute_jac(xp)

print(jacobian.shape)
print(jacobian[0])  # show first row
