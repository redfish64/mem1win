import torch

import torch.nn as nn
import torch.autograd as ag
import torch.optim as o
import torch.nn.functional as f
import pdb

def test1():
    #x = torch.full((5,),2.,requires_grad=True)
    x = torch.arange(5,dtype=float,requires_grad=True)
    y = torch.full((5,),3.,requires_grad=True)

    a = x + y
    b = a * a
    c = x + 3
    d = x * 2
    e = d + y


def test2():
    torch.manual_seed(6)
    x = torch.randn(4, 4, requires_grad=True)
    y = torch.randn(4, 4, requires_grad=True)
    z = x * y
    l = z.sum()
    dl = torch.tensor(1.)
    back_sum = l.grad_fn
    dz = back_sum(dl)
    back_mul = back_sum.next_functions[0][0]
    dx, dy = back_mul(dz)
    back_x = back_mul.next_functions[0][0]
    back_x(dx)
    back_y = back_mul.next_functions[1][0]
    back_y(dy)
    print(x.grad)
    print(y.grad)
    pdb.set_trace()
    
def test3():
    torch.manual_seed(6)
    x = torch.randn(4, requires_grad=True)
    y = torch.randn(4, requires_grad=True)
    z = torch.cat(((x[0] * y[0]).unsqueeze(dim=0),(x[0] + y[0]).unsqueeze(dim=0)))
    l = z.sum()
    dl = torch.tensor(1.)
    back_sum = l.grad_fn
    dz = back_sum(dl)
    back_mul = back_sum.next_functions[0][0]
    dx, dy = back_mul(dz)
    back_x = back_mul.next_functions[0][0]
    back_x(dx)
    back_y = back_mul.next_functions[1][0]
    back_y(dy)
    print(x.grad)
    print(y.grad)
    
    pdb.set_trace()
