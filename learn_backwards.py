import torch

import torch.nn as nn
import torch.autograd as g



mem_input = torch.full((5,),1.0,requires_grad=True)
params = torch.full((5,),2.0,requires_grad=True)
a = mem_input - params
mem_output = a * a
out = a.sum()
