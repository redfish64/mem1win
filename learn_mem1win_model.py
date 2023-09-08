import torch

import torch.nn as nn
import torch.autograd as ag
import torch.optim as o
import torch.nn.functional as f
import grad
import JacModule as jm
from importlib import reload
reload(jm) #hack to reload JacModule as I debug it

import pdb

class MemWrapper(nn.Module):
    def __init__(self,core_layer, mem_in_layer, mem_out_layer, n_batch_size, memory):
        super().__init__()
        self.core_layer = core_layer
        self.mem_in_layer = mem_in_layer
        self.mem_out_layer = mem_out_layer
        self.n_mem = n_mem
        self.n_batch_size = n_batch_size
        self.memory = memory

    def forward(self,inp,*args,**kwargs):
        assert (inp.shape[0] == self.n_batch_size)
        core_inp = self.mem_in_layer(inp)

        out = self.core_layer(core_inp,*args,**kwargs)
        self.last_run_core_layer_out = out

        return out

    def update_memory_for_next_cycle(self,mem):
        self.memory = self.mem

#params
n_core_layers = 5
n_mem = 6
n_core_inout = 4
n_batch_size = 4

n_train_loops = 10

learning_rate = 3e-4

jac_grad_decay = 0.99

#first lets fake a core    
core = nn.Sequential(nn.Sequential(*(nn.Linear(n_core_inout,n_core_inout) for _ in range(n_core_layers))),
                     nn.Softmax())

mw_list = []

#now lets "wrap" the core
for i in range(n_core_layers):
    memory = nn.Parameter(torch.full((n_batch_size,n_mem),0.))
    mem_in_layer = jm.MemJacLinear(memory,n_core_inout,n_core_inout) # this layer doesn't need to be jac compatible
    # but we need to include memory in the input
    mem_out_layer = jm.MemJacLinear(memory,n_core_inout,n_mem)

    memory.running_jac_grad = torch.zeros((n_batch_size,n_mem,n_mem))
    mem_out_layer.weight.running_jac_grad = torch.zeros((n_batch_size,n_mem,n_mem,n_mem+n_core_inout))
    mem_out_layer.bias.running_jac_grad = torch.zeros((n_batch_size,n_mem,n_mem))

    #we're just linearly replacing each layer but in a real model, it could be more tricky with multiple heads, etc.
    #so the end result is we keep a mw_list which we use to do all mem related stuff without having to revisit the
    #original model
    mw = MemWrapper(core[0][i],mem_in_layer, mem_out_layer, n_batch_size, memory)
    core[0][i]=mw
    mw_list.append(mw)

#this will train everything at once, the wrapper and the main model. We could train just the mem wrapper for pretrained
#models, it's up to you.
core_optim = o.SGD(core.parameters(),lr=learning_rate)

#training
for i in range(n_train_loops):
    train_in = torch.rand((n_batch_size,n_core_inout)) #HACK
    train_target = torch.rand((n_batch_size,n_core_inout)) #HACK

    core_out = core(train_in)

    loss = f.cross_entropy(core_out,train_target)
    loss.backward()

    for mw in mw_list:
        mol = mw.mem_out_layer

        with torch.no_grad():
            #TODO 2.5 here we are ignoring the model when computing the gradiant and directly
            #         using mw.last_run_core_layer_out, since it would slow things down a lot
            #         if we recomputed the model.  we probably want to try recomputing the model
            #         if things don't work well.
            #note, in the current state, the grad for mol.bias is always 1. We could hardcode this, but
            #if we change code creating the memwrapper model maybe that will change.
            res = jm.jac_grad(mol,mw.last_run_core_layer_out,repeated_grad_jac_params=[mol.weight,mol.bias],batch_indexed_grad_jac_params=[mw.memory])
        
            #update the current weight and bias based on the setting the memory in the past
            mol.weight.grad = (mw.memory.grad.unsqueeze(-1).unsqueeze(-1) * mol.weight.running_jac_grad).sum(dim=0).sum(dim=0)
            mol.bias.grad = (mw.memory.grad.unsqueeze(-1) * mol.bias.running_jac_grad).sum(dim=0).sum(dim=0)

            #clear out grad for the next training loop (jac grad doesn't need this)
            mw.memory.grad = None

            #update the running jac grad's for the next training loop
            mol.weight.running_jac_grad = mol.weight.running_jac_grad * mw.memory.jac_grad.unsqueeze(-1) * jac_grad_decay + mol.weight.jac_grad
            mol.bias.running_jac_grad = mol.bias.running_jac_grad * mw.memory.jac_grad * jac_grad_decay + mol.bias.jac_grad
            mw.memory.running_jac_grad = mw.memory.running_jac_grad * mw.memory.jac_grad * jac_grad_decay + mw.memory.jac_grad

            

    #train core if you want
    #note: for a pretrained model, we could train just squishing parameters here
    core_optim.step()
    core_optim.zero_grad()

print("one loop done")



