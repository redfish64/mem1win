import torch

import torch.nn as nn
import torch.autograd as ag
import torch.optim as o
import torch.nn.functional as f
import grad
import JacModule as jm

import pdb

class MemWrapper(nn.Module):
    def __init__(self,core_layer, squishing_layer, mem_out_layer, n_batch_size, n_mem):
        super().__init__()
        self.core_layer = core_layer
        self.squishing_layer = squishing_layer
        self.mem_out_layer = mem_out_layer
        self.n_mem = n_mem
        self.n_batch_size = n_batch_size
        self.memory = nn.Parameter(torch.full((n_batch_size,n_mem,),0.))

    def forward(self,inp,*args,**kwargs):
        assert (inp.shape[0] == self.n_batch_size)
        core_inp = self.squishing_layer(inp,self.memory)

        out = self.core_layer(core_inp,*args,**kwargs)
        self.last_run_core_layer_out = out

        return out

    def update_memory_for_next_cycle(self,mem):
        self.memory = self.mem

class SquishLinear(nn.Module):
    def __init__(self,n_in_lengths,n_out_length,**kwargs):
        super().__init__()
        self.n_in_lengths = n_in_lengths
        self.n_out_length = n_out_length

        total_in_length = sum(n_in_lengths)
        self.lin = nn.Linear(total_in_length, n_out_length)

    def forward(self, *args, **kwargs):
        ins_data_split = len(self.n_in_lengths)
        in_data = torch.cat(args[0:ins_data_split],dim=-1)
        return self.lin(in_data,*args[ins_data_split:],**kwargs)
     
#params
n_core_layers = 5
n_mem = 6
n_core_inout = 4
n_batch_size = 4

n_train_loops = 10

learning_rate = 3e-4

cycle_decay = 0.99

#first lets fake a core    
core = nn.Sequential(nn.Sequential(*(nn.Linear(n_core_inout,n_core_inout) for _ in range(n_core_layers))),
                     nn.Softmax())

#params without our memory stuff
orig_core_params = list(core.parameters())

mw_list = []

#now lets "wrap" the core
for i in range(n_core_layers):
    squishing_layer = SquishLinear((n_core_inout,n_mem), n_core_inout)
    mem_out_layer = SquishLinear((n_core_inout,n_mem),n_mem)

    #we're just linearly replacing each layer but in a real model, it could be more tricky with multiple heads, etc.
    #so the end result is we keep a mw_list which we use to do all mem related stuff without having to revisit the
    #original model
    mw = MemWrapper(core[0][i],squishing_layer, mem_out_layer, n_batch_size, n_mem)
    core[0][i]=mw
    mw_list.append(mw)

core_optim = o.SGD(orig_core_params,lr=learning_rate)


#this would be reinitialized everytime a new set of data is started (a new book, unrelated audio recording, etc.)
#along with memory
last_mem_in_to_squish_params_grad = None

#training
for i in range(n_train_loops):
    train_in = torch.rand((n_batch_size,n_core_inout))
    train_target = torch.rand((n_batch_size,n_core_inout))

    core_out = core(train_in)

    loss = f.cross_entropy(core_out,train_target)

    #mem_out to mem_in (very slow can't help it)
    #the only thing we may possibly be able to do is breakout and call the grad_fn and next_functions
    #manually
    for mw in mw_list:
        mol = mw.mem_out_layer
        
        res = jm.jac_grad(mol,mw.last_run_core_layer_out,[mol.memory,mol.weight,mol.bias])
        


        #target loss to immediate params (this runs through squishing_layer and core_layer, but not mem_out params)
        imm_loss_to_sq_mem_in_grad = ag.grad((loss,),(squishing_param_tensors+mem_in),retain_graph=True)
        imm_loss_to_squishing_param_tensors_grad = grad.grad_from_list(loss_to_sq_mem_in_grad[0:len(squishing_param_tensors)])
        loss_to_mem_in_grad = grad.grad_from_list(loss_to_sq_mem_in_grad[len(squishing_param_tensors):])

        loss_to_past_squishing_params_grad = grad.combine_grads(loss_to_mem_in_grad,last_mem_in_to_squish_params_grad)
        loss_to_squishing_params_grad = imm_loss_to_squishing_param_tensors_grad + loss_to_past_squishing_params_grad

        loss_to_mem_out_params_grad = loss_to_mem_in_grad * last_mem_in_to_mem_out_grad


        mem_to_mem_grad = torch.full((n_batch_size,n_mem,n_mem),0.)
        mem_to_wrapper_params_grad = torch.full((n_batch_size,n_mem,len(wrapper_params)),0.)

        for bi in range(n_batch_size):
            for mi in range(n_mem):
                (mem_item_to_mem_grad,*mem_item_to_wrapper_params_grad) = ag.grad((mw.last_run_mem_out[bi,mi],),[mw.memory[bi]]+wrapper_params,retain_graph=True)
                pdb.set_trace()
                mem_to_mem_grad[bi,mi] = mem_item_to_mem_grad
                mem_to_wrapper_params_grad[bi,mi] = mem_item_to_wrapper_params_grad

        pdb.set_trace()

    #train core if you want
    #note: for a pretrained model, we could train just squishing parameters here
    loss.backward()
    core_optim.step()
    core_optim.zero_grad()





