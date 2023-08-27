import torch

import torch.nn as nn
import torch.autograd as ag
import torch.optim as o
import torch.nn.functional as f

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
        self.last_run_mem_out = self.mem_out_layer(out,self.memory)

        return out

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

#first lets fake a core    
core = nn.Sequential(nn.Sequential(*(nn.Linear(n_core_inout,n_core_inout) for _ in range(n_core_layers))),
                     nn.Softmax())

#params without our memory stuff
orig_core_params = list(core.parameters())

mw_list = []

#HACK: we are always getting "RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior." and I don't know why. it's really odd.
new_core = nn.Sequential()
new_core_sub_layer = nn.Sequential()
new_core.add_module("main_body",new_core_sub_layer)
new_core.add_module("softymax",core[1])

#now lets "wrap" the core
for i in range(n_core_layers):
    squishing_layer = SquishLinear((n_core_inout,n_mem), n_core_inout)
    mem_out_layer = SquishLinear((n_core_inout,n_mem),n_mem)

    #we're just linearly replacing each layer but in a real model, it could be more tricky with multiple heads, etc.
    #so the end result is we keep a mw_list which we use to do all mem related stuff without having to revisit the
    #original model
    mw = MemWrapper(core[0][i],squishing_layer, mem_out_layer, n_batch_size, n_mem)
    new_core_sub_layer.add_module(f'{i=}',mw)
    mw_list.append(mw)

core_optim = o.SGD(orig_core_params,lr=learning_rate)


#this would be reinitialized everytime a new set of data is started (a new book, unrelated audio recording, etc.)
#along with memory
#last_wrapper_params_to_mem_in = torch.zeros_like(wrapper_params)

#training
for i in range(n_train_loops):
    train_in = torch.rand((n_batch_size,n_core_inout))
    train_target = torch.rand((n_batch_size,n_core_inout))

    core_out = new_core(train_in)

    loss = f.cross_entropy(core_out,train_target)

    with torch.no_grad():
        #mem_out to mem_in (very slow can't help it)
        #the only thing we may possibly be able to do is breakout and call the grad_fn and next_functions
        #manually
        for mw in mw_list:
            squishing_param_tensors = [t for param in mw.squishing_layer.parameters() for t in param]
            mem_out_layer_param_tensors = [t for param in mw.mem_out_layer.parameters() for t in param]
            wrapper_params = squishing_param_tensors + mem_out_layer_param_tensors

            mem_in = list(mw.memory for mw in mw_list)

            pdb.set_trace()
            #target loss to immediate params (this runs through squishing_layer and core_layer, but not mem_out params)
            (loss_to_squishing_params_grad, *loss_to_mem_in_grad) = ag.grad((loss,),squishing_param_tensors,retain_graph=True)
            pdb.set_trace()
            #orig: (loss_to_squishing_params_grad, *loss_to_mem_in_grad) = ag.grad((loss,),(squishing_param_tensors+mem_in),retain_graph=True)

            mem_to_mem_grad = torch.full((n_batch_size,n_mem,n_mem),0.)
            mem_to_wrapper_params_grad = torch.full((n_batch_size,n_mem,wrapper_params.shape[0]),0.)

            for bi in range(n_batch_size):
                for mi in range(n_mem):
                    (mem_item_to_mem_grad,mem_item_to_wrapper_params_grad) = ag.grad((mw.last_run_mem_out[bi,mi],),(mw.memory,wrapper_params),retain_graph=True)
                    mem_to_mem_grad[bi,mi] = mem_item_to_mem_grad
                    mem_to_wrapper_params_grad[bi,mi] = mem_item_to_wrapper_params_grad

        
        #train core if you want
        loss.backward()
        core_optim.step()
        core_optim.zero_grad()





