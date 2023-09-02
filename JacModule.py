#modules that can work with jacrev. The only difference is that it's parameters are passed into it during the forward call.

import torch
import torch.nn as nn
import torch.func as f
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pdb
@dataclass
class _JacContext:
    params: list
    jc_to_param_locs : dict

_jac_context = None

class JacModule(ABC,nn.Module):
    @abstractmethod
    def forward_with_params(self,inp,*args,**kwargs):
        """
        Like nn.Module.forward except for the following:
        1. parameters returned by parameters(recurse=False) (or equivalent objects that wrap the tensors
           returned by parameters(recurse=False)) are passed as the first arguments. It may be obvious,
           but these parameters MUST be used instead of internal parameters
        2. whole batches are not included in inp, but the individual items in the batch. This faciliates the use
           of jacobian, so that we don't get a jacobian showing grads from item x in batch against the results of item y
           in batch (which would always be zero), and would be really big depending on the batch size.
        """
        pass

    def forward(self,inp,*args,**kwargs):
        if(_jac_context is not None):
            param_locs = _jac_context.jc_to_param_locs[self]
            if(param_locs is not None):
                params = _jac_context.params[param_locs[0]:param_locs[1]]

        if(params is None):
            params = self.parameters(recurse=False)

        return self.forward_with_params(inp,*(tuple(params)+args),**kwargs)
    
def jac_grad(module,inp_batch,selected_grad_jac_modules,*extra_module_args,fake_one_row_batch=False,**extra_module_kwargs):
    """
    Runs the module against the input batch and returns the gradiant and output of selected jacmodules
    module - the module to run. Doesn't have to be a JacModule
    inp_batch - a batch of input values as a tensor in the shape (B,...) where B is the number of items in the batch
    selected_grad_jac_modules - JacModule's in this list will have the gradiant computed against their parameters
    fake_one_row_batch - each call is given a single input value in the batch individually (looped over with vmap)
                         Sometimes modules work better whan there is a batch dimension, so if this is true,
                         the input will be unsqueezed in dim zero before being passed.
    """

    jc_to_plocs = {}
    inp_params = []
    for jc in selected_grad_jac_modules:
        curr_params = list(jc.parameters(recurse=False))
        start_index = len(inp_params)
        end_index = start_index + len(curr_params)

        jc_to_plocs[jc]=(start_index,end_index)
        inp_params.append(curr_params)
        
    def model_func(input_item,params):
        global _jac_context
        _jac_context = _JacContext(params,jc_to_plocs)

        output = module(input_item.unsqueeze(0) if fake_one_row_batch else input_item,
                        *extra_module_args,**extra_module_kwargs)
        _jac_context = None

        #we specify output twice, because the first output value will be replaced by the grads, and the second left as is
        #(this is due to has_aux=True below)
        return output,output

    jacrev_argnums = tuple(list(range(1,len(inp_params)+1)))
    vmap_in_dims = tuple([0]+[None]*len(inp_params))

    return f.vmap(f.jacrev(model_func,argnums=jacrev_argnums,has_aux=True),in_dims=vmap_in_dims)(inp_batch,*inp_params)

class JacLinear(JacModule):
    def __init__(self,n_in,n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.weight = nn.Parameter(torch.randn((n_out,n_in)) * 0.01)
        self.bias = nn.Parameter(torch.zeros((n_out,)) * 0.01)

    def forward_with_params(self,inp,weight,bias): 
        return nn.functional.linear(inp,weight,bias)

def test1():
    torch.manual_seed(42)
    jl = JacLinear(3,2)
    ib = torch.rand((5,3))
    
    r = jac_grad(jl,ib,[jl])
    print(f'{r=}')
    
