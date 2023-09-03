#modules that can work with jacrev. The only difference is that it's parameters are passed into it during the forward call.

import torch
import torch.nn as nn
import torch.func as f
from dataclasses import dataclass
import pdb
@dataclass
class _JacContext:
    params: list
    param_to_array_index : dict
    
_jac_context = None

class JacModule(nn.Module):
    def get_jac_param(self,param):
        if(_jac_context is None):
            return param

        param_loc = _jac_context.param_to_array_index[id(param)]
        if(param_loc is None):
            return param

        return _jac_context.params[param_loc]

def jac_grad(module,inp_batch,*extra_module_args,selected_grad_jac_params=None,fake_one_row_batch=False,**extra_module_kwargs):
    """
    Runs the module against the input batch and returns the output. The jacobian is placed into the individual
    selected parameters
    module - the module to run. Doesn't have to be a JacModule
    inp_batch - a batch of input values as a tensor in the shape (B,...) where B is the number of items in the batch
    selected_grad_jac_params - if not none, the jacobian will only be calculated against the given parameters.
                               Otherwise it will be calculated against all parameters
    fake_one_row_batch - each call is given a single input value in the batch individually (looped over with vmap)
                         Sometimes modules work better whan there is a batch dimension, so if this is true,
                         the input will be unsqueezed in dim zero before being passed.
    """

    param_to_array_index = {}

    if(selected_grad_jac_params is None):
        selected_grad_jac_params = list(module.parameters()) # a list because we loop through them twice

    for i,p in enumerate(selected_grad_jac_params):
        param_to_array_index[id(p)]=i

    def model_func(input_item,*params):
        global _jac_context
        _jac_context = _JacContext(params,param_to_array_index)

        output = module(input_item.unsqueeze(0) if fake_one_row_batch else input_item,
                        *extra_module_args,**extra_module_kwargs)
        _jac_context = None

        #we specify output twice, because the first output value will be replaced by the grads, and the second left as is
        #(this is due to has_aux=True below)
        return output,output

    jacrev_argnums = tuple(list(range(1,len(selected_grad_jac_params)+1)))
    vmap_in_dims = tuple([0]+[None]*len(selected_grad_jac_params))

    (grads,res) = f.vmap(f.jacrev(model_func,argnums=jacrev_argnums,has_aux=True),in_dims=vmap_in_dims)(inp_batch,*selected_grad_jac_params)

    for g,p in zip(grads,selected_grad_jac_params):
        p.jac_grad = g

    return res



class JacLinear(JacModule):
    def __init__(self,n_in,n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.weight = nn.Parameter(torch.randn((n_out,n_in)) * 0.01)
        self.bias = nn.Parameter(torch.zeros((n_out,)) * 0.01)

    def forward(self,inp):
        return nn.functional.linear(inp,self.get_jac_param(self.weight),self.get_jac_param(self.bias))

def test1():
    torch.manual_seed(42)
    jl = JacLinear(3,2)
    ib = torch.rand((5,3))
    
    r = jac_grad(jl,ib)
    print(f'{r=}')
    
