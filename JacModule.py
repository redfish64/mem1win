#modules that can work with jacrev. The only difference is that it's parameters are passed into it during the forward call.

import torch
import torch.nn as nn
import torch.func as f
from dataclasses import dataclass
import pdb
import util as u

class JacParameter(nn.Parameter):
    """
    A JacParameter allows for a parameter to be used as a `grad_jac_params` argument to jac_grad.
    When used, it must be wrapped in a call as follows: get_jac_param(<param>)
    """
    def __new__(cls,data,requires_grad,is_batch_indexed=False):
        """
        is_batched_indexed - if true, then each parameter is expected to have its outermost dimension the same size as
                             the number of items per batch. If false, each JacParameter object will be repeated for each
                             batch item. 
        """
        instance = torch.nn.parameter.Parameter._make_subclass(cls, data, requires_grad)
        instance.is_batch_indexed = is_batch_indexed
        instance.jac_grad = None
        instance.jac_param = None
        instance.running_jac_grad = None
        instance.requires_grad = requires_grad
        return instance

    def calc_grad_from_running_jac_grad(self,jac_out_to_loss_grad,reset_grad=False):
        """
        Figures out the grad for the given jacparameter given the conversion from the jac_grad output
        to the final loss output.

        jac_out_to_loss_out - the gradiant from the loss to the output of whatever this jac parameter is being calculated to
        """
        if(self.running_jac_grad is None):
            self.running_jac_grad = torch.zeros_like(self.jac_grad,dtype=self.data.dtype)

        #number of dimensions in the memory output
        n_mem_dims = jac_out_to_loss_grad.dim() 
        #number of dimensions of the parameter
        n_dims = self.data.dim() - (1 if self.is_batch_indexed else 0)

        #in jac_grad, the memory dims appear first (are on the outside) and the jac parameter dims are on the inside
        #we need to expand out the jac_out_to_loss which is in the shape of the jac parameter
        #to the jac_grad, which has the memory dims on the outside
        dimmed_loss_grad = jac_out_to_loss_grad.view(list(jac_out_to_loss_grad.size()) + [1] * n_dims)

        res = dimmed_loss_grad * self.running_jac_grad

        res = torch.sum(res, list(range(1 if self.is_batch_indexed else 0,n_mem_dims)))

        return res

    def update_running_jac_grad(self,decay, cycle_out_to_cycle_out):
        """
        This updates the jac gradiant for a cycle through the model. It takes the existing running gradiant and averages
        it with the current gradiant. This way we get an estimation of how a parameter affects the output of
        the current round from all prior rounds.

        decay - how much to decay the previous old gradiants when updating, a value from 0 to 1.
        cycle_out_to_cycle_out - the gradiant from how the output of the last cycle affects the current one
        """

        res = u.cross_product_batch_ldim_tensors(cycle_out_to_cycle_out,3,self.running_jac_grad,3)

        self.running_jac_grad = res * (1. - decay) + self.jac_grad * decay
        
def jac_grad(module,inp_batch,grad_jac_params,*extra_module_args,fake_one_row_batch=False,vmap_randomness='error',**extra_module_kwargs):
    """
    Runs the module against the input batch and returns the output. The jacobian is placed into the individual
    selected parameters, as `<param>.jac_grad`
    module - the module or function to run.
    inp_batch - a batch of input values as a tensor in the shape (B,...) where B is the number of items in the batch
    grad_jac_params - the jacobian will be calculated against the given JacParameter objects. The results
                      will appear in `<jac_param>.jac_grad`
    fake_one_row_batch - each call is given a single input value in the batch individually (looped over with vmap)
                         Sometimes modules work better whan there is a batch dimension, so if this is true,
                         the input will be unsqueezed in dim zero before being passed.
    vmap_randomness    - like the vmap randomness parameter, man
    """


    repeated_grad_jac_params = []
    batch_indexed_grad_jac_params = []

    for p in grad_jac_params:
        assert isinstance(p,JacParameter),f"All parameters passed as `grad_jac_params` must be instances of `JacParameter`, got {p}"
        if(p.is_batch_indexed):
            batch_indexed_grad_jac_params.append(p)
        else:
            repeated_grad_jac_params.append(p)
            

    #note that we recreate grad_jac_params, but ordered so that all batch_indexed_grad_jac_params are first. This
    #way when we call vmap below, we can more easily refer to them
    all_grad_jac_params = batch_indexed_grad_jac_params + repeated_grad_jac_params

    def model_func(input_item,*params):
        for jp,p in zip(params,all_grad_jac_params):
            if(p.is_batch_indexed):
                p.jac_param = jp.unsqueeze(0)
            else:
                p.jac_param = jp

        if fake_one_row_batch:
            output = module(input_item.unsqueeze(0)).squeeze(0)
        else:
            output = module(input_item.unsqueeze(0))

        for p in all_grad_jac_params:
            p.jac_param = None

        #we specify output twice, because the first output value will be replaced by the grads, and the second left as is
        #(this is due to has_aux=True below)
        return output,output

    jacrev_argnums = tuple(list(range(1,len(batch_indexed_grad_jac_params)+len(repeated_grad_jac_params)+1)))
    vmap_in_dims = tuple([0]+[0]*len(batch_indexed_grad_jac_params)+[None]*len(repeated_grad_jac_params))
    

    (grads,res) = f.vmap(f.jacrev(model_func,argnums=jacrev_argnums,has_aux=True),in_dims=vmap_in_dims,randomness=vmap_randomness)(inp_batch,*batch_indexed_grad_jac_params,*repeated_grad_jac_params)

    for g,p in zip(grads,all_grad_jac_params):
        p.jac_grad = g

    return res

def get_jac_param(jp):
    return jp.jac_param if jp.jac_param is not None else jp

class JacLinear(nn.Module):
    def __init__(self,n_in,n_out,requires_grad,has_bias=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.weight = JacParameter(torch.randn((n_out,n_in)) * 0.01,requires_grad)
        self.bias = JacParameter(torch.zeros((n_out,)) * 0.01,requires_grad) if has_bias else None

    def get_jac_params(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])
    def forward(self,inp):
        return nn.functional.linear(inp,get_jac_param(self.weight),get_jac_param(self.bias) if self.bias is not None else None)

#concatenates two inputs into one and passes to a linear
class SquishJacLinear(nn.Module):
    def __init__(self,in_lengths,out_length,requires_grad,**kwargs):
        super().__init__()
        self.in_lengths = in_lengths
        self.out_length = out_length

        total_in_length = sum(in_lengths)

        self.weight = JacParameter(torch.randn((out_length,total_in_length)) * 0.01,requires_grad)
        self.bias = JacParameter(torch.zeros((out_length,)),requires_grad)

    def forward(self, *args):
        assert len(args) == len(self.in_lengths), f'wrong number of args, got {len(args)} args, want {len(self.in_lengths)}'
        print(f'HACK {args=}')
        in_data = torch.cat(args,dim=-1)
        return nn.functional.linear(in_data,get_jac_param(self.weight),get_jac_param(self.bias))
     
def test1():
    torch.manual_seed(42)
    jl = JacLinear(3,2)
    ib = torch.rand((5,3))
    
    r = jac_grad(jl,ib,jl.parameters())
    print(f'{r=}')
    print(f'pjg={[p.jac_grad for p in jl.parameters()]}')
    pdb.set_trace()
    
def test2():
    torch.manual_seed(42)
    jl = JacLinear(3,2)
    ib = torch.rand((5,3))
    
    r = jac_grad(jl,ib,[jl.bias])
    print(f'{r=}')
    print(f'{jl.bias.jac_grad=}')
    pdb.set_trace()
    
# def test2():
#     torch.manual_seed(42)
#     jl = SquishJacLinear(3,2)
#     ib = torch.rand((5,3))
#     mem = torch.rand((5,1))
    
#     r = jac_grad(jl,ib,
#     print(f'{r=}')
    
