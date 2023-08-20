import torch
import torch.nn as nn

from mem1w import M1Wrapper, JacobianModel

class FullM1WWrapper(nn.Module):

    def __init__(self,full_model,n_inputs, n_memory,optimizer, n_batch_size, decay,paths):
        self.full_model = full_model
        self.n_inputs = n_inputs
        self.n_memory = n_memory
        self.optimizer = optimizer
        self.n_batch_size = n_batch_size
        
        for path in paths:
            
            lin_inputs = n_inputs * n_memory
            def make_nn_fn(wm_out):
                def nn_fn(params,mem_in):
                    weight = params.view[:lin_inputs]
                    bias = params.view[lin_inputs:]
                    output = nn.functional.linear(torch.cat((mem_in,wm_out)),weight,bias)

            params = torch.cat((torch.ones(lin_inputs),torch.zeros(n_memory)))
            mem_out_jm = JacobianModel(params,optimizer,make_nn_fn)

            mw = M1Wrapper(model,mem_out_jm,n_memory, n_batch_size, decay)
