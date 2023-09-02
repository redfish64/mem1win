import torch

import torch.nn as nn

def _mult_out_shapes(shapes):
    total = 1
    for s in shapes:
        for v in s:
            total *= v

    return total
        
class Grad():
    def __init__(self,n_outputs,input_shapes,grads):
        """
        n_outputs - number of individual outputs
        input_shapes - shape of each input
        grads - 2d tensor of n_outputs x total length of all inputs
        """
        self.n_outputs = n_outputs
        self.input_shapes = input_shapes
        self.grads = grads

    def apply(self,learning_rate,params):
        """
        does simple straight gradiant descent learning. output must be 1
        and represents the loss.
        
        TODO 2 use an arbritrary optimizer
        params - list of tensors or a tensor of params to modify. The tensors
                 must be in the shape of input_shapes
        """
        if not isinstance(params, list):
            params = [params]

        params += self.grads * learning_rate * -1.

def grad_from_list(grads):
    """
    Creates a single output gradiant from a list of tensors containing gradiants
    grads - single tensor or list of tensors
    """
    if not isinstance(grads, list):
        grads = [grads]
    grad_shapes = [g.shape for g in grads]

    grads_tensor = torch.cat(grads.view(-1))
    return Grad(1, grad_shapes,grads_tensor)

def grad_zeros(n_outputs, input_shapes):
    """
    initializes a Grad object of zeros with a given number of outputs and
    a list of input lengths. It can then be populated individually (TODO 2 do something to speed that up if can be)
    n_outputs - number of outputs
    input_lengths - lengths of each input tensor.
    """
    g = Grad(n_outputs,input_shapes,torch.zeros((n_outputs,_mult_out_shapes(input_shapes))))
    return g

