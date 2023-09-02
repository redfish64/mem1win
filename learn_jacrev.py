import torch

import torch.nn as nn
import torch.nn.functional as nf
import torch.func as f

import pdb

def test1():
    torch.manual_seed(42)
    params = torch.rand(8)
    x = torch.randn(3)

    def model_func(params_vector):
        weight = params_vector[:6].reshape(2, 3)
        bias = params_vector[6:]

        output = nn.functional.linear(x,weight,bias)

        return output

    grad = f.jacrev(model_func)(params)

    print(grad)

#test with jacrev call also producing output
def test2():
    torch.manual_seed(42)
    params = torch.rand(8)
    x = torch.randn(3)

    def model_func(params_vector):
        weight = params_vector[:6].reshape(2, 3)
        bias = params_vector[6:]

        output = nn.functional.linear(x,weight,bias)

        return output,output

    grad = f.jacrev(model_func,has_aux=True)(params)

    print(grad)
 
#test moving input (x) into the call
def test3():
    torch.manual_seed(42)
    params = torch.rand(8)
    x = torch.randn(3)

    def model_func(x,params_vector):
        weight = params_vector[:6].reshape(2, 3)
        bias = params_vector[6:]

        output = nn.functional.linear(x,weight,bias)

        return output,output

    grad = f.jacrev(model_func,argnums=1,has_aux=True)(x,params)

    print(grad)
 
#test with batches. note that first result matches test3()
def test4():
    torch.manual_seed(42)
    params = torch.rand(8)
    x = torch.randn(5,3)
    
    def model_func(x,params_vector):
        weight = params_vector[:6].reshape(2, 3)
        bias = params_vector[6:]

        output = nn.functional.linear(x,weight,bias)

        return output,output

    grad = f.vmap(f.jacrev(model_func,argnums=1,has_aux=True),in_dims=(0,None))(x,params)

    print(grad)
 
#test with multiple parameter tensors (in batch and all the rest)
#
#note that the output looks a little strange compared to the tests above, but that's
#because weight is now 2x3 and bias is independent rather than a flat 8 params
def test5():
    torch.manual_seed(42)
    weight = torch.rand(2,3)
    bias = torch.rand(2)
    x = torch.randn(5,3)
    
    def model_func(x,weight_in,bias_in):
        #pdb.set_trace()
        output = nn.functional.linear(x,weight_in,bias_in)

        return output,output

    grad = f.vmap(f.jacrev(model_func,argnums=(1,2),has_aux=True),in_dims=(0,None,None))(x,weight,bias)
    print(grad)

    return grad
 
