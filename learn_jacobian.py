import torch
from torch import nn
from torch.autograd.functional import jacobian,vjp
import pdb

#test using _scaled_dot_product_attention whatever that is (no documentation)
def test1():
    channel_size = 8
    win_size = 16
    batch_size = 6
    wc_size = win_size * channel_size

    x = torch.randn(batch_size,win_size, channel_size)
    params = torch.rand(channel_size * win_size * 2)
    params.requires_grad=True

    def model_func(params_vector):
        query = params_vector[:wc_size].reshape(win_size, channel_size)
        key = params_vector[wc_size:].reshape(win_size, channel_size)

        #output = torch.matmul(x, weight.t()) + bias
        output = nn.functional._scaled_dot_product_attention(query,key,x)

        return output

    jacobian_matrix = jacobian(model_func, params)

    print(jacobian_matrix)

#test with the function being nn.functional.linear
def test2():
    x = torch.randn(1, 3)
    params = torch.rand(8)
    params.requires_grad=True

    def model_func(params_vector):
        weight = params_vector[:6].reshape(2, 3)
        bias = params_vector[6:]

        #output = torch.matmul(x, weight.t()) + bias
        output = nn.functional.linear(x,weight,bias)

        return output

    jacobian_matrix = jacobian(model_func, params)

    print(jacobian_matrix)

#train with the jacobian parameters to make sure they are actually gradiants as we understand them
def test3():
    x = torch.randn(1, 3)
    params = torch.rand(8)
    params.requires_grad=True

    def model_func(params_vector):
        weight = params_vector[:6].reshape(2, 3)
        bias = params_vector[6:]

        #output = torch.matmul(x, weight.t()) + bias
        output = nn.functional.linear(x,weight,bias)

        return output

    optimizer = torch.optim.AdamW([params], lr=3e-4)

    with torch.no_grad():
        for i in range(10000):
            x = torch.randn(8, 3)
            jacobian_matrix = jacobian(model_func, params)
            out = model_func(params)
            if(i%1000 == 0):
                #print(jacobian_matrix)
                print(f'output: {out}')
            jacobian_matrix[0,0] *= -1.
            loss = ((out < 0.42) * -2. + 1).unsqueeze(dim=2) * jacobian_matrix
            params.grad = loss.sum(dim=0).sum(dim=0)


            optimizer.step()



#test with the function being nn.functional.linear
def test4():
    mem_size = 6
    wm_out_size = 5
    in_size = mem_size + wm_out_size
    weights_size = in_size * mem_size
    bias_size = mem_size
    
    wm_out = torch.randn(wm_out_size)
    mem_in = torch.randn(mem_size)
    params = torch.rand(weights_size + bias_size)
    pdb.set_trace()
    
    def model_func(params,mem_in):
        weight = params[:weights_size].reshape(mem_size, in_size)
        bias = params[weights_size:]

        #pdb.set_trace()
        output = nn.functional.linear(torch.cat((mem_in,wm_out)),weight,bias)

        return output

    jacobian_matrix = jacobian(model_func, (params,mem_in))

    print(jacobian_matrix)
    print(jacobian_matrix[0].shape)
    print(jacobian_matrix[1].shape)

#test with a more convienent params (weights and biases separated and not flattened)
def test5():
    mem_size = 6
    wm_out_size = 5
    in_size = mem_size + wm_out_size
    
    wm_out = torch.randn(wm_out_size)
    mem_in = torch.randn(mem_size)
    weight = torch.rand(mem_size,in_size)
    bias = torch.rand(mem_size)
    
    def model_func(weight,bias,mem_in):

        #pdb.set_trace()
        output = nn.functional.linear(torch.cat((mem_in,wm_out)),weight,bias)

        return output

    jacobian_matrix = jacobian(model_func, (weight,bias,mem_in))

    print(jacobian_matrix)
    print(jacobian_matrix[0].shape)
    print(jacobian_matrix[1].shape)
    print(jacobian_matrix[2].shape)

#test with batches
def test6():
    items_per_batch = 3
    mem_size = 6
    wm_out_size = 5
    in_size = mem_size + wm_out_size
    
    wm_out = torch.randn(items_per_batch,wm_out_size)
    mem_in = torch.randn(items_per_batch,mem_size)
    weight = torch.rand(mem_size,in_size)
    bias = torch.rand(mem_size)
    
    def model_func(weight,bias,mem_in):

        #pdb.set_trace()
        output = nn.functional.linear(torch.cat((mem_in,wm_out),dim=1),weight,bias)

        return output

    jacobian_matrix = jacobian(model_func, (weight,bias,mem_in))

    print(jacobian_matrix)
    print(jacobian_matrix[0].shape)
    print(jacobian_matrix[1].shape)
    print(jacobian_matrix[2].shape)

def test7():
    def exp_reducer(x):
        return x.exp().sum(dim=1)
    inputs = torch.rand(4, 4)
    v = torch.ones(4)
    return vjp(exp_reducer, inputs, v)

def test8():
    items_per_batch = 3
    mem_size = 6
    wm_out_size = 5
    in_size = mem_size + wm_out_size
    
    wm_out = torch.randn(items_per_batch,wm_out_size)
    mem_in = torch.randn(items_per_batch,mem_size)
    weight = torch.rand(mem_size,in_size)
    bias = torch.rand(mem_size)
    
    def model_func(weight,bias,mem_in):

        #pdb.set_trace()
        output = nn.functional.linear(torch.cat((mem_in,wm_out),dim=1),weight,bias)

        return output


    v = torch.ones((items_per_batch,mem_size))

    jacobian_matrix = vjp(model_func, (weight,bias,mem_in),v)

    print(jacobian_matrix)
    print(jacobian_matrix[0].shape)
    print(jacobian_matrix[1].shape)
    print(jacobian_matrix[2].shape)

