# modules that can work with jacrev. The only difference is that it's parameters are passed into it during the forward call.

import torch
import torch.nn as nn
import torch.func as f
from dataclasses import dataclass
import pdb
import util as u
import itertools

def jac_grad(module, inp_batch, batch_indexed_grad_jac_params, repeated_grad_jac_params, batch_indexed_params, *extra_module_args, vmap_randomness='error', **extra_module_kwargs):
    """
    Runs the module against the input batch and returns the output. The jacobian is placed into the individual
    selected parameters, as `<param>.jac_grad`
    module - the module or function to run.
    inp_batch - a batch of input values as a tensor in the shape (B,...) where B is the number of items in the batch
    repeated_grad_jac_params - params that will be repeated for each item of the batch
    batch_indexed_grad_jac_params - params that have a separate value for each item in the batch
    batch_indexed_grad_jac_params - params that have a separate value for each item in the batch but the jac grad is not calculated
    grad_jac_params - the jacobian will be calculated against the given JacParameter objects. The results
                      will appear in `<jac_param>.jac_grad`
    vmap_randomness    - vmap randomness parameter

    returns the gradiants for each of the grad_jac_params in the same order provided and the output of "module"
    """

    # note the order of these lists matches the order of the arguments in vmap below. This
    # is so we can associate the original tensor with the associated jacrev tensor
    all_params = batch_indexed_params + \
        batch_indexed_grad_jac_params + repeated_grad_jac_params

    def model_func(input_item, *params):
        for index, jp, p in zip(range(len(params)), params, all_params):
            if (index < len(batch_indexed_params) + len(batch_indexed_grad_jac_params)):
                p.jac_param = jp.unsqueeze(0)
            else:
                p.jac_param = jp

        output = module(input_item.unsqueeze(0)).squeeze(0)

        for p in all_params:
            del p.jac_param

        # we specify output twice, because the first output value will be replaced by the grads, and the second left as is
        # (this is due to has_aux=True below)
        return output, output

    # +1 is for the input batch
    n_indexed_params = len(batch_indexed_grad_jac_params) + \
        len(batch_indexed_params)+1
    n_non_indexed_params = len(repeated_grad_jac_params)
    graded_params_start = 1+len(batch_indexed_params)
    graded_params_end = graded_params_start + len(batch_indexed_grad_jac_params) + len(repeated_grad_jac_params)

    # the argument indexes we want to compute the jacobian gradiant against
    jacrev_argnums = tuple(list(range(graded_params_start, graded_params_end)))
    # the argument indexes we want to vmap to index against when it loops through the batch.
    vmap_in_dims = tuple([0]*n_indexed_params+[None]*n_non_indexed_params)

    (grads, res) = f.vmap(f.jacrev(model_func, argnums=jacrev_argnums, has_aux=True), in_dims=vmap_in_dims,
                          randomness=vmap_randomness)(inp_batch, *batch_indexed_params, *batch_indexed_grad_jac_params, *repeated_grad_jac_params)

    return grads, res


def get_jac_param(jp):
    # we do this check for the attribute here, for when the parameter is run outside of a call
    # to the jac_grad function
    return jp.jac_param if hasattr(jp, 'jac_param') else jp


class Snake(object):
    """

    """

    def __init__(self, loop_fn, model_params, loop_param, other_batch_indexed_params, decay=0.99, **jac_grad_kw):
        super(Snake, self).__init__()
        self.loop_fn = loop_fn
        self.model_params = model_params
        self.loop_param = loop_param
        self.other_batch_indexed_params = other_batch_indexed_params
        self.decay = decay
        self.jac_grad_kw = jac_grad_kw
        self.jac_params = [self.loop_param] + self.model_params
        self.running_jac_grads = [torch.zeros(*loop_param.shape, *jp.shape) for jp in self.model_params]

    def _calc_grad_from_running_jac_grad(self, running_jac_grad, jp, next_loop_param):
        """
        Figures out the grad for the given jacparameter.
        """
        # number of dimensions in the memory output
        n_loop_dims = next_loop_param.dim()
        n_dims = running_jac_grad.dim()
        dimmed_loss_grad = next_loop_param.grad.view(
            list(next_loop_param.grad.shape) + [1] * (n_dims - n_loop_dims))

        res = dimmed_loss_grad * running_jac_grad

        res = torch.sum(
            res, list(range(0, n_loop_dims)))

        return res

    def _calc_running_jac_grad(self, loop_param_jac_grad, jp_running_jac_grad, jp_jac_grad):
        """
        This updates the jac gradiant for a cycle through the model. It takes the existing running gradiant and averages
        it with the current gradiant. This way we get an estimation of how a parameter affects the output of
        the current round from all prior rounds.

        decay - how much to decay the previous old gradiants when updating, a value from 0 to 1.
        cycle_out_to_cycle_out - the gradiant from how the output of the last cycle affects the current one
        """
        #The split point is the position in the shape where out params end and the in params start.
        #There are two factos to consider here.
        # 1. The loop param always has a batch parameter dimension on the outer left side
        # 2. The loop_param_jac_grad and jp_running_jac_grad always have the same split point
        #    because they always have the same output, which is the loop_param. 
        split_point = self.loop_param.dim()

        #
        # calc the effect on the previous cycles relative to the current cycle loop_param out
        res = u.cross_product_batch_ldim_tensors(loop_param_jac_grad, split_point, jp_running_jac_grad, split_point)

        # average that with the current cycle effect using a given percentage, named decay
        return res * \
            (1. - self.decay) + jp_jac_grad * self.decay

    def run_jacobian(self, in_data):
        """
        runs the snake, returning the output and the gradiants. update_param_grads should be called next, after total
        loss is calculated and backward() is called.
        """
        grads, data_out = jac_grad(self.loop_fn, in_data, [
                                   self.loop_param], self.model_params, self.other_batch_indexed_params, **self.jac_grad_kw)

        data_out = data_out.detach()
        data_out.requires_grad = True

        return grads,data_out

        
    def update_param_grads(self, grads, next_loop_param):
        """
        Expects loss to be calculated relative to the parameters in init()
        Will update the grad by the running jac grad for each parameter in the snake.
        Also updates the running jac grad for each parameter. Should be called after
        run_jacobian
        """
        loop_param_jac_grad = grads[0]

        for index, jp_jac_grad in enumerate(grads[1:]):
            jp = self.model_params[index]
            jp_running_jac_grad = self.running_jac_grads[index]
            running_portion_grad = self._calc_grad_from_running_jac_grad(jp_running_jac_grad, jp,next_loop_param)
            if jp.grad is None:
                jp.grad = running_portion_grad
            else:
                jp.grad += running_portion_grad

            self.running_jac_grads[index] = self._calc_running_jac_grad(
                loop_param_jac_grad, jp_running_jac_grad, jp_jac_grad)


class JacLinear(nn.Module):
    def __init__(self, n_in, n_out, requires_grad=True, has_bias=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.weight = torch.nn.Parameter(
            torch.randn((n_out, n_in)) * 0.01, requires_grad)
        self.bias = torch.nn.Parameter(torch.zeros(
            (n_out,)) * 0.01, requires_grad) if has_bias else None

    def get_jac_params(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def forward(self, inp):
        return nn.functional.linear(inp, get_jac_param(self.weight), get_jac_param(self.bias) if self.bias is not None else None)

class LilOleModel(torch.nn.Module):
    def __init__(self, n_in, n_out,scale=1.):
        super().__init__()
        self.scale = scale
        self.m = torch.nn.Sequential(JacLinear(n_in, n_out),
                                     torch.nn.Tanh())

    def forward(self,in_data):
        return self.m(in_data) * self.scale

def test_snake(n_env, n_action, n_mem, n_batches, n_iters, n_test_iters, lr, start_env_fn, env_fn, calc_agent_loss_fn):

    n_loop_values = n_mem+n_action+n_env

    #the loop parameter is the memory, action, and predicted environment
    mem_action_penv = torch.randn(n_batches, n_loop_values) * .1
    mem_action_penv.requires_grad = True

    # this corresponds to the memory the predictor can create to help it predict
    pred_mem_model = LilOleModel(n_loop_values, n_mem)

    # this corresponds to the actions the actor can take
    actor_model = LilOleModel(n_loop_values, n_action,0.1) 

    # the prediction model tries to predict what the environment will be given the current
    # memory, action and environment
    pred_env_model = JacLinear(n_loop_values, n_env)

    def snake_loop_fn(in_data):
        #we use the predicted environment from the last round as input to
        #get the next round values
        #
        #TODO 4 I'm not sure whether the actual environment or the predicted
        #environment is better here. The problem with the actual environment, is
        #that it doesn't actually loop, so it becomes tricky as to when and how
        #to insert the actual environment into the snake.
        models_input = get_jac_param(mem_action_penv)
        new_pred_env = pred_env_model(models_input)
        new_mem = pred_mem_model(models_input)
        new_action = actor_model(models_input)

        return torch.cat((new_mem,new_action,new_pred_env),dim=1)

    all_model_params = list(itertools.chain(pred_mem_model.parameters(),
                                            actor_model.parameters(),
                                            pred_env_model.parameters()))

    snake = Snake(snake_loop_fn, all_model_params, mem_action_penv, [])

    optim = torch.optim.SGD(all_model_params, lr=lr)

    action_start = n_mem
    action_end = action_start + n_action
    env_start = action_end
    env_end = action_end + n_env

    hidden_env,env = start_env_fn(n_batches,0)

    #setup the initial environment for each item in batch
    with torch.no_grad():
        mem_action_penv[:,env_start:env_end].copy_(env)

    #training
    for i in range(n_iters):
        with torch.no_grad():
            optim.zero_grad()

            if mem_action_penv.grad is not None:
                # mem_action_env is not in any optim, since we don't update the values based
                # on the gradiant, but only use them for the snake, so we have to zero them manually
                # here
                mem_action_penv.grad.zero_()

        #the agent reacts to the current environment (env) with a new memory, action, and
        #predicted environment
        grads,next_mem_action_penv = snake.run_jacobian(env)

        next_action = next_mem_action_penv[:,action_start:action_end]
        next_penv = next_mem_action_penv[:,env_start:env_end]

        #calculate the next environment so we can compute loss
        next_hidden_env,next_env,agent_loss = env_fn(n_batches, hidden_env, env, next_action, i)

        #the environment predictor's loss is how off it is from the actual environment
        #TODO 4 I think that maybe we need to somehow prioritize the values of some elements
        #in the environment over others based on action memory. Have to think about how this
        #could be set up.
        pred_env_loss = (next_penv-next_env).abs().sum()

        #there is no memory loss. the memory part of the model gets updated as part of the snake updating
        #the previous cycles

        #TODO 3 maybe balance these?
        total_loss = (pred_env_loss + agent_loss[:,0]).sum(dim=0)

        total_loss.backward()

        with torch.no_grad():
            snake.update_param_grads(grads,next_mem_action_penv)
            if(i%100 == 0):
                print(f'{i=}\n,{hidden_env=}\n{next_env=}\n{next_penv=}\n{next_action=}')
            optim.step()
            mem_action_penv.copy_(next_mem_action_penv)
            env = next_env
    # env = start_env

    # #testing
    # for i in range(n_test_iters):
    #     in_data = env_fn(n_batches, env, action, i)


def test1():

    def calc_agent_loss_fn(action,hidden_env,env):
        env_pos = hidden_env[:,0].unsqueeze(1)
        #env_speed = hidden_env[1]

        #if the environment position is within 1 of 42, there is no loss, otherwise there is an ever increasing loss
        agent_loss = (env_pos - 42. + action).abs().max(torch.tensor(1.))
        agent_loss = agent_loss * (agent_loss != 1.)

        return agent_loss

    def start_env_fn(n_batches,seed):
        torch.manual_seed(seed)
        hidden_env = torch.rand((n_batches,2)) * torch.tensor([[10.,0.1]]) + torch.tensor([[42.,-0.05]])
        return hidden_env,calc_agent_loss_fn(torch.zeros((n_batches,1)),hidden_env,None)
    
    #simulate a blind cave. The agent doesn't know where it is or its speed
    def env_fn(batches, hidden_env, env, action, index):
        """
        Very simple accelerator, speed and position
        """
        #just an up/down accelerator
        action_accelerator = action[:,0]

        env_pos = hidden_env[:,0]
        env_speed = hidden_env[:,1]
        
        env_pos += env_speed
        env_speed += action_accelerator

        #we detach here so that the backward() grad_fn chain doesn't extend into what was already
        #done by the snake, sss
        hidden_env = torch.stack((env_pos,env_speed),dim=1).detach()
        hidden_env.requires_grad = True

        env = calc_agent_loss_fn(action,hidden_env,env)

        return hidden_env,env,env

    test_snake(n_env=1, n_action=1, n_mem=3, n_batches=4, n_iters=100000, n_test_iters=100, lr=1e-4,
               start_env_fn=start_env_fn,env_fn=env_fn, calc_agent_loss_fn=calc_agent_loss_fn)

if __name__ == '__main__':
    test1()
    
