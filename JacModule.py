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

    def _calc_grad_from_running_jac_grad(self, running_jac_grad, jp):
        """
        Figures out the grad for the given jacparameter.
        """
        # number of dimensions in the memory output
        n_loop_dims = self.loop_param.dim()
        n_dims = running_jac_grad.dim()
        dimmed_loss_grad = self.loop_param.grad.view(
            list(self.loop_param.grad.shape) + [1] * (n_dims - n_loop_dims))

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

    def run(self, in_data):
        """
        Expects loss to be calculated relative to the parameters in init()
        Will update the grad by the running jac grad for each parameter in the snake.
        Also updates the running jac grad for each parameter. Should be called each cycle
        """
        grads, data_out = jac_grad(self.loop_fn, in_data, [
                                   self.loop_param], self.model_params, self.other_batch_indexed_params, **self.jac_grad_kw)

        loop_param_jac_grad = grads[0]

        for index, jp_jac_grad in enumerate(grads[1:]):
            jp = self.model_params[index]
            jp_running_jac_grad = self.running_jac_grads[index]
            running_portion_grad = self._calc_grad_from_running_jac_grad(jp_running_jac_grad, jp)
            if jp.grad is None:
                jp.grad = running_portion_grad
            else:
                jp.grad += running_portion_grad

            self.running_jac_grads[index] = self._calc_running_jac_grad(
                loop_param_jac_grad, jp_running_jac_grad, jp_jac_grad)

        return data_out


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


def test_snake(n_env, n_action, n_mem, n_batches, n_iters, n_test_iters, lr, start_env_fn, env_fn, calc_agent_loss_fn):

    # this stores memory and the last action taken. It requires grad so it can be used in a snake
    # as the loop param
    mem = torch.zeros(n_batches, n_mem, requires_grad=True)
    action = torch.zeros(n_batches, n_action, requires_grad=True)
    env = torch.zeros(n_batches, n_env, requires_grad=True)
    mem_action_env = torch.cat((mem, action, env), dim=1)

    # this corresponds to the memory the predictor can create to help it predict
    pred_mem_model = JacLinear(n_env+n_mem+n_action, n_mem)

    # the prediction model tries to predict what the environment will be given the current
    # memory, action and environment
    pred_env_model = torch.nn.Linear(n_env+n_mem+n_action, n_env)

    # this corresponds to the actions the actor can take
    actor_model = JacLinear(n_env+n_mem+n_action, n_action)

    def snake_loop_fn(in_data):
        models_input = torch.cat((in_data, get_jac_param(mem), get_jac_param(action)), dim=1)
        new_pred_env = pred_env_model(models_input)
        new_mem = pred_mem_model(models_input)
        new_action = actor_model(models_input)

        return torch.cat(torch.cat((new_mem,new_action,new_pred_env),dim=1))

    all_model_params = itertools.chain(pred_mem_model.parameters(),
                                       actor_model.parameters(),
                                       pred_env_model.parameters())
    snake = Snake(snake_loop_fn, all_model_params, , [action])

    def actor_snake_loop_fn(in_data):
        res = actor_snake_model(
            torch.cat((in_data, get_jac_param(mem), get_jac_param(action)), dim=1))
        return res

    # the actor snake loops over action, and decides what the agent should do, based on how the predictor is
    # predicting pleasure/pain
    actor_snake = Snake(actor_snake_loop_fn, list(
        actor_snake_model.parameters()), action, [mem])

    pred_optim = torch.optim.SGD(pred_win_model.parameters(), lr=lr)
    pred_snake_optim = torch.optim.SGD(pred_snake_model.parameters(), lr=lr)
    actor_snake_optim = torch.optim.SGD(actor_snake_model.parameters(), lr=lr)

    hidden_env,env = start_env_fn(n_batches,0)

    #training
    for i in range(n_iters):
        with torch.no_grad():
            hidden_env,env = env_fn(n_batches, hidden_env, env, action, i)

            pred_optim.zero_grad()
            if mem.grad is not None:
                mem.grad.zero_()
                # mem and action is not in the pred_optim, since we don't update the values based
                # on the gradiant, but only use them for the snake, so we have to zero them manually
                # here
                action.grad.zero_()

        #how good or bad the agent is doing in its environment
        agent_loss = calc_agent_loss_fn(env)

        pred_loss = (pred_win_model(mem_action) -
                     agent_loss).sum()

        # note that even though pred_optim only has pred_win_model parameters, this backward call
        # will also update the grad for mem_action
        pred_loss.backward()
        pred_optim.step()

        # this will set the gradiants for the pred_snake_model
        next_mem = pred_snake.run(env)
        pred_snake_optim.step()

        with torch.no_grad():
            # we need to zero out mem_action again, because now we calculate the gradiant
            # vs a winning environment (ie 1). Basically higher is better
            mem.grad.zero_()
            action.grad.zero_()

        #PERF we're recalculating pred_win_model(mem_action) a second time to reset the derivatives,
        #     we could also keep the result and use retain_graph=True. Don't know if it's worth it
        #     as a performance/memory tradeoff 
        actor_loss = (1. - pred_win_model(mem_action)).sum(0)
        actor_loss.backward()

        next_action = actor_snake.run(env)
        actor_snake_optim.step()

        with torch.no_grad():
            mem.copy_(next_mem)
            action.copy_(next_action)

    # env = start_env

    # #testing
    # for i in range(n_test_iters):
    #     in_data = env_fn(n_batches, env, action, i)





def test1():

    def calc_agent_loss_fn(hidden_env):
        env_pos = hidden_env[:,0]
        #env_speed = hidden_env[1]

        #if the environment position is within 1 of 42, there is no loss, otherwise there is an ever increasing loss
        agent_loss = (env_pos - 42.).abs().max(torch.tensor(1.))
        agent_loss = agent_loss * (agent_loss != 1.)

        return agent_loss

    def start_env_fn(n_batches,seed):
        torch.manual_seed(seed)
        hidden_env = torch.rand((n_batches,2)) * torch.tensor([[10.,0.1]]) + torch.tensor([[42.,-0.05]])
        return hidden_env,calc_agent_loss_fn(hidden_env)
    
    #simulate a blind cave. The agent doesn't know where it is or its speed
    def env_fn(batches, hidden_env, env, action, index):
        """
        Very simple accelerator, speed and position
        """
        #just an up/down accelerator
        action_accelerator = action[0]

        env_pos = hidden_env[0]
        env_speed = hidden_env[1]
        
        env_pos += env_speed
        env_speed += action_accelerator

        hidden_env[0] = env_pos
        hidden_env[1] = env_speed

        env = calc_agent_loss_fn(hidden_env).unsqueeze(dim=0)

        return hidden_env,env

    test_snake(n_env=2, n_action=1, n_mem=3, n_batches=4, n_iters=1000, n_test_iters=100, lr=1e-4,
               start_env_fn=start_env_fn,env_fn=env_fn, calc_agent_loss_fn=calc_agent_loss_fn)

if __name__ == '__main__':
    test1()
    
