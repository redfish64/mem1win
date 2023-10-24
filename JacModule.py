import torch
import torch.nn as nn
import torch.nn.functional as nf
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

    def __init__(self, loop_fn, model_params, loop_param, other_batch_indexed_params,decay=0.05, **jac_grad_kw):
        super(Snake, self).__init__()
        self.loop_fn = loop_fn
        self.model_params = model_params
        self.loop_param = loop_param
        self.other_batch_indexed_params = other_batch_indexed_params
        self.decay = decay
        self.jac_grad_kw = jac_grad_kw
        self.jac_params = [self.loop_param] + self.model_params
        self.running_jac_grads = [torch.zeros(*loop_param.shape, *jp.shape) for jp in self.model_params]

    def _calc_grad_from_running_jac_grad(self, running_jac_grad, jp, next_loop_param_grad):
        """
        Figures out the grad for the given jacparameter.
        """
        # number of dimensions in the memory output
        n_loop_dims = next_loop_param_grad.dim()
        n_dims = running_jac_grad.dim()
        dimmed_loss_grad = next_loop_param_grad.view(
            list(next_loop_param_grad.shape) + [1] * (n_dims - n_loop_dims))

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
        res = u.matmul_batch_ldim_tensors(loop_param_jac_grad, split_point, jp_running_jac_grad, split_point)

        # average that with the current cycle effect using a given percentage, named decay
        return res * \
            (1. - self.decay) + jp_jac_grad * self.decay

    def run_jacobian(self, in_data):
        """
        runs the snake, returning the output and the gradiants. update_param_grads should be called next, after total
        loss is calculated and backward() is called.
        """
        grads, data_out = jac_grad(self.loop_fn, in_data,
                                   [self.loop_param], self.model_params, self.other_batch_indexed_params,
                                   **self.jac_grad_kw)

        data_out = data_out.detach()
        data_out.requires_grad = True

        return grads,data_out

        
    def update_param_grads(self, grads, next_loop_param_grad):
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

            #update the running_jac_grad to the end of the current cycle
            jp_running_jac_grad = self._calc_running_jac_grad(
                loop_param_jac_grad, jp_running_jac_grad, jp_jac_grad)

            #link the running_jac_grad which extends to the end of the current cycle to the next_loop_param.grad
            #which goes from the end of the current cycle to the result
            running_portion_grad = self._calc_grad_from_running_jac_grad(jp_running_jac_grad, jp,next_loop_param_grad)
            if jp.grad is None:
                jp.grad = running_portion_grad
            else:
                jp.grad += running_portion_grad

            self.running_jac_grads[index] = jp_running_jac_grad


class JacLinear(nn.Module):
    def __init__(self, n_in, n_out, requires_grad=True, has_bias=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.weight = torch.nn.Parameter(
            torch.randn((n_out, n_in)) * 0.1, requires_grad)
        self.bias = torch.nn.Parameter(torch.zeros(
            (n_out,)), requires_grad) if has_bias else None

    def get_jac_params(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def forward(self, inp):
        return nn.functional.linear(inp, get_jac_param(self.weight), get_jac_param(self.bias) if self.bias is not None else None)

class PassThru(torch.nn.Module):
    def __init__(self, m, start,end):
        super().__init__()
        self.m = m
        self.start = start
        self.end = end

    def forward(self,in_data):
        res = self.m(in_data)
        res = res + in_data[:,self.start:self.end]

        return res

class ZeroToOne(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self,in_data):
        res = self.m(in_data)
        res = nf.tanh(res) * 0.5 + 0.5

        return res

    
class LilOleModel(torch.nn.Module):
    def __init__(self, n_in, n_out,scale=1.,has_tanh=True,n_hidden = 10):
        super().__init__()
        self.scale = scale


        mods = [JacLinear(n_in, n_hidden),
                torch.nn.ReLU(),
                JacLinear(n_hidden, n_hidden),
                torch.nn.ReLU(),
                JacLinear(n_hidden, n_out)]
        if(has_tanh):
            mods.append(torch.nn.Tanh())

        self.m = torch.nn.Sequential(*mods)

    def forward(self,in_data):
        res = self.m(in_data)
        return res

def test_snake(n_env, n_action, n_mem, n_batches, n_iters, n_test_iters, lr, pred_mem_model,actor_model,pred_env_model,start_env_fn, env_fn, calc_agent_loss_fn,decay):

    n_loop_values = n_mem+n_action+n_env

    #the loop parameter is the memory, action, and predicted environment
    mem_action_env = torch.randn(n_batches, n_loop_values) * .1
    mem_action_env.requires_grad = True

    def snake_loop_fn(in_data):
        #mem_action_env already contains in_data (which is the environment), so we ignore it here
        models_input = get_jac_param(mem_action_env)
        new_pred_env = pred_env_model(models_input)
        new_mem = pred_mem_model(models_input)
        new_action = actor_model(models_input)

        return torch.cat((new_mem,new_action,new_pred_env),dim=1)

    l_pred_mem_model_params = list(pred_mem_model.parameters())
    l_actor_model_params = list(actor_model.parameters())
    l_pred_env_params = list(pred_env_model.parameters())
    all_model_params = l_pred_mem_model_params + l_actor_model_params + l_pred_env_params

    snake = Snake(snake_loop_fn, all_model_params, mem_action_env,[],decay=decay)

    optim = torch.optim.SGD(all_model_params, lr=lr)

    mem_start = 0
    mem_end = n_mem
    action_start = n_mem
    action_end = action_start + n_action
    env_start = action_end
    env_end = action_end + n_env

    hidden_env,env = start_env_fn(n_batches,0)

    #setup the initial environment for each item in batch
    with torch.no_grad():
        mem_action_env[:,env_start:env_end].copy_(env)

    #training
    for i in range(n_iters):
        with torch.no_grad():
            optim.zero_grad()

            if mem_action_env.grad is not None:
                # mem_action_env is not in any optim, since we don't update the values based
                # on the gradiant, but only use them for the snake, so we have to zero them manually
                # here
                mem_action_env.grad.zero_()

        #the agent reacts to the current environment (env) with a new memory, action, and
        #predicted environment
        grads,next_mem_action_penv = snake.run_jacobian(env)

        next_mem = next_mem_action_penv[:,0:action_start]
        next_action = next_mem_action_penv[:,action_start:action_end]
        next_penv = next_mem_action_penv[:,env_start:env_end]

        #calculate the next environment so we can compute loss
        next_hidden_env,next_env = env_fn(n_batches, hidden_env, env, next_action, i)

        #the environment predictor's loss is how off it is from the actual environment
        #TODO 4 I think that maybe we need to somehow prioritize the values of some elements
        #in the environment over others based on action memory. Have to think about how this
        #could be set up.
        pred_env_loss = ((next_penv-next_env)**2).sum()

        #the agent loss uses the pred_env_loss to try and decide what to do
        agent_loss = (next_penv - next_action)**2

        #there is no memory loss. the memory part of the model gets updated as part of the snake updating
        #the previous cycles

        #TODO 3 maybe balance these?
        total_loss = (pred_env_loss + agent_loss).sum(dim=0)
        #total_loss = (pred_env_loss + agent_loss).sum(dim=0)

        total_loss.backward()

        with torch.no_grad():
            snake.update_param_grads(grads,next_mem_action_penv.grad)
            if(i%199 == 0):
                print(f'{i=}\n{next_mem=}\n{next_env=}\n{next_penv=}\n{next_action=}')
                if(i > 10000000):
                    pdb.set_trace()
            optim.step()
            mem_action_env.copy_(next_mem_action_penv)

            mem_action_env[:,env_start:env_end].copy_(env)

            #H*CK to see what happens if we add env to mem
            #mem_action_env[:,mem_start:mem_end] = (mem_action_env[:,mem_start:mem_end] + mem_action_env[:,env_start:env_end]).detach()

            env = next_env
            hidden_env = next_hidden_env
    # env = start_env

    # #testing
    # for i in range(n_test_iters):
    #     in_data = env_fn(n_batches, env, action, i)


def test1():
    """
    Single flat environment. Model can't figure out what to do.
    """
    def calc_agent_loss_fn(action,hidden_env,env):
        loss = (action-env).abs().detach()
        return loss

    def start_env_fn(n_batches,seed):
        torch.manual_seed(seed)
        return None,torch.zeros((n_batches,1))
    
    def env_fn(batches, hidden_env, env, action, index):
        env = torch.ones((batches,1)) * -0.0050
        return hidden_env,env

    test_snake(n_env=1, n_action=1, n_mem=3, n_batches=1, n_iters=100000, n_test_iters=100, lr=1e-4,
               start_env_fn=start_env_fn,env_fn=env_fn, calc_agent_loss_fn=calc_agent_loss_fn)

def test2():
    """
    Sign wave environment
    """
    def calc_agent_loss_fn(action,hidden_env,env):
        loss = (action-env).abs().detach()
        return loss

    def start_env_fn(n_batches,seed):
        torch.manual_seed(seed)
        return torch.rand((n_batches,1)),torch.zeros((n_batches,1))
    
    def env_fn(batches, hidden_env, env, action, index):
        env = torch.cos(hidden_env).detach()
        hidden_env += 0.001
        return hidden_env,env

    test_snake(n_env=1, n_action=1, n_mem=3, n_batches=1, n_iters=100000, n_test_iters=100, lr=1e-4,
               start_env_fn=start_env_fn,env_fn=env_fn, calc_agent_loss_fn=calc_agent_loss_fn,decay=0.3)

def test3():
    """
    Simple pattern environment
    """
    def calc_agent_loss_fn(action,hidden_env,env):
        loss = ((action-env)**2).detach()
        return loss

    def start_env_fn(n_batches,seed):
        torch.manual_seed(seed)
        return torch.arange(end=n_batches),torch.zeros((n_batches,1))
    
    def env_fn(batches, hidden_env, env, action, index):
        #env = ((torch.remainder(hidden_env,2) == 0)) * 1.
        env = ((torch.remainder(hidden_env,3) == 0) + (torch.remainder(hidden_env,5) == 0)) * 1.
        #env = torch.ones_like(env) #H*CK!!!
        hidden_env += 1
        return hidden_env,env

    n_env = 1
    n_action = 1
    n_mem = 5
    n_loop_values = n_env+n_action+n_mem
    

    # this corresponds to the memory the predictor can create to help it predict
    #pred_mem_model = LilOleModel(n_loop_values, n_mem)
    #    pred_mem_model = PassThru(LilOleModel(n_loop_values, n_mem),0,n_mem)
    pred_mem_model = JacLinear(n_loop_values, n_mem)

    # this corresponds to the actions the actor can take
    actor_model = ZeroToOne(JacLinear(n_loop_values, n_action))

    # the prediction model tries to predict what the environment will be given the current
    # memory, action and environment
    pred_env_model = ZeroToOne(JacLinear(n_loop_values, n_env))

    #H*CK to see what happens if we make the actor and environment model initially very interested in the memory
    # with torch.no_grad():
    #     actor_model.weight[0:n_mem] += 1.
    #     pred_env_model.weight[0:n_mem] += 1.

    test_snake(n_env=n_env, n_action=n_action, n_mem=n_mem, n_batches=1, n_iters=100000, n_test_iters=100, lr=1e-2,
               pred_mem_model=pred_mem_model,actor_model=actor_model,pred_env_model=pred_env_model,
               start_env_fn=start_env_fn,env_fn=env_fn, calc_agent_loss_fn=calc_agent_loss_fn,decay=0.3)

def test_snake2(n_env, n_mem, n_batches, n_iters, n_test_iters, lr, pred_mem_model,pred_env_model,start_env_fn, env_fn, calc_agent_loss_fn,decay):
    """
    Like test_snake, but doesn't have an actor model, only an environment and a predictor
    """

    n_loop_values = n_mem

    #the loop parameter is only the memory
    mem = torch.randn(n_batches, n_loop_values) * .1
    mem.requires_grad = True

    def snake_loop_fn(in_data):
        models_input = torch.cat((get_jac_param(mem),in_data),dim=1)
        new_mem = pred_mem_model(models_input)

        return new_mem


    snake = Snake(snake_loop_fn, list(pred_mem_model.parameters()), mem,[],decay=decay)

    all_model_params = list(pred_mem_model.parameters()) + list(pred_env_model.parameters())

    optim = torch.optim.SGD(all_model_params, lr=lr)

    hidden_env,env = start_env_fn(n_batches,0)

    #the last gradiants from the mem over the pred_mem_model parameters
    last_grads = None

    #training
    for i in range(n_iters):
        with torch.no_grad():
            optim.zero_grad()

            if mem.grad is not None:
                # mem is not in any optim, since we don't update the values based
                # on the gradiant, but only use them for the snake, so we have to zero them manually
                # here
                mem.grad.zero_()

        #calculate the next environment so we can compute loss
        next_hidden_env,next_env = env_fn(n_batches, hidden_env, env, i)

        mem_env = torch.cat((mem,env),dim=1)

        next_penv = pred_env_model(mem_env)

        loss = ((next_penv-next_env)**2).sum()

        loss.backward()

        with torch.no_grad():
            if(last_grads is not None):
                snake.update_param_grads(last_grads, mem.grad)

            #the agent reacts to the current environment (env) with a new memory, action, and
            #predicted environment
            last_grads,next_mem = snake.run_jacobian(env)

            if(i%199 == 0):
                print(f'{i=}\n{next_mem=}\n{next_env=}\n{next_penv=}')
                if((next_env-next_penv.abs()) < 0.5):
                    print('good')
                else:
                    print('bad')
                
                if(i > 10000000):
                    pdb.set_trace()
            optim.step()

            mem.copy_(next_mem)
            env = next_env
            hidden_env = next_hidden_env
    # env = start_env

    # #testing
    # for i in range(n_test_iters):
    #     in_data = env_fn(n_batches, env, action, i)

def test4():
    """
    Simple pattern environment
    """
    def calc_agent_loss_fn(action,hidden_env,env):
        loss = ((action-env)**2).detach()
        return loss

    def start_env_fn(n_batches,seed):
        torch.manual_seed(seed)
        return torch.arange(end=n_batches),torch.zeros((n_batches,1))
    
    def env_fn(batches, hidden_env, env, index):
        #env = ((torch.remainder(hidden_env,2) == 0)) * 1.
        env = ((torch.remainder(hidden_env,3) == 0) + (torch.remainder(hidden_env,5) == 0)) * 1.
        #env = torch.ones_like(env) #H*CK!!!
        hidden_env += 1
        return hidden_env,env.unsqueeze(1)

    n_batches = 1
    n_env = 1
    n_mem = 10
    n_inp_values = n_env+n_mem
    

    # this corresponds to the memory the predictor can create to help it predict
    pred_mem_model = torch.nn.Sequential(torch.nn.LayerNorm((n_mem+n_env,),elementwise_affine=False),
                                         JacLinear(n_inp_values, n_mem))

    # the prediction model tries to predict what the environment will be given the current
    # memory, action and environment
    pred_env_model = ZeroToOne(JacLinear(n_inp_values, n_env))

    #H*CK to see what happens if we make the actor and environment model initially very interested in the memory
    # with torch.no_grad():
    #     actor_model.weight[0:n_mem] += 1.
    #     pred_env_model.weight[0:n_mem] += 1.

    test_snake2(n_env=n_env, n_mem=n_mem, n_batches=n_batches, n_iters=100000, n_test_iters=100, lr=1e-2,
                pred_mem_model=pred_mem_model,pred_env_model=pred_env_model,
                start_env_fn=start_env_fn,env_fn=env_fn, calc_agent_loss_fn=calc_agent_loss_fn,decay=0.3)

def testx():

    def calc_agent_loss_fn(action,hidden_env,env):
        env_pos = hidden_env[:,0].unsqueeze(1)
        #env_speed = hidden_env[1]

        #if the environment position is within 1 of 0, there is no loss, otherwise there is an ever increasing loss
        agent_loss = (env_pos - 0. + action).abs().max(torch.tensor(1.))
        agent_loss = agent_loss * (agent_loss != 1.)

        return agent_loss

    def start_env_fn(n_batches,seed):
        torch.manual_seed(seed)
        hidden_env = torch.rand((n_batches,2)) * torch.tensor([[1,0.1]]) + torch.tensor([[-0.5,-0.05]])
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
        env_pos = torch.clamp(env_pos,-5.,5.)

        hidden_env = torch.stack((env_pos,env_speed),dim=1)

        #we detach here because the optimizer shouldn't be able to peek at the hidden environment to
        #know how to get the loss less. The system needs to use the predicted environment to figure out
        #the loss
        env = calc_agent_loss_fn(action,hidden_env,env).detach()

        return hidden_env,env,env

    test_snake(n_env=1, n_action=1, n_mem=3, n_batches=1, n_iters=100000, n_test_iters=100, lr=3e-3,
               start_env_fn=start_env_fn,env_fn=env_fn, calc_agent_loss_fn=calc_agent_loss_fn,decay=0.3)

if __name__ == '__main__':
    test4()
    
