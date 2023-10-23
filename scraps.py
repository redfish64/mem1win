def imagine_rest_of_story(enc,ms,block_pos_list,batch,current_works,tokens_to_print=100,context_tokens=100):
    with torch.no_grad():
        for i in range(len(block_pos_list)):
            # print part of story so far for context
            cw = current_works[i]
            bp = cw[i]

            enc.decode(cw[max(0,bp-context_tokens)

            sub_batch = torch.unsqueeze(batch[i],dim=0)

            logits = ms.ltm(block_pos_list,batch)

#this would be reinitialized everytime a new set of data is started (a new book, unrelated audio recording, etc.)
#along with memory
last_mem_in_to_squish_params_grad = None

class MemOutLayer(nn.Module):
    def __init__(self,in_lengths,out_length,**kwargs):
        super().__init__()
        self.in_lengths = in_lengths
        self.out_length = out_length

        total_in_length = sum(in_lengths)

        self.weight = nn.Parameter(torch.randn((out_length,total_in_length)) * 0.01)
        self.bias = nn.Parameter(torch.zeros((out_length,)) * 0.01)

    def forward(self, *args, **kwargs):
        ins_data_split = len(self.n_in_lengths)
        in_data = torch.cat(args[0:ins_data_split],dim=-1)
        return nn.functional.linear(in_data,self.get_jac_param(self.weight),self.get_jac_param(self.bias))
     

        #target loss to immediate params (this runs through mem_in_layer and core_layer, but not mem_out params)
        imm_loss_to_sq_mem_in_grad = ag.grad((loss,),(squishing_param_tensors+mem_in),retain_graph=True)
        imm_loss_to_squishing_param_tensors_grad = grad.grad_from_list(loss_to_sq_mem_in_grad[0:len(squishing_param_tensors)])
        loss_to_mem_in_grad = grad.grad_from_list(loss_to_sq_mem_in_grad[len(squishing_param_tensors):])

        loss_to_past_squishing_params_grad = grad.combine_grads(loss_to_mem_in_grad,last_mem_in_to_squish_params_grad)
        loss_to_squishing_params_grad = imm_loss_to_squishing_param_tensors_grad + loss_to_past_squishing_params_grad

        loss_to_mem_out_params_grad = loss_to_mem_in_grad * last_mem_in_to_mem_out_grad


        mem_to_mem_grad = torch.full((n_batch_size,n_mem,n_mem),0.)
        mem_to_wrapper_params_grad = torch.full((n_batch_size,n_mem,len(wrapper_params)),0.)

        for bi in range(n_batch_size):
            for mi in range(n_mem):
                (mem_item_to_mem_grad,*mem_item_to_wrapper_params_grad) = ag.grad((mw.last_run_mem_out[bi,mi],),[mw.memory[bi]]+wrapper_params,retain_graph=True)
                pdb.set_trace()
                mem_to_mem_grad[bi,mi] = mem_item_to_mem_grad
                mem_to_wrapper_params_grad[bi,mi] = mem_item_to_wrapper_params_grad

        pdb.set_trace()

class JacParameter(nn.Parameter):
    def __init__(self,is_batch_indexed=False,**kwargs):
        """
        is_batched_indexed - if true, then each parameter is expected to have its outermost dimension the same size as
                             the number of items per batch. If false, each JacParameter object will be repeated for each
                             batch item. 
        """
        super().__init__(**kwargs)
        self.is_batch_indexed = is_batch_indexed

    @property
    def data(self):
        pdb.set_trace()
        # if we are being called from jac_grad, we use the parameters passed into the underlying jacrev call, otherwise
        # the straight data
        if(_jac_context is None):
            return super(JacParameter).data

        param_loc = _jac_context.param_to_array_index[id(self)]
        if(param_loc is None):
            return super(JacParameter).data

        return _jac_context.params[self]
        
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
                           
def _get_or_update_submodule(model,key,update_fn=None):
    if isinstance(model, nn.ModuleList) and isinstance(key, int) or isinstance(model, nn.ModuleDict) and isinstance(key, str):
        ret_sub_model = model[key]
        if(update_fn is not None):
            if(ret_sub_model is None):
                return None
            model[key] = update_fn(ret_sub_model)
        return ret_sub_model
    else:
        ret_sub_model = getattr(model, key)
        if(update_fn is not None):
            if(ret_sub_model is None):
                return None
            setattr(model,key,update_fn(ret_sub_model))
        return ret_sub_model

# Recursive function to navigate through the path
def wrap_sub_module(model, path, update_fn, path_index=0):
    if model is None:
        raise ValueError('bad path element {path=}, {path[path_index]=}')
    if path_index == len(path):
        raise ValueError('internal error, at end of path')
        
    if path_index == len(path-1):
        _get_or_update_submodule(model,path[path_index], update_fn)
    else:
        next_model = _get_or_update_submodule(model,path[path_index])
        wrap_sub_module(model,path,path_index+1)

class JacLinear(nn.Module):
    def __init__(self,n_in,n_out,requires_grad,has_bias=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.weight = torch.nn.Parameter(torch.randn((n_out,n_in)) * 0.01,requires_grad)
        self.bias = torch.nn.Parameter(torch.zeros((n_out,)) * 0.01,requires_grad) if has_bias else None

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
    
        self.running_jac_grads = ([torch.zeros(*loop_param.shape, *loop_param.shape[1:])] +
                                  [torch.zeros(*loop_param.shape, *jp.shape) for jp in self.model_params])
def test1():
    def data_fn(batches, action, index):
        # two oscillating inputs, offseted by item in batch
        v1 = index * 0.05 * 3.14159
        v2 = index * 0.075 * 3.14159 + 1

        res0 = torch.tensor([v1, v2])
        res = torch.stack([res0 + b for b in range(batches)])
        res = torch.cos(res)

        return res

    def calc_pain_pleasure_fn(action, env):
        # happy if action is the same sign as the sum of the two inputs
        ca = action.clamp(min=-1., max=1.)
        return (env.sum(dim=1) * ca.sum(dim=1)).unsqueeze(0).T

    test_snake(2, data_fn, calc_pain_pleasure_fn)
def create_snake_sss(ms):
    snake = Snake(snake_loop_fn, 

