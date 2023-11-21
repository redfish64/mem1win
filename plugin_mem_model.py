import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
import JacModule as jm
from dataclasses import dataclass

class PluginMemModel(nn.Module):
    def __init__(self,config,mem_model,pred_model):
        super().__init__()
        self.register_buffer('memory',torch.zeros((config.batch_size,
                                                   config.n_memory,
                                                   config.n_embd),
                             device=config.device,requires_grad=True))
        
        self.mem_model = mem_model
        self.pred_model = pred_model

        def snake_loop_fn(x):
            res,_,_ = pred_model(x,jm.get_jac_param(self.memory))
            res = torch.cat((res.view(config.batch_size,-1),self.memory.view(config.batch_size,-1)),dim=1)
            out = self.mem_model(res)
            return out

        self.snake = jm.Snake(snake_loop_fn, self.get_mem_params(),self.memory, [], config.jac_grad_decay, device=config.device)

        self.last_grads = None

    def get_mem_params(self):
        return list(self.mem_model.get_jac_params())

    def get_out_params(self):
        return list(self.pred_model.parameters())

    def forward(self,x, targets=None,compute_last_loss=False):
        res = self.pred_model(x,self.memory,targets,compute_last_loss)
        
        return res

    def update_memory_and_mem_params(self,inp):
        with torch.no_grad():
            if(self.last_grads is not None):
                self.snake.update_param_grads(self.last_grads, self.memory.grad)

            self.last_grads, next_mem = self.snake.run_jacobian(inp)

            self.memory.copy_(next_mem)

            # we need to clear the memory grad for the next round at some point so we do it here
            # TODO 3.5 maybe not the best place due to the obscure location, but where?
            self.memory.grad = None

    def configure_optimizers(self, learning_rate):
        optimizer = torch.optim.SGD(self.parameters(),lr=learning_rate)

        return optimizer

    def reset_memory_for_item_in_batch(self,index):
        with torch.no_grad():
            self.memory[index].zero_()

    def save_and_reset_memory(self):
        """Returns current memory for running estimated loss, etc. and creates a clone all zeroed out """
        old_mem = self.memory.clone()
        self.memory.zero_()
        return old_mem

    def restore_memory(self,mem):
        self.memory.copy_(mem)

@dataclass
class PMMConfig:
    batch_size: int = 1
    n_memory: int = 64
    n_embd: int = 256
    jac_grad_decay : float = 0.1 
    device : str = 'cpu'
