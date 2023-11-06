import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
import JacModule as jm
from dataclasses import dataclass



class SimpleMemModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.register_buffer('memory',torch.zeros((config.batch_size,
                                                   config.n_memory),
                             device=config.device,requires_grad=True))
        
        self.tok_embedding=nn.Embedding(config.vocab_size,config.n_embd,device=config.device)

        n_mem_inp = config.n_memory + config.block_size * config.n_embd
        n_mem_hidden = config.n_inner_mem_size

        self.mem_model = nn.Sequential(
            jm.JacLinear(n_mem_inp, n_mem_hidden,device=config.device),
            nn.ReLU(),
            jm.JacLinear(n_mem_hidden, n_mem_hidden,device=config.device),
            nn.ReLU(),
            jm.JacLinear(n_mem_hidden, config.n_memory,device=config.device))

        n_pred_inp = config.n_memory + config.block_size * config.n_embd
        n_pred_hidden = config.n_inner_pred_size
        n_pred_out = config.vocab_size * config.block_size
        self.pred_model = nn.Sequential(
            nn.Linear(n_pred_inp, n_pred_hidden,device=config.device),
            nn.ReLU(),
            nn.Linear(n_pred_hidden, n_pred_hidden,device=config.device),
            nn.ReLU(),
            nn.Linear(n_pred_hidden, n_pred_out,device=config.device),
        )

        def snake_loop_fn(x):
            #TODO 2.9 we aren't currently updating the token embedding weights in the
            #memory, because that would require jacobian weights to be used within the
            #embedding model. We could do this, but I'm not sure yet whether that is the
            #best thing or not, so... eh.
            emb = self.tok_embedding(x)
            inp = torch.cat((jm.get_jac_param(self.memory),emb.view(emb.shape[0],-1)),dim=1)
            out = self.mem_model(inp)
            return out

        self.snake = jm.Snake(snake_loop_fn, self.get_mem_params(),self.memory, [], config.jac_grad_decay, device=config.device)

        self.last_grads = None

    def get_mem_params(self):
        return list(self.mem_model.parameters())

    def get_out_params(self):
        return list(self.pred_model.parameters())

    def forward(self,x, targets=None,compute_last_loss=False):
        emb = self.tok_embedding(x)
        inp = torch.cat((self.memory,emb.view(emb.shape[0],-1)),dim=1)
        res = self.pred_model(inp)

        if targets is not None:
            # if we are given some desired targets also calculate the loss

            #PERF in mem_model, the last Linear is broken out so that only the final token in the window
            #gets returned in inference mode(below). We don't care about efficiency yet, so we aren't
            #doing that right now
            logits = res
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if(compute_last_loss):
                last_item_loss = F.cross_entropy(logits[:,-1,:].view(-1, logits.size(-1)), targets[:,-1].view(-1), ignore_index=-1)
            else:
                last_item_loss = None
        else:
            logits = emb[:,[-1],:]
            loss = None
            last_item_loss = None

        return logits, loss, last_item_loss
        
        return res

    def update_memory_and_mem_params(self,x):
        with torch.no_grad():
            if(self.last_grads is not None):
                self.snake.update_param_grads(self.last_grads, self.memory.grad)

            self.last_grads, next_mem = self.snake.run_jacobian(x)

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

@dataclass
class SMMConfig:
    batch_size: int = 1
    n_memory: int = 64
    vocab_size: int = 256
    n_embd: int = 256
    block_size: int = 1
    n_inner_mem_size : int = 64
    n_inner_pred_size : int = 64
    jac_grad_decay : float = 0.1 
    device : str = 'cpu'
