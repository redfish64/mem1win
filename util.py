import torch
import torch.nn as nn

from torch.nn import functional as F

import re


# this reduces the init weights and bias by a lot to see if we can find whether some of the weights are not being used by using the plot function below 
class MyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, init_x : float = 0.0001) -> None:
        super().__init__(in_features,out_features,bias,device,dtype)
        with torch.no_grad():
            self.weight *= init_x
            if self.bias is not None:
                self.bias *= init_x

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.last_forward_out = super().forward(input=input)
        return self.last_forward_out

class CharEncoder:
    def __init__(self,text=None):
        if text is not None:
            self.chars = sorted(list(set(text)))
            self.n_tokens = len(self.chars)
            stoim = { ch:i for i,ch in enumerate(self.chars)}
            itosm = { i:ch for i,ch in enumerate(self.chars)}
            self.encode = lambda s : [stoim[c] for c in s]
            self.decode = lambda arr : ''.join([itosm[i] for i in arr])
        else: #ascii mode
            self.encode = lambda s : [min(ord(c),127) for c in s]
            self.decode = lambda arr : ''.join([chr(i) for i in arr])
            self.n_tokens = 128
            
            
def read_file(filename,device,enc=None,print_sample=False):
    with open(filename,'r') as f:
        text = f.read()
        if(print_sample):
            print(text[:1000])
        
        if(enc is None):
            enc = CharEncoder(text)
        data = torch.tensor(enc.encode(text),device=device)
        n = int(0.9*len(data))
        train_data = data[:n]
        val_data = data[n:]
        
        return enc,train_data,val_data


def get_batch(split,train_data,val_data,batch_size,block_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y
    
class Head(nn.Module):
    def __init__(self,n_embd,block_size,head_size,dropout,device):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size, bias=False,device=device)
        self.value = nn.Linear(n_embd,head_size, bias=False,device=device)
        self.query = nn.Linear(n_embd,head_size, bias=False,device=device)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size,device=device)))

        if(dropout is not None):
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x) # --> (B,T,head_size)
        q = self.query(x) # --> (B,T,head_size)
        
        wei = q @ k.transpose(-2,-1) * C ** -0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0.0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        if(self.dropout is not None):
            wei = self.dropout(wei)
        v = self.value(x) # --> (B,T,head_size)
        return wei @ v # (B,T,head_size)

    def generate(self,idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            logits,loss = self(idx)
            logits = logits[-1,:]
            probs= F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next.view(1,-1)),dim =1)
        return idx
        


class MultiHead(nn.Module):
    def __init__(self,n_embd,block_size,num_heads, head_size,dropout,device):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd,block_size,head_size,dropout,device) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd,device=device)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1) # h[x] returns (B,T,head_size) and we cat the result together for each input value (token in first level)
        out = self.proj(out)
        
        return out
    


class FeedForward(nn.Module):
    def __init__(self, n_embd,dropout,device):
        super().__init__()
        self.net = nn.Sequential(
                       nn.Linear(n_embd,4*n_embd,device=device), # `4*` because the inner layer of the feed forward network should be 4x
                       nn.ReLU(),
                       nn.Linear(4*n_embd,n_embd,device=device))
        if(dropout is not None):
            self.net.append(nn.Dropout(dropout))
    def forward(self,x):
        return self.net(x)



class Block(nn.Module):
    def __init__(self, n_embd, block_size, n_head, head_size, dropout, device, use_layernorm=True,use_residual=True):
        super().__init__()
        
        self.sa = MultiHead(n_embd,block_size,n_head,head_size,dropout,device)
        self.ffwd = FeedForward(n_embd,dropout,device=device)
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        if(self.use_layernorm):
            self.ln1 = nn.LayerNorm(n_embd,device=device)
            self.ln2 = nn.LayerNorm(n_embd,device=device)
    
    def forward(self, x):
        if(self.use_layernorm):
            if(self.use_residual):
                x = x+self.sa(self.ln1(x)) # `x+` adds residual connection
                x = x+self.ffwd(self.ln2(x))
            else:
                x = self.sa(self.ln1(x)) # `x+` adds residual connection
                x = self.ffwd(self.ln2(x))
        else:
            if(self.use_residual):
                x = x+self.sa + self.ffwd(x)
            else:
                x = self.sa + self.ffwd(x)
        
        return x
    


class AttentionModel(nn.Module):
    def __init__(self,vocab_size,block_size,num_heads,head_size,dropout,n_layer,device,use_layernorm=True):
        super().__init__()
        n_embd = num_heads * head_size
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd,device=device) # (B,T,C)
        self.position_embedding_table = nn.Embedding(block_size,n_embd,device=device)
        self.blocks = nn.Sequential(*[
            Block(n_embd, block_size, num_heads, head_size, dropout,device=device) for _ in range(n_layer)])
        self.use_layernorm = use_layernorm
        if(self.use_layernorm):
            self.ln_f = nn.LayerNorm(n_embd, device=device)

        self.lm_head = nn.Linear(n_embd,vocab_size,device=device) # (B,T,vocab_size)
        
        self.block_size=block_size
        self.device=device
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        if(self.use_layernorm):
            x = self.ln_f(x)
        logits= self.lm_head(x)
        
        if(targets is None):
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(-1,C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits,targets)

        return logits,loss

    def generate(self,idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.block_size:]
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs= F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next.view(1,-1)),dim =1)
        return idx
            

@torch.no_grad()
def estimate_loss(model,train_data,val_data,eval_iters,batch_size,block_size):
    out = {}
    model.eval() #changes to "eval" mode
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split,train_data,val_data,batch_size,block_size)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train() #return to 'train' mode
    
    return out

            

@torch.no_grad()
def generate_text(model,enc,max_items):
    res = model.generate(torch.zeros((1,1),dtype=torch.long,device=model.device),max_items)
    return enc.decode(res.view(-1).tolist())
    

import signal
import logging
import time

class DelayedKeyboardInterrupt:

    def __init__(self,on_interrupt_fn=None,eat_interrupt=False):
        self.on_interrupt_fn = on_interrupt_fn
        self.eat_interrupt = eat_interrupt

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
                
    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')
    
    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            print('KeyboardInterrupt delayed')
            if(self.on_interrupt_fn is not None):
                self.on_interrupt_fn()
            if(not self.eat_interrupt):
                self.old_handler(*self.signal_received)

def dki_example():
    with DelayedKeyboardInterrupt():
        # stuff here will not be interrupted by SIGINT
        print("start critical")
        time.sleep(5)
        print("end critical")

    time.sleep(0.2)
    print("stopped yet?")
    time.sleep(5)
    print("should have by now!")


def train(max_iters,train_data,val_data,batch_size,block_size,m,optimizer,eval_iters,eval_interval):
    print(f'Training up to {max_iters}, to stop early, interrupt with Ctrl-C. After interrupt the model will NOT be corrupted')
    for steps in range(max_iters):
        with DelayedKeyboardInterrupt():
            #print(f'{steps=}')
            xb, yb = get_batch('train',train_data,val_data,batch_size,block_size)
            #print(f'{xb.shape=},{yb.shape=}')
            logits,loss = m(xb,yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if(steps%eval_interval==0):
                est_loss = estimate_loss(m,train_data,val_data,eval_iters,batch_size,block_size)
                print(f'{steps=},{est_loss=}')

    print(f'ran for {steps}, est_loss={estimate_loss(m,train_data,val_data,eval_iters,batch_size,block_size)}')

import subprocess

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def git_is_dirty() -> str:
    out = subprocess.check_output(['git', 'status']).decode('ascii').strip()
    #print(out)
    r = re.match('On branch [^\n]+\nnothing to commit, working tree clean',out)
    return (r is None),out

def _get_or_update_submodule(model,key,update_fn=None):
    if isinstance(model, nn.ModuleList) and isinstance(key, int) or
       isinstance(model, nn.ModuleDict) and isinstance(key, str):
        ret_sub_model = model[key]
        if(update_fn is not None):
            if(ret_sub_model is None):
                return None
            model[key] = update_fn(ret_sub_model)
        return ret_sub_model
    else:
        ret_sub_module = getattr(model, key)
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

