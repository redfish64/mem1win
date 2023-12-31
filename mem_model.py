"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
import JacModule as jm
import util as u


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias,device='cpu'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim,device=device))
        self.bias = nn.Parameter(torch.zeros(ndim,device=device)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, mem, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn =     jm.JacLinear(config.n_embd, 3 * config.n_embd, requires_grad=True, has_bias=config.bias,device=config.device)
        # output projection
        self.c_proj =     jm.JacLinear(config.n_embd, config.n_embd, requires_grad=True, has_bias=config.bias,device=config.device)

        # TOOD 3 we are setting requires_grad to True, because if its false, its not picked up by the optimizer
        # the optimizer is also responsible for zeroing out the grads after each cycle which is necessary or the
        # mem grads will blow up.
        # Not sure if it honestly should be true or not. If it is true, than the straight param to loss (without
        # going through memory) pathway will be added in. But this should really only affect memory, I guess, so
        # dunno?
        self.c_mem_attn = jm.JacLinear(config.n_mem_embd, 3 * config.n_embd, requires_grad=True, has_bias=config.bias,device=config.device)
        self.c_mem_proj = jm.JacLinear(config.n_embd, config.n_mem_embd, requires_grad=True, has_bias=config.bias,device=config.device)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_mem_embd = config.n_mem_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = (not config.no_flash) and hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        total_size = config.block_size + config.n_memory_per_layer
        if not self.flash:
            if(config.no_flash):
                print("Flash attention turned off")

            else:
                print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
                
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("no_flash_attn_mask",
                                 torch.tril(torch.ones(config.block_size,total_size,device=config.device),diagonal=config.n_memory_per_layer)
                                 .view(1, 1, config.block_size,total_size))
        else:
            self.register_buffer("attn_mask",
                                 torch.tril(torch.ones(config.block_size,total_size,device=config.device),diagonal=config.n_memory_per_layer)
                                 .view(1, 1, config.block_size,total_size)) == 1

    def get_mem_params(self):
        return (self.c_mem_attn.get_jac_params() +
                self.c_mem_proj.get_jac_params())
        
    def get_out_params(self):
        return (self.c_attn.get_jac_params() +
                self.c_proj.get_jac_params())
        
    def forward(self, mem, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        _,mT,mC = mem.size()

        #PERF we are calculating mq here but not using it.
        _, mk, mv = self.c_mem_attn(mem[0:B]).split(self.n_embd, dim=2)
        mk = mk.view(B, mT, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        mv = mv.view(B, mT, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        fk = torch.cat((mk,k),dim=2)
        fq = q # we don't calculate the query value for the memory, because that doesn't affect the output
        # and is only used for the memory change 
        fv = torch.cat((mv,v),dim=2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(fq, fk, fv, attn_mask=self.attn_mask, dropout_p=self.dropout if self.training else 0)
        else:
            # manual implementation of attention
            att = (fq @ fk.transpose(-2, -1)) * (1.0 / math.sqrt(fk.size(-1)))
            att = att.masked_fill(self.no_flash_attn_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ fv # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y

    def _calc_mem_out(self,mem,x):

        B, T, C = x.size() # batch size of input (for jacobian this will always be one regardless of actualy batch size)
        #, sequence length, embedding dimensionality (n_embd)

        _,mT,mC = mem.size() 
        
        
        #this is a copy of forward. The only changes are that we calculate q from the memory only, instead of
        #the input only in the forward method, and also we use no mask.

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        #PERF we are calculating q here but not using it
        _, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        mq, mk, mv = self.c_mem_attn(mem).split(self.n_embd, dim=2)
        #TODO 2 only batch size of 1 works right now, because jacrev only provides input with a single item in the
        #batch dimension, and mem will contain all batches
        mk = mk.view(B, mT, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        mq = mq.view(B, mT, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        mv = mv.view(B, mT, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        fk = torch.cat((mk,k),dim=2)
        fq = mq # we don't calculate the query value for the input, because that doesn't affect the memory
        fv = torch.cat((mv,v),dim=2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(fq, fk, fv, dropout_p=self.dropout if self.training else 0)
        else:
            # manual implementation of attention
            att = (fq @ fk.transpose(-2, -1)) * (1.0 / math.sqrt(fk.size(-1)))
            #there is no mask for _calc_mem_out because we are dealing with memory grad changes which will
            #take effect next cycle.
            att = F.softmax(att, dim=-1)

            #TODO 3 is dropout in mem a good idea?
            att = self.attn_dropout(att)
            y = att @ fv # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, mem.shape[1], C) # re-assemble all head outputs side by side

        # output projection
        new_mem = self.c_mem_proj(y)

        return new_mem
       


        
class MLP(nn.Module):

    def __init__(self, config, n_embd,requires_grad):
        super().__init__()
        self.c_fc    = jm.JacLinear(n_embd, 4 * n_embd, requires_grad, has_bias=config.bias,device=config.device)
        self.gelu    = nn.GELU()
        self.c_proj  = jm.JacLinear(4 * n_embd, n_embd, requires_grad, has_bias=config.bias,device=config.device)
        self.dropout = nn.Dropout(config.dropout)

    def get_jac_params(self):
        return self.c_fc.get_jac_params() + self.c_proj.get_jac_params()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.register_buffer('memory',torch.zeros((config.batch_size,
                                                   config.n_memory_per_layer,
                                                   config.n_mem_embd),
                                                  device=config.device,requires_grad=True))

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias,device=config.device)
        self.attn = CausalSelfAttention(self.memory,config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias,device=config.device)
        #TODO 3 do we really want layernorm for mem?
        #we don't use elementwise_affine because we don't want any parameters. If
        #we have parameters we need to make them compatible with JacModule, such as with
        #JacLinear. TODO 3 maybe make a JacLayerNorm with parameters?
        self.mem_ln = torch.nn.LayerNorm(config.n_mem_embd, elementwise_affine=False,device=config.device)
        self.mlp = MLP(config,config.n_embd,True)
        self.mem_mlp = MLP(config,config.n_mem_embd,False)
        self.jac_grad_decay = config.jac_grad_decay
        self.batch_size = config.batch_size
        self.n_memory_per_layer = config.n_memory_per_layer
        self.n_mem_embd = config.n_mem_embd
        self.mem_grad_multiplier = torch.tensor(config.mem_grad_multiplier,device=config.device)
        self.last_grads = None

        def loop_fn(x):
            m = self.attn._calc_mem_out(self.mem_ln(jm.get_jac_param(self.memory)),self.ln_1(x))
            m = self.mem_ln(self.mem_mlp(self.mem_ln(m)))
            return m

        self.snake = jm.Snake(loop_fn,self.get_mem_params(), self.memory,[],config.jac_grad_decay,device=config.device,vmap_randomness='different')

    def forward(self, x):
        x = x + self.attn(jm.get_jac_param(self.memory),self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def get_mem_params(self):
        return (self.attn.get_mem_params() +
                #co: no parameters for LayerNorm without elementwise_affine
                #self.mem_ln_1.get_jac_params() + 
                #self.mem_ln_2.get_jac_params() +
                self.mem_mlp.get_jac_params())
        
    def get_out_params(self):
        return (self.attn.get_out_params() +
                self.mlp.get_jac_params())
        
    def update_memory_and_mem_params(self,x):
        """
        Must be called only after forward() is called and backward() so that memory.grad is populated
        Also sets memory.grad to None
        x - x is the previous input used for forward()
        """
        if(self.last_grads is not None):
            self.snake.update_param_grads(self.last_grads,self.memory.grad)

        self.last_grads, next_mem = self.snake.run_jacobian(x)

        jac_params = self.get_mem_params() + [self.memory]

        if(self.mem_grad_multiplier != 1.):
            for jp in jac_params:
                jp.grad *= self.mem_grad_multiplier 
        
        self.memory.copy_(next_mem)

        #the memory grad should have been calculated by backwards, and we used it above to calculate the jp.grads, but
        #we don't want to actually train it, since it doesn't represent an actual static parameter.
        #(maybe shouldn't do this here, it's weird, but where else?)
        self.memory.grad = None

    def save_and_reset_memory(self):
        """Returns current memory for running estimated loss, etc. and creates a clone all zeroed out """
        old_mem = self.memory
        self.memory = torch.zeros_like(self.memory,requires_grad=True,device=old_mem.device)
        return old_mem

    def reset_memory(self):
        with torch.no_grad():
            self.memory *= 0.

        
    def restore_memory(self,mem):
        self.memory = mem

    def reset_memory_for_item_in_batch(self,index):
        with torch.no_grad():
            self.memory[index].zero_()

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_mem_embd: int = 256
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    batch_size: int = 8
    n_memory_per_layer: int = 8
    jac_grad_decay: float = 0.01
    no_flash: bool = False 
    mem_grad_multiplier: float = 1.0
    device : str = 'cpu'


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd,device=config.device),
            wpe = nn.Embedding(config.block_size, config.n_embd,device=config.device),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias,device=config.device),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False,device=config.device)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_mem_params(self):
        return [hp for h in self.transformer.h for hp in h.get_mem_params()]
        
    def get_out_params(self):
        return [hp for h in self.transformer.h for hp in h.get_out_params()]

    def save_and_reset_memory(self):
        return [b.save_and_reset_memory() for b in self.transformer.h]

    def reset_memory(self):
        for b in self.transformer.h:
            b.reset_memory()

    def restore_memory(self,saved_memory):
        for s,b in zip(saved_memory,self.transformer.h):
            b.restore_memory(s)

    def reset_memory_for_item_in_batch(self,index):
        for b in self.transformer.h:
            b.reset_memory_for_item_in_batch(index)

        
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None,compute_last_loss=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        self.last_block_xs = []
        for block in self.transformer.h:
            self.last_block_xs.append(x)
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if(compute_last_loss):
                last_item_loss = F.cross_entropy(logits[:,-1,:].view(-1, logits.size(-1)), targets[:,-1].view(-1), ignore_index=-1)
            else:
                last_item_loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            last_item_loss = None

        return logits, loss, last_item_loss

    def update_memory_and_mem_params(self,x):
        with torch.no_grad():
            for block,x in zip(self.transformer.h,self.last_block_xs):
                block.update_memory_and_mem_params(x)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


