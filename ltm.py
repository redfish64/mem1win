import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import logging.config
import argparse
import glob
import tiktoken
import time
from typing import NamedTuple
import re
import os
import pdb
import matplotlib.pyplot as plt

# the reason I copied util into this git tree is so that there is a consistent commit. When we run each
# session, I want to record the git state along with all parameters, so that I can reproduce/track the results
import util
import gutenberg


class ModelTrailingAvgStats:
    def __init__(self):
        self.u_block_pos = None
        self.u_max_block_pos = None
        self.loss = None
        self.query_loss = None

    def update_ta(self, perc, name, val):
        if self.__dict__[name] is None:
            self.__dict__[name] = val
        else:
            self.__dict__[name] = val * perc + self.__dict__[name] * (1.0-perc)

class ModelStats:
    def __init__(self):
        #last time print_status printed a message
        self.last_print_time = 0.0
        self.last_save_time = 0.0
        self.n_batches_trained = 0
        self.n_works_read = 0
        self.tas = ModelTrailingAvgStats()
 
stats = ModelStats()

# trailing_avg_perc: amount of the past to keep, so should be closer to 0.99 than 0.01
def print_status(ms,time_between_updates, trailing_avg_perc, loss=None, query_loss=None):
    #yes we use global variables here, but this is just for displaying status
    global stats
    #last_print_time, trailing_avg_loss, trailing_avg_query_loss, n_batches_trained

    curr_time = time.time()
    stats.n_batches_trained += 1

    # stats.tas.update_ta(trailing_avg_perc,'u_block_pos', u_block_pos)
    # stats.tas.update_ta(trailing_avg_perc,'u_max_block_pos', u_max_block_pos)
    stats.tas.update_ta(trailing_avg_perc,'loss', loss.item())
    stats.tas.update_ta(trailing_avg_perc,'query_loss', query_loss.item())

    if(curr_time - stats.last_print_time < time_between_updates):
        return

    stats.last_print_time = curr_time

    
    print(f'works_read: {stats.n_works_read}, batches_trained: {stats.n_batches_trained:8} loss: {stats.tas.loss:8.4f} query_loss: {stats.tas.query_loss:8.4f}')

#Long term transformer head. Keeps some memory around using a buffer for the next round and another vector
#
# The term "work" in this file refers to an individual piece of text, whether a story, an article, an email thread, etc.
# Any time looking at past data makes sense.

# WARNING raw within batch must be processed sequentially. Do not randomly sample from a large text.
# TODO 3 Currently only works with block sizes of one.
#
# block_2p_pos_embedding_table - since memory that are stored come from a previous block and there is no
#  inherit positional encoding, we want to give a model a chance to choose an encoding. Also since a single work
#  could be very long, the encoding is based on the 2^ of the block index. So if we're on block 10 and an old
#  value is at block 5, it would be found in position 3 (2^3)
class LTMHead(nn.Module):
    def __init__(self,n_embd,block_sqr_pos_embedding_table,batch_size,block_size,head_size,dropout,n_memory,device,
                 lq_add,decay):
        super().__init__()
        
        self.block_sqr_pos_embedding_table = block_sqr_pos_embedding_table
        self.key = nn.Linear(n_embd,head_size, bias=False,device=device)
        self.value = nn.Linear(n_embd,head_size, bias=False,device=device)
        self.query = nn.Linear(n_embd,head_size, bias=False,device=device)
        self.device = device
        self.head_size = head_size
        self.n_embd = n_embd

        self.lq_add = lq_add
        self.decay = decay

        # for memory,we impose a second block based positional encoding over and above
        # the token positional encoding at the start of the model. This allows the model
        # to know how old the token is.  In order to save space, we use a log2 based
        # distribution, which means that each index older is twice the length of the
        # previous.
        self.n_memory = n_memory
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        self.register_buffer('memory',torch.zeros((batch_size,n_memory,n_embd),dtype=torch.float,device=device))

        # distance in blocks the old value was seen compared to currently run block
        self.register_buffer('memory_block_dist',torch.zeros((batch_size,n_memory),dtype=torch.int32,device=device))

        # the sum of query * key for the memory with a decay. used to sort and remove memory not used anymore
        self.register_buffer('memory_rank',torch.zeros((batch_size,n_memory),dtype=torch.float,device=device))

        # tril mask - memory never get a chance to query, so all we need to do is prepend a tril with a bunch
        # of false masks in the shape (block_size, n_memory + block_size)
        # ex. assuming batch size = 1, block_size = 3 and n_memory = 2
        #         [[ 1.,  1.,  1.,  0.,  0.],
        #          [ 1.,  1.,  1.,  1.,  0.],
        #          [ 1.,  1.,  1.,  1.,  1.]]])

        if(dropout is not None):
            self.dropout = nn.Dropout(dropout)

    # resets the memory and the batch size for the model
    def reset_for_new_works(self,new_batch_size):
        self.batch_size = new_batch_size
        self.memory = torch.zeros((self.batch_size,self.n_memory,self.n_embd),dtype=torch.float,device=self.device)
        self.memory_block_dist = torch.zeros((self.batch_size,self.n_memory),dtype=torch.int32,device=self.device)
        self.memory_rank = torch.zeros((self.batch_size,self.n_memory),dtype=torch.float,device=self.device)

    # we expect the training example and a running loss for the u vectors. We will add to the query_loss our contribution
    # when returning
    # block_pos_list - block pos of each work in batch

    # mask - masks wei for any value that is true, should be a boolean tensor in the shape
    # (batch_size, n_memory+block_size). Note that if you want block_size to be greater than one, this should
    # be a tril mask at least. It can also be used for our split and find the best/worst memory algorithm
    # update_memory - if true, memory will be updated according to u
    # u_target - the usefulness target to train u against. If None, then query_loss won't be returned
    def forward(self, block_pos_list, inp, mask=None,update_memory=False, u_target=None):

        # if(stats.n_batches_trained % 100 == 99):
        #     pdb.set_trace()

        #clear out memory when we start a new work, so the model isn't "remembering" data from prior works
        #PERF a little slow maybe. We could put in in a tensor I guess
        for ix,block_pos in enumerate(block_pos_list):
            if(block_pos == 0):
                self.memory[ix] = 0.
                self.memory_block_dist[ix] = 0

        #we add to the block dist, aging the old blocks, since we are now going to add the current
        #batch into memory
        #Note that by adding it here at the top of the function, the minimum value is effectively 1.
        # This makes it so the log2 is zero and matches with the first position in block_sqr_pos_embedding_table
        self.memory_block_dist += 1

        m_pos_ix = torch.log2(self.memory_block_dist).int()
        m_pos_emb = self.block_sqr_pos_embedding_table(m_pos_ix)

        m_with_emb = self.memory + m_pos_emb

        m_with_emb_and_inp = torch.cat((m_with_emb,inp),dim=1)

        k = self.key(m_with_emb_and_inp) # --> (B,n_memory+block_size,head_size)
        q = self.query(inp) # --> (B,block_size,head_size)

        # wei:
        # -- ex. assuming batch size = 1, block_size = 3 and n_memory = 2
        # --         keys go right (dim 2)-->
        # --         querys go down (dim 1)
        # --         |
        # --         |        +---cutoff between old keys and new keys is here
        # --         \/      \/
        # tensor([[[ 0.,  0.,  0.,  0.,  0.],  <-- ex. last entry is k[0,4] * q[0,0] (where first index is the batch)
        #          [ 3.,  6.,  3.,  3.,  3.],
        #      ------  cut off between old value queries and block queries -----
        #          [ 3.,  6.,  3.,  3.,  3.]]
        #          [ 3.,  6.,  3.,  3.,  3.]]
        #          [ 3.,  6.,  3.,  3.,  3.]]


        mbs = self.n_memory+self.block_size
        wei = q @ k.transpose(-2,-1) * mbs ** -0.5 # --> (B,n_memory+block_size,block_size), `* C ** -0.5` is to scale the wei to be smaller so the softmax is more smooth, as taken from "Attention is all you need" Section 3.2.1 Scaled Dot-Product Attention
        if(mask is not None):
            wei = wei.masked_fill(self.mask,float('-inf')) # do the mask fill

        # TODO 4 try using an abs() for loss. This way we can have a negative key (and negate
        # the value) instead of everything being positive. 
        wei = wei * wei # The idea is to make orthanganol vectors have a result that is
                        # near zero, all others should be postive.
        wei = wei / wei.max(dim=2,keepdim=True).values

        #co: we don't handle dropout right now
        # if(self.dropout is not None):
        #     wei = self.dropout(wei)
        v = self.value(m_with_emb_and_inp) # --> (B,n_memory+block_size,head_size)

        ## choose which memory to keep in the next block

        #self.memory_rank = (self.memory_rank + wei[:,:,0:-self.block_size].sum(dim=1) * self.memory_block_dist) * (1. - self.decay)
        mean_wei = wei.sum(dim=1).mean(dim=1)
        self.memory_rank = (self.memory_rank + wei[:,:,0:-self.block_size].sum(dim=1)/mean_wei.unsqueeze(dim=1) - 1.) - self.decay
        #self.memory_rank = -self.memory_block_dist

        #sort by which ones the model thinks is most valuable
        mr_srt,mr_idx = self.memory_rank.sort(dim=1,descending=True)

        #this is used to help rearrange the other tensors according to usefulness sort
        batch_indices = torch.arange(self.batch_size,device=self.device).unsqueeze(1).expand(-1,self.n_memory)

        #batch_indices is B,T
        #mr_idx is B,T
        #mr_srt is B,T,C

        #memory sorted by useful. This will
        #become the new memory note that we do not include block level positional
        #encoding here, and copy the memory as they were.
        mem_sorted_by_mr = self.memory[batch_indices,mr_idx]

        #memory_block_dist is B,T
        #u_idx is B,T
        #this will rearrange memory_block_dist, so that it's sorted according to useful
        #   memory_block_dist_sorted_by_useful[i,j] = memory_block_dist[batch_indices[i,j],u_idx[i,j]] for all i,j
        memory_block_dist_sorted_by_mr = self.memory_block_dist[batch_indices, mr_idx]

        memory_rank_sorted_by_mr = self.memory_rank[batch_indices, mr_idx]

        #replace the lowest ranked memory(s) with the relavant block info
        #PERF I don't think we need to clone here
        #TODO 3 make [1.0] a parameter for initial rank
        replace_size = min(self.block_size,self.n_memory)
        self.memory = mem_sorted_by_mr.clone().detach()
        self.memory[:,-replace_size:] = inp[:,-replace_size:].clone().detach()
        self.memory_block_dist = memory_block_dist_sorted_by_mr.clone().detach()
        self.memory_block_dist[:,-replace_size:] = 0
        self.memory_rank = memory_rank_sorted_by_mr.clone().detach()
        self.memory_rank[:,-replace_size:] = 1

        qt_loss = (wei + self.lq_add).log().sum()

        #return the result and the usefulness loss. The usefulness loss must be saved by the rest of the
        #model to be trained in addition to the main training
        return wei @ v,qt_loss # (B,T,head_size), single_value


    def generate(self,idx, max_new_tokens):
        #TODO 1 old placeholder code here
        for _ in range(max_new_tokens):
            logits,loss = self(idx)
            logits = logits[-1,:]
            probs= F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next.view(1,-1)),dim =1)
        return idx
        
class LTMMultiHead(nn.Module):
    def __init__(self,n_embd,block_sqr_pos_embedding_table, batch_size, block_size,num_heads, head_size,dropout,n_memory,device,lq_add,decay):
        super().__init__()
        self.heads = nn.ModuleList([LTMHead(n_embd,block_sqr_pos_embedding_table, batch_size, block_size,head_size,dropout,n_memory,device,lq_add,decay) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd,device=device)
    
    def reset_for_new_works(self,new_batch_size):
        for h in self.heads:
            h.reset_for_new_works(new_batch_size)

    def forward(self, block_pos_list, x):
        outs = []

        total_query_loss = 0

        for h in self.heads:
            h_out, query_loss = h(block_pos_list,x)
            outs.append(h_out)
            total_query_loss += query_loss

        out = torch.cat(outs,dim=-1)
        out = self.proj(out)

        return out,total_query_loss
    

class LTMBlock(nn.Module):
    def __init__(self, n_embd, block_sqr_pos_embedding_table, batch_size, block_size, n_head, head_size, dropout, n_memory, device,lq_add,decay):
        super().__init__()

        self.sa = LTMMultiHead(n_embd, block_sqr_pos_embedding_table, batch_size, block_size,n_head,head_size,dropout,n_memory,device,lq_add,decay)
        self.ffwd = util.FeedForward(n_embd,dropout,device=device)
        self.ln1 = nn.LayerNorm(n_embd,device=device)
        self.ln2 = nn.LayerNorm(n_embd,device=device)
        self.n_memory = n_memory
        self.n_embd = n_embd
        self.block_size = block_size
        self.batch_size = batch_size
    
    def reset_for_new_works(self,new_batch_size):
        self.sa.reset_for_new_works(new_batch_size)

    def forward(self, tpl):
        block_pos_list, x, total_query_loss = tpl
        sa,query_loss = self.sa(block_pos_list,self.ln1(x))

        total_query_loss += query_loss
        x = x+sa # `x+` adds residual connection
        x = x+self.ffwd(self.ln2(x))
        return block_pos_list,x,total_query_loss
    
class LTMAttentionModel(nn.Module):
    def __init__(self,vocab_size,batch_size, block_size,num_heads,head_size,dropout,n_layer,n_memory,device,lq_add,decay):
        super().__init__()
        n_embd = num_heads * head_size
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd,device=device) # (B,T,C)
        self.position_embedding_table = nn.Embedding(block_size,n_embd,device=device)
        block_square_pos_embedding_table = nn.Embedding(32,n_embd,device=device) #TODO 3 hardcoding embedding table size 
        self.blocks = nn.Sequential(*[
            LTMBlock(n_embd, block_square_pos_embedding_table, batch_size, block_size, num_heads, head_size, dropout,n_memory,device,lq_add,decay) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, device=device)

        self.lm_head = nn.Linear(n_embd,vocab_size,device=device) # (B,T,vocab_size)
        
        self.block_size=block_size
        self.device=device

    def reset_for_new_works(self,new_batch_size):
        for b in self.blocks:
            b.reset_for_new_works(new_batch_size)
            
    # block_pos_list: this is the position in each work that is being loaded. It's needed to be passed in so
    # that the Heads can reset their memory when a new work is started.
    def forward(self, block_pos_list, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        block_pos_list,x,query_loss = self.blocks((block_pos_list,x,0.))
        x = self.ln_f(x)
        logits= self.lm_head(x)
        
        if(targets is None):
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(-1,C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits,targets)

        return logits,loss,query_loss

    def generate(self,idx, max_new_tokens):
        #TODO 1 old code
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.block_size:]
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs= F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next.view(1,-1)),dim =1)
        return idx
            

def read_next_valid_work(works_read_set,iter,min_size=0,max_size=None):
    for zf in iter:
        logging.debug(f'Looking at {zf}')
        if(zf in works_read_set):
            logging.info(f'Skipping {zf} since already read')
            continue

        data,fn,error = gutenberg.get_etext(zf,min_size,max_size)
        if(data is None):
            logging.info(f'Skipping {zf}: {error}')
            continue

        logging.info(f'Reading {zf}, {fn}: {data[:100]}...')

        return data,zf,fn
    return None,None,None

class CurrentWork():
    #data: the text of the work
    #zf: the zip file name
    #fn: the file name
    def __init__(self,data,zf,fn):
        self.data = data
        self.zf = zf
        self.fn = fn
        self.block_pos = 0

#loads batches of works in sequential order, so that each block in a batch follows from the previous block in that
#indexed position.  This is important for the LTM model, which needs to see the previous block to predict the next block.
#
# iter_fn: a function that returns an iterator over the works to read. It can be called more than once for multiple runs
# works_read_set: This is used to skip works that have already been read
# num_loops: how many times to loop over the works
#
# TODO 2.5 Right now if there aren't enough works to fill a batch and at the end of num_loops then will exit with none
#  rather than creating a short batch. This is harder than it seems because the LTM model keeps track of internal
#  buffers of "remembered" values, so we have to keep the works in the same position in the batch for the entire
#  story. So if a story before the last one finishes, we won't be able to move the later ones to replace it.
class WorkLoaderIter(): #TODO 3 renamed to WorkLoader, not doing now so I can reuse old model files

    def __init__(self,enc,works_read_set,iter_fn,batch_size,block_size,num_loops,device, min_size=0,max_size=None):
        self.enc = enc
        self.works_read_set = works_read_set
        self.iter_fn = iter_fn
        self.batch_size = batch_size
        self.block_size = block_size
        self.min_size = min_size
        self.max_size = max_size
        self.iter = iter_fn()
        self.current_works = [None for _ in range(batch_size)]
        self.num_loops = num_loops
        self.curr_loop = 0
        self.device = device
 
    def read_next_work(self):
        global stats
        next_work,zf,fn = read_next_valid_work(self.works_read_set,self.iter,self.min_size,self.max_size)
        if(next_work is None):
            self.curr_loop += 1
            if(self.curr_loop == self.num_loops):
                logging.info('No more works, and at end of loops')
                return None

            stats.n_works_read = 0
            
            logging.info(f'At end of works, restarting iterator, loop {self.curr_loop} out of {self.num_loops}')
            self.works_read_set = {}
            self.iter = self.iter_fn()
            return self.read_next_work()

        stats.n_works_read += 1

        #TODO 2.8 set disallowed_special to set() to allow special tokens since we have dirty data. Need to
        # find out how special tokens are encoded and strip them from the input
        logging.debug(f'Encoding {zf}, {fn}, {next_work[:100]}...')
        return CurrentWork(torch.tensor(self.enc.encode(next_work),device=self.device),zf,fn)

    def __getstate__(self):
        new_dict = self.__dict__.copy()
        new_dict['enc'] = None
        new_dict['iter_fn'] = None
        new_dict['iter'] = None

        #rather than saving the tokens of all the works being processed into the pt file, we just save the filenames,
        #and then reload the data when we reload the model in init_after_load
        new_dict['current_works'] = None
        new_dict['current_work_zf'] = [cw.zf for cw in self.current_works]
        new_dict['current_work_bp'] = [cw.block_pos for cw in self.current_works]

        return new_dict

    # MUST be called after loading a previous save
    def init_after_load(self,enc,iter_fn):
        self.enc = enc
        self.iter_fn = iter_fn
        self.iter = iter_fn()

        #reload current works into model (which we removed when saving to save space)
        self.current_works = []
        for i in range(len(self.current_work_zf)):
            zf = self.current_work_zf[i]
            bp = self.current_work_bp[i]
            data,fn,error = gutenberg.get_etext(zf)
            if(data is None):
                raise Exception('Couldn\'t read work that was in process of being read, {zf}: {error}')
            cw = CurrentWork(torch.tensor(self.enc.encode(data),device=self.device),zf,fn)
            cw.block_pos = bp
            self.current_works.append(cw)

        #get rid of vars only used for saving, since we replaced them with self.current_works
        self.current_work_zf = None
        self.current_work_bp = None
            
    def __iter__(self):
        return self
        
    def __next__(self):
        batch_list = []
        target_list = []

        for i in range(self.batch_size):
            cw = self.current_works[i]
            token_pos = 0 if cw is None else cw.block_pos * self.block_size

            #if we're near enough to the end of the current work to not be able to get a full block, then
            #read the next work. We add +1 so that the targets are also loadable (targets are symbol + 1)
            #TODO 2.8 we might want to read partial blocks for the end but for now we'll keep it simple
            if(cw is None or token_pos + self.block_size + 1>= len(cw.data)):
                cw = self.read_next_work()
                if(cw is None): #if no more works to fill the batch then we're done
                    raise StopIteration
                #we record that the previous file was read so if we save and reload the model, we won't
                #read it again. Note that since we are saving the current_works with the model when we
                #save, we don't have to worry that when we reload we would skip the current file, since
                #it will be loaded when we load the model.
                self.works_read_set[cw.zf] = True
                self.current_works[i] = cw
                token_pos = 0

            block = cw.data[token_pos:token_pos+self.block_size]
            target = cw.data[token_pos+1:token_pos+self.block_size+1]
            self.current_works[i].block_pos += 1
            batch_list.append(block)
            target_list.append(target)

        return ([cw.block_pos for cw in self.current_works],
                torch.stack(batch_list,dim=0),
                torch.stack(target_list,dim=0),
                )

    #TODO 2 get pickling working. Shouldn't be too hard, we just store works_read_set current_works_block_pos and
    # current work filenames

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [DIR]...",
        description="runs long term memory transformer against a directory tree of zipped text files",
        exit_on_error=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument('--dir_tree',required=True,help='directory tree to look for zipped text files for training')
    shared_parser.add_argument('--bos', action='store_true', help='if set, will drop into a debugger on start')
    

    train_parser = subparsers.add_parser("train", help="Train a new or existing model", parents=[shared_parser])
    train_parser.add_argument('--batch_size', nargs='?', type=int,default=32,help='num of batches to process at once')
    train_parser.add_argument('--block_size', nargs='?', type=int,default=1,help='block size in tokens. This along with memory_size decides the number of input params')
    train_parser.add_argument('--memory_size', nargs='?', type=int,default=256,help='memory for the model to "remember" while its guessing the next token')
    train_parser.add_argument('--n_loops', nargs='?', type=int,default=1,help='number of times to loop through data')
    train_parser.add_argument('--n_heads', nargs='?', type=int,default=6,help='number of heads per layer')
    train_parser.add_argument('--head_size', nargs='?', type=int,default=8,help='size of each head')
    train_parser.add_argument('--dropout', nargs='?', type=float,default=0.2,help='dropout percentage')
    train_parser.add_argument('--n_layer', nargs='?', type=int,default=6,help='number of layers')
    train_parser.add_argument('--min_text_size', nargs='?', type=int,default=10000,help='min size of text files to process (others will be skipped)')
    train_parser.add_argument('--max_text_size', nargs='?', type=int,default=1024*1024*10,help='max size of text files to process (others will be skipped)')
    train_parser.add_argument('--decay', nargs='?', type=float,default=0.01,help="This decays the rank of each memory unit by a constant amount (subtracted). It is used so that memory that wasn't used in a long time can be eventually forgotten")
    train_parser.add_argument('--query_loss_add', nargs='?', type=float,default=1,help='amount to add to the query loss before a log() is done to wei. wei ranges from 0-1. A higher value here will lower the aggressiveness used to make wei near zero. A lot of wei values near zero is beneficial because a zero wei means that the model isnt dependent on a key and therefore can more likely be dropped from memory without a bad effect')
    train_parser.add_argument('--trailing_avg_perc', nargs='?', type=float,default=0.99,help='only used for display. When printing status, we use a trailing avg. This is the percentage to keep from the previous runs, the remainder is for the current one.')
    train_parser.add_argument('--device', nargs='?', default="cpu",help='device to run on (cpu or cuda)')
    train_parser.add_argument('--test_dir_tree',nargs='?',help='directory tree to look for zipped text files for testing')
    train_parser.add_argument('--print_status', nargs='?',type=float,default=30,help='time between printing status updates and estimating loss in seconds')
    train_parser.add_argument('--save_model_time', nargs='?',type=float,default=60*15,help='time between saving the model in seconds')
    train_parser.add_argument('--no_save', action='store_true',help='won\'t save the model automatically. The model can still be saved by calling ')
    train_parser.add_argument('--learning_rate', nargs='?',type=float,default=3e-4,help='learning rate for optimizer')
    train_parser.add_argument('--load_model', nargs='?',help='load a previously generated model')
    train_parser.add_argument('--save_model_filename', nargs='?',help='name to save model after training (or interrupt)')
    train_parser.add_argument('--log_config_file', nargs='?',default='logging.conf',help='log config file, see: https://docs.python.org/3/howto/logging.html#configuring-logging')
    train_parser.add_argument('--query_loss_adj', nargs='?', type=float,default=0.1,help='relative speed to train query loss vs normal loss')

    imagine_parser = subparsers.add_parser("imagine", help="Imagine the rest of a story", parents=[shared_parser])
    imagine_parser.add_argument('--perc', nargs='?', type=float,default=0.5,help='amount of story to read before starting to imagine the rest')
    imagine_parser.add_argument('--load_model', required=True, help='load a previously generated model')
    imagine_parser.add_argument('--zip_file', required=True, help='zipped txt file containing story')

    return parser

class ModelState(NamedTuple):
    ltm: LTMAttentionModel
    wl: WorkLoaderIter
    optimizer: torch.optim.Optimizer
    args: argparse.Namespace

def create_files_iter_fn(args):
    return lambda: glob.iglob(f'{args.dir_tree}/**/*.zip',recursive=True)

def create_model_state(enc,args):

    #max_token_value is one less than the vocab size as far as I can tell, see:
    #https://github.com/openai/tiktoken/blob/7830ed537badecefb5a357448be722bfd58f9fca/tiktoken/core.py
    ltm = LTMAttentionModel(enc.max_token_value+1,args.batch_size, args.block_size,args.n_heads,args.head_size,args.dropout,args.n_layer,
                            args.memory_size,args.device,args.query_loss_add,args.decay)
    
    wl = WorkLoaderIter(enc,{},create_files_iter_fn(args),args.batch_size,args.block_size,args.n_loops,args.device,args.min_text_size,args.max_text_size)

    optimizer = torch.optim.AdamW(ltm.parameters(), lr=args.learning_rate)

    return ModelState(ltm,wl,optimizer,args)

def choose_new_filename(filename):
    if(filename is None):
        rev = util.get_git_revision_short_hash()
        dirty = "_dirty" if util.git_is_dirty() else ""
        filename = f"wr{stats.n_works_read}_bt{stats.n_batches_trained}_{rev}{dirty}.pt"

    #add '-1', '-2' etc. if filename already exists
    while(os.path.exists(filename)):
        m = re.match("^(.*?)(-(\d+))?(\..*)?$",filename)
        filename = f"{m.group(1)}-{int(m.group(3) or 0)+1}{m.group(4) or ''}"
    return filename

def load_model_state(fn,enc,args):
    (ms,stats) = torch.load(fn)
    ms.wl.init_after_load(enc,create_files_iter_fn(args))
    return (ms,stats)


def raw_save_model_state(ms,stats):
    #xxx = ms.ltm.blocks[0].sa.heads[0].memory_block_dist
    #xxx = ms.ltm.blocks[3].sa.heads[0].memory_block_dist.sort(dim=1)[0][0]
    fn=choose_new_filename(ms.args.save_model_filename)
    logging.info(f'Saving model to {fn}')
    torch.save((ms,stats),fn)

should_save_model = True
    
def save_model_state(ms,stats):
    #xxx = ms.ltm.blocks[0].sa.heads[0].memory_block_dist
    #xxx = ms.ltm.blocks[3].sa.heads[0].memory_block_dist.sort(dim=1)[0][0]
    if(should_save_model):
        fn=choose_new_filename(ms.args.save_model_filename)
        logging.info(f'Saving model to {fn}')
        torch.save((ms,stats),fn)
    else:
        logging.info('Model not being saved')

def imagine_rest_of_story(enc,ms,zf,perc=0.5,context_token_length=150,n_output=1000):
    with torch.no_grad():
        def create_single_file_iter_fn():
            return lambda: [zf]

        wli = iter(WorkLoaderIter(enc,{},create_single_file_iter_fn(),batch_size=1,block_size=ms.args.block_size,num_loops=1,
                             device=ms.args.device))
        ms.ltm.reset_for_new_works(1)

        print(f'Reading story {zf}...')

        (block_pos_list,batch,targets) = next(wli)
        logits,loss,query_loss=ms.ltm(block_pos_list,batch)
        cutoff_point = int(len(wli.current_works[0].data) * perc)

        start_point = max(0,cutoff_point - context_token_length)
        print(f'Story so far:\n{enc.decode(wli.current_works[0].data[start_point:cutoff_point].tolist())}\n')

        print_token_count = int(cutoff_point * 0.05)

        #first we run against the story up to the point where we start imagining
        for (block_pos_list,batch,targets) in wli:
            logits,loss,query_loss=ms.ltm(block_pos_list,batch)

            if(block_pos_list[0] % print_token_count == 0):
                print(f'{block_pos_list[0]} out of {cutoff_point} tokens read')

            if(block_pos_list[0] > cutoff_point):
                break
                           

        print('AI imagining the rest of the story...\n')

        for i in range(n_output):
            logits = logits.squeeze(dim=0).squeeze(dim=0)
            probs = F.softmax(logits, dim=0)
            idx_next = torch.multinomial(probs,num_samples=1).unsqueeze(dim=0)
            print(enc.decode(idx_next[0].tolist()),end='',flush=True)
            block_pos_list[0] += 1
            
            logits,loss,query_loss=ms.ltm(block_pos_list,idx_next)
            

def run_training(args,enc,ms):
    global should_save_model

    if(args.no_save):
        should_save_model = False
        logging.warning("--no_save is set to true, model will not be saved automatically")
        
    should_save_model = not args.no_save
    
    def save_and_intr():
        save_model_state(ms,stats)
        pdb.set_trace()

    stats.last_save_time = time.time()
    for idx,(block_pos_list,batch,targets) in enumerate(ms.wl):
        with util.DelayedKeyboardInterrupt(save_and_intr,eat_interrupt=True):
            logits,loss,query_loss = ms.ltm(block_pos_list,batch,targets)
            ms.optimizer.zero_grad(set_to_none=True)
            (loss+query_loss*args.query_loss_adj).backward()
            ms.optimizer.step()
            print_status(ms,args.print_status,args.trailing_avg_perc,loss,query_loss)
            if(time.time() - stats.last_save_time >= args.save_model_time):
                save_model_state(ms,stats)
                stats.last_save_time = time.time()

def main():
    global stats

    parser = init_argparse()
    args = parser.parse_args()

    logging.config.fileConfig('logging.conf')

    enc = tiktoken.get_encoding("gpt2")
    if(args.load_model is not None):
        print(f"loading model {args.load_model}")
        (ms,stats) = load_model_state(args.load_model,enc,args)
    else:
        ms = create_model_state(enc,args)

    print(sum(p.numel() for p in ms.ltm.parameters())/1e6, 'M parameters')
    if(args.bos):
        pdb.set_trace()
        
    if(args.command == "train"):
        run_training(args,enc,ms)
        save_model_state(ms,stats)
    elif(args.command == "imagine"):
        imagine_rest_of_story(enc,ms,args.zip_file,args.perc)

def myplot(x):
    plt.plot(x.detach().cpu())
    plt.show()
        
if __name__ == "__main__":
    main()
    
