import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import logging.config
import argparse
import glob
import time
from typing import NamedTuple
import re
import os
import pdb
import matplotlib.pyplot as plt
import mem_model as model
import JacModule as jm

# the reason I copied util into this git tree is so that there is a consistent commit. When we run each
# session, I want to record the git state along with all parameters, so that I can reproduce/track the results
import util
import gutenberg


class ModelTrailingAvgStats:
    def __init__(self,trailing_avg_perc):
        self.u_block_pos = None
        self.u_max_block_pos = None
        self.loss = None
        self.last_item_loss = None
        self.trailing_avg_perc = trailing_avg_perc
        self.stat_vals = {}

    def update_ta(self, name, val):
        if not (name in self.stat_vals):
            self.stat_vals[name] = val
        else:
            self.stat_vals[name] = val * self.trailing_avg_perc + self.stat_vals[name] * (1.0-self.trailing_avg_perc)

class ModelStats:
    def __init__(self,trailing_avg_perc):
        #last time print_status printed a message
        self.last_print_time = 0.0
        self.last_save_time = 0.0
        self.last_est_loss_time = 0.0
        self.n_batches_trained = 0
        self.n_works_read = 0
        self.tas = ModelTrailingAvgStats(trailing_avg_perc)
 
stats = None

# trailing_avg_perc: amount of the past to keep, so should be closer to 0.99 than 0.01
def print_status(ms,time_between_updates, stat_vals):
    #yes we use global variables here, but this is just for displaying status
    global stats
    #last_print_time, trailing_avg_loss, trailing_avg_query_loss, n_batches_trained

    curr_time = time.time()
    stats.n_batches_trained += 1

    for n,v in stat_vals.items():
        if(v is not None):
            stats.tas.update_ta(n,v)

    if(curr_time - stats.last_print_time < time_between_updates):
        return False

    stats.last_print_time = curr_time

    def format_loss(v):
        if(v is None):
            return "None"
        if(abs(v) < 1e-3):
            return f'{v:1.3e}'
        return f'{v:1.6f}'

    def format_stats(name,stats):
        out = ''
        for k,v in stats.items():
            out += f'{k}: {format_loss(v)} '

        return f'\n({name}::{out})'
    
    print(f'works_read: {stats.n_works_read}, batches_trained: {stats.n_batches_trained:8} {format_stats("curr",stat_vals)} {format_stats("trailing",stats.tas.stat_vals)}')

    return True

class CurrentWork():
    #data: the text of the work
    #zf: the zip file name
    #fn: the file name
    def __init__(self,data,zf,fn,offset):
        self.data = data
        self.zf = zf
        self.fn = fn
        self.block_pos = 0
        self.offset = offset

class SingleRandomWorkLoader():
    """
    loads batches in random order from a single file
    """

    def __init__(self,enc,zf,batch_size,block_size,num_loops,device):
        self.enc = enc
        self.zf = zf
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        data,_,_ = gutenberg.get_etext(zf)

        #choose an approx number of batches that match number of loops for non random training
        self.num_batches = num_loops * len(data) // self.block_size // self.batch_size
        self.data_tensor = torch.tensor(self.enc.encode(data),device=self.device)
        self.curr_loop = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        if(self.curr_loop >= self.num_batches):
            raise StopIteration

        self.curr_loop += 1

        ix = torch.randint(self.data_tensor.shape[0] - self.block_size, (self.batch_size,))

        # Extract sequences starting at the random indices
        input_data = [self.data_tensor[start:start+self.block_size] for start in ix]
        input_data = torch.stack(input_data)

        # Create target sequences: they are the same as input sequences but shifted by one token
        target_data = [self.data_tensor[start+1:start+self.block_size+1] for start in ix]
        target_data = torch.stack(target_data)

        return (ix // self.block_size).tolist(), input_data.to(self.device), target_data.to(self.device)



#
class WorkLoader():
    """
    loads batches of works in sequential order, so that each block in a batch follows from the previous block in that
    indexed position. 
    """

    def __init__(self,enc,works_read_set,iter_fn,batch_size,block_size,num_loops,device, pre_start_token,post_end_token,min_size=0,max_size=None,offset_loops=True,trailing_empty_blocks=1,update_global_stats=True):
        """
        Parameters:
        iter_fn: a function that returns an iterator over the works to read. It can be called more than once for multiple runs
        works_read_set: This is used to skip works that have already been read

        num_loops: how many times to loop over the works

        pre_start_token: token to print before file is started. Since we are possibly offsetting the file at different
        positions we need to have something to put in the buffer before it starts. Ex, suppose pre_start_token is ' ',
        then if the offset was -3, the first block may contains '   It was a dark and stormy night...'

        post_end_token: This is the token to use after the file ends. Since we process in batches there are times
        when its convienent to let a batch continue to process even after the data has ended. Ex, suppose we are
        estimating the loss with a batch of multiple stories, when one ends before the other, we will want a token
        that we can append to the shorter story so both batches have the same length. Otherwise, same as pre_start_token.

        """
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
        self.offset_loops=offset_loops
        self.trailing_empty_blocks = trailing_empty_blocks
        self.pre_start_token = pre_start_token
        self.post_end_token = post_end_token
        self.update_global_stats = update_global_stats

    @staticmethod
    def _read_next_valid_work(works_read_set,iter,min_size=0,max_size=None):
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

    def _read_next_work(self):
        """
        Reads the next work in current_works and updates curr_loop and stats. If there are no works left, restarts as
        long as curr_loop < num_loops, otherwise returns None. Can be safetly called multiple times after
        it has already returned None.
        """
        global stats
        if(self.curr_loop == self.num_loops): # if we are already tried to read another work and there are none
            return None
        next_work,zf,fn = WorkLoader._read_next_valid_work(self.works_read_set,self.iter,self.min_size,self.max_size)
        if(next_work is None):
            self.curr_loop += 1
            if(self.curr_loop == self.num_loops):
                logging.info('No more works, and at end of loops')
                return None

            
            logging.info(f'At end of works, restarting iterator, loop {self.curr_loop} out of {self.num_loops}')
            if(self.update_global_stats):
                stats.n_works_read = 0
            self.works_read_set = {}
            self.iter = self.iter_fn()
            return self._read_next_work()

        if(self.update_global_stats):
            stats.n_works_read += 1

        #TODO 2.8 set disallowed_special to set() to allow special tokens since we have dirty data. Need to
        # find out how special tokens are encoded and strip them from the input
        logging.debug(f'Encoding {zf}, {fn}, {next_work[:100]}...')
        offset = self.curr_loop % self.block_size if self.offset_loops else 0
        return CurrentWork(torch.tensor(self.enc.encode(next_work),device=self.device),zf,fn,offset)

    def __getstate__(self):
        new_dict = self.__dict__.copy()
        new_dict['enc'] = None
        new_dict['iter_fn'] = None
        new_dict['iter'] = None

        #rather than saving the tokens of all the works being processed into the pt file, we just save the filenames,
        #and then reload the data when we reload the model in init_after_load
        new_dict['current_works'] = None
        new_dict['current_work_offsets'] = [cw.offset for cw in self.current_works]
        new_dict['current_work_zf'] = [cw.zf for cw in self.current_works]
        new_dict['current_work_bp'] = [cw.block_pos for cw in self.current_works]

        return new_dict

    def reset(self):
        self.works_read_set = {}
        self.iter = self.iter_fn()
        self.curr_loop=0

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
            offset = self.current_work_offsets[i]
            data,fn,error = gutenberg.get_etext(zf)
            if(data is None):
                raise Exception('Couldn\'t read work that was in process of being read, {zf}: {error}')
            cw = CurrentWork(torch.tensor(self.enc.encode(data),device=self.device),zf,fn,offset)
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

        n_works_finished = 0

        for i in range(self.batch_size):
            cw = self.current_works[i]
            token_pos = 0 if cw is None else cw.block_pos * self.block_size - cw.offset

            #if we're near enough to the end of the current work to not be able to get a full block, then
            #read the next work. We add +1 so that the targets are also loadable (targets are symbol + 1)
            #TODO 2.8 we might want to read partial blocks for the end but for now we'll keep it simple
            if(cw is None or token_pos >= len(cw.data) + self.trailing_empty_blocks * self.block_size):
                cw = self._read_next_work()
                if(cw is None): #if no more works to fill the batch then we'll keep creating empty trailing blocks
                    #until all works are finished
                    n_works_finished += 1
                else:
                    #we record that the previous file was read so if we save and reload the model, we won't
                    #read it again. Note that since we are saving the current_works with the model when we
                    #save, we don't have to worry that when we reload we would skip the current file, since
                    #it will be loaded when we load the model.
                    self.works_read_set[cw.zf] = True
                    self.current_works[i] = cw
                    token_pos = -cw.offset

            def get_block_data(data,start,end):
                #if we're before the data begins completely
                if(end <= 0):
                    return torch.full((end-start,),self.pre_start_token,device=self.device)

                #if we're past the end of the data completely
                if(start >= data.shape[0]):
                    return torch.full((end-start,),self.post_end_token,device=self.device)

                #if we are starting before the data begins
                if(start < 0):
                    out = torch.full((-start,),self.pre_start_token,device=self.device)
                    curr_start = 0
                else:
                    out = torch.tensor([],device=self.device,dtype=int)
                    curr_start = start
                curr_end = min(end,data.shape[0])
                out = torch.cat((out,data[curr_start:curr_end]))
                #if we are ending after the data ends
                if(curr_end < end):
                    out = torch.cat((out,torch.full((end-curr_end,),self.post_end_token,device=self.device)))

                return out

            if cw is None:
                block = torch.full((self.block_size,),self.post_end_token,device=self.device)
                target = torch.full((self.block_size,),self.post_end_token,device=self.device)
            else:
                block = get_block_data(cw.data,token_pos,token_pos + self.block_size)
                target = get_block_data(cw.data,token_pos+1,token_pos+1 + self.block_size)
                self.current_works[i].block_pos += 1

            batch_list.append(block)
            target_list.append(target)

        #if all the works are finished we're really done
        if(n_works_finished == self.batch_size):
            raise StopIteration

        return ([(0 if cw is None else cw.block_pos-1) for cw in self.current_works],
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
    shared_parser.add_argument('--bos', action='store_true', help='if set, will drop into a debugger on start')
    shared_parser.add_argument('--wrap_model', action='store_true', help='if set, will wrap the model with the memory layer')
    shared_parser.add_argument('--pre_start_token', nargs='?', type=int,default=0, help='token to use to fill the buffer before the file begins. This is used when offseting the position the buffer reads the files differently for each loop')
    shared_parser.add_argument('--post_end_token', nargs='?', type=int,default=1, help='token to use to fill the buffer after the file ends.')
    shared_parser.add_argument('--device', nargs='?', default="cpu",help='device to run on (cpu or cuda)')

    train_parser = subparsers.add_parser("train", help="Train a new or existing model", parents=[shared_parser])

    train_data_group = train_parser.add_mutually_exclusive_group(required=True)
    train_data_group.add_argument('--dir_tree',help='directory tree to look for zipped text files for training')
    train_data_group.add_argument('--single_train_file',help='single zipped file of training data. Don\'t make this too big, it\'s loaded into memory all at once.')

    train_parser.add_argument('--batch_size', nargs='?', type=int,default=64,help='num of batches to process at once')
    train_parser.add_argument('--block_size', nargs='?', type=int,default=256,help='block size in tokens. This along with memory_size decides the number of input params')
    train_parser.add_argument('--n_loops', nargs='?', type=int,default=100,help='number of times to loop through data')
    train_parser.add_argument('--n_head', nargs='?', type=int,default=6,help='number of heads per layer')
    train_parser.add_argument('--n_embd', nargs='?', type=int,default=384,help='number of embedding dimensions')
    train_parser.add_argument('--n_mem_embd', nargs='?', type=int,default=96,help='number of embedding dimensions for memory. These are more costly than normal embeddings due to the jacobian')
    train_parser.add_argument('--n_memory_per_layer', nargs='?', type=int,default=8,help='number of memory values to store across cycles')
    train_parser.add_argument('--dropout', nargs='?', type=float,default=0.2,help='dropout percentage')
    train_parser.add_argument('--n_layer', nargs='?', type=int,default=6,help='number of layers')
    train_parser.add_argument('--min_text_size', nargs='?', type=int,default=10000,help='min size of text files to process (others will be skipped)')
    train_parser.add_argument('--max_text_size', nargs='?', type=int,default=1024*1024*10,help='max size of text files to process (others will be skipped)')
    train_parser.add_argument('--trailing_avg_perc', nargs='?', type=float,default=0.01,help='only used for display. When printing status, we use a trailing avg. This is the percentage to keep from the previous runs, the remainder is for the current one.')
    train_parser.add_argument('--test_dir_tree',nargs='?',help='directory tree to look for zipped text files for testing. All of these files will be run to estimate loss so don\'t add too many here')
    train_parser.add_argument('--print_status_time', nargs='?',type=float,default=10,help='time between printing status updates and estimating loss in seconds')
    train_parser.add_argument('--save_model_time', nargs='?',type=float,default=60*15,help='time between saving the model in seconds')
    train_parser.add_argument('--est_loss_time', nargs='?',type=float,default=60,help='time between estimating the loss of the model in seconds')
    train_parser.add_argument('--no_save', action='store_true',help='won\'t save the model automatically. The model can still be saved by interrupt')
    train_parser.add_argument('--no_bias', action='store_true',help='GPT param, whether to have a bias or not. GPT creator seems to think it\'s better not to have')
    train_parser.add_argument('--no_flash', action='store_true',help='True if scaled_dot_product_attention function shouldn\'t be used even if available')
    train_parser.add_argument('--random_batches', action='store_true',help='True if we want to use random batches rather than go through the data sequentially. This will of course make the memory useless')
    train_parser.add_argument('--learning_rate', nargs='?',type=float,default=6e-4,help='learning rate for optimizer')
    train_parser.add_argument('--weight_decay', nargs='?',type=float,default=1e-1,help='GPT module parameter to try and force the weights closer to zero')
    train_parser.add_argument('--jac_grad_decay', nargs='?',type=float,default=0.03,help='Amount to decay the effect of the previous cycles grads on the current learning step. From 0.0 to 1.0')
    train_parser.add_argument('--mem_grad_multiplier', nargs='?',type=float,default=1,help='Multiples the memory gradiants by this value before optimizing.')
    train_parser.add_argument('--beta1', nargs='?',type=float,default=0.9,help='AdamW parameter to control running average for momentum')
    train_parser.add_argument('--beta2', nargs='?',type=float,default=0.95,help='AdamW parameter to control running average for momentum')
    train_parser.add_argument('--load_model', nargs='?',help='load a previously generated model')
    train_parser.add_argument('--save_model_filename', nargs='?',help='name to save model after training (or interrupt)')
    train_parser.add_argument('--log_config_file', nargs='?',default='logging.conf',help='log config file, see: https://docs.python.org/3/howto/logging.html#configuring-logging')

    imagine_parser = subparsers.add_parser("imagine", help="Imagine the rest of a story", parents=[shared_parser])
    imagine_parser.add_argument('--perc', nargs='?', type=float,default=0.5,help='amount of story to read before starting to imagine the rest')
    imagine_parser.add_argument('--load_model', required=True, help='load a previously generated model')
    imagine_parser.add_argument('--zip_file', required=True, help='zipped txt file containing story')

    return parser

class ModelState(NamedTuple):
    mdl: nn.Module
    wl: WorkLoader
    test_wl: WorkLoader
    optimizer: torch.optim.Optimizer
    config: argparse.Namespace

def create_files_iter_fn(dt):
    return lambda: glob.iglob(f'{dt}/**/*.zip',recursive=True)


def create_model_state(enc,parser,config):

    #max_token_value is one less than the vocab size as far as I can tell, see:
    #https://github.com/openai/tiktoken/blob/7830ed537badecefb5a357448be722bfd58f9fca/tiktoken/core.py
    gptconf = model.GPTConfig(block_size=config.block_size,vocab_size=config.vocab_size,n_layer=config.n_layer,
                              n_head=config.n_head,n_mem_embd=config.n_mem_embd,n_embd=config.n_embd,dropout=config.dropout,bias=not config.no_bias,
                              batch_size=config.batch_size,n_memory_per_layer=config.n_memory_per_layer,
                              jac_grad_decay=config.jac_grad_decay,
                              no_flash=config.no_flash,
                              mem_grad_multiplier=config.mem_grad_multiplier,
                              device=config.device)
    m = model.GPT(gptconf)

    if(config.random_batches):
        if config.single_train_file is None:
            parser.error("--single_train_file is required when --random_batches is set.")
        wl = SingleRandomWorkLoader(enc,config.single_train_file,config.batch_size,config.block_size,config.n_loops,config.device)
    else:
        if config.dir_tree is None:
            parser.error("--dir_tree is required when --random_batches is false.")
        wl = WorkLoader(enc,{},create_files_iter_fn(config.dir_tree),config.batch_size,config.block_size,config.n_loops,config.device,config.pre_start_token,config.post_end_token,config.min_text_size,config.max_text_size,update_global_stats=True)

    optimizer = m.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), config.device)

    if(config.test_dir_tree is not None):
        #TODO 3 HACK we are limiting the batch size to 1, since if there aren't enough files to fill all the batches
        #then WorkLoader will return empty batch items for the missing ones which can really screw up the
        #estimated loss
        test_wl = WorkLoader(enc,{},create_files_iter_fn(config.test_dir_tree),1,config.block_size,1,config.device,config.pre_start_token,config.post_end_token,config.min_text_size,config.max_text_size,update_global_stats=False)
    else:
        test_wl = None

    return ModelState(m,wl,test_wl,optimizer,config)


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

def load_model_state(fn,enc,config):
    (ms,stats) = torch.load(fn)
    ms.wl.init_after_load(enc,create_files_iter_fn(config))
    return (ms,stats)


def raw_save_model_state(ms,stats):
    #xxx = ms.ltm.blocks[0].sa.heads[0].memory_block_dist
    #xxx = ms.ltm.blocks[3].sa.heads[0].memory_block_dist.sort(dim=1)[0][0]
    fn=choose_new_filename(ms.config.save_model_filename)
    logging.info(f'Saving model to {fn}')
    torch.save((ms,stats),fn)

should_save_model = True
    
def save_model_state(ms,stats):
    #xxx = ms.ltm.blocks[0].sa.heads[0].memory_block_dist
    #xxx = ms.ltm.blocks[3].sa.heads[0].memory_block_dist.sort(dim=1)[0][0]
    if(should_save_model):
        fn=choose_new_filename(ms.config.save_model_filename)
        logging.info(f'Saving model to {fn}')
        torch.save((ms,stats),fn)
    else:
        logging.info('Model not being saved')

def imagine_rest_of_story(enc,ms,zf,perc=0.5,context_token_length=150,n_output=1000):
    with torch.no_grad():
        def create_single_file_iter_fn():
            return lambda: [zf]

        config = ms.config
        
        wli = iter(WorkLoader(enc,{},create_single_file_iter_fn(),1,config.block_size,1,
                              config.device,config.pre_start_token,config.post_end_token,config.min_text_size,config.max_text_size))
        ms.mdl.reset_memory()

        print(f'Reading story {zf}...')

        #we read one block so that we read the story into memory
        (block_pos_list,batch,targets) = next(wli)
        logits,loss,last_item_loss=ms.mdl(batch)

        cutoff_point = int(len(wli.current_works[0].data) * perc)


        print_block_count = int(cutoff_point * 0.05 / config.block_size)
        #pdb.set_trace()

        #first we run against the story up to the point where we start imagining
        for (block_pos_list,batch,targets) in wli:
            logits,loss,query_loss=ms.mdl(batch)

            if(block_pos_list[0] % print_block_count == 0):
                print(f'{block_pos_list[0] * config.block_size} out of {cutoff_point} tokens read')

            if(block_pos_list[0] * config.block_size > cutoff_point):
                break
                           
            
        print('AI imagining the rest of the story...\n')

        print(f'Story so far:\n{enc.decode(batch[0])}\n')

        idx = batch[0]

        for i in range(n_output):
            logits = logits[0][0]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            print(enc.decode(idx_next.tolist()),end='',flush=True)

            idx = torch.cat((idx[1:],idx_next),dim=0)
            logits,loss,query_loss=ms.mdl(idx.view(batch.shape))
            

def est_loss(ms,config):
    if(ms.test_wl is None):
        logging.info('not estimating loss because no test_dir_tree specified')
        return
    with torch.no_grad():
        saved_model_memory = ms.mdl.save_and_reset_memory()

        total_iters = 0
        total_loss = 0.0
        total_last_item_loss = 0.0

        ms.test_wl.reset()

        for idx,(block_pos_list,batch,targets) in enumerate(ms.test_wl):
            logits,loss,last_item_loss = ms.mdl(batch,targets,compute_last_loss=True)
            total_loss += loss.item()
            total_last_item_loss += last_item_loss.item()
            total_iters += 1

        avg_loss = total_loss / total_iters
        avg_last_item_loss = total_last_item_loss / total_iters
        print(f'est_loss: {avg_loss:8.4f} {avg_last_item_loss:8.4f}')
        
        ms.mdl.restore_memory(saved_model_memory)

def run_training(config,enc,ms):
    global should_save_model

    if(config.no_save):
        should_save_model = False
        logging.warning("--no_save is set to true, model will not be saved automatically")
        
    should_save_model = not config.no_save
    
    def save_and_intr():
        save_model_state(ms,stats)
        pdb.set_trace()

    def get_avg_abs_grad(params):
        tag = 0.
        cnt = 0 
        for p in params:
            if(p.grad is not None):
                ag = p.grad.abs().mean().item()
                tag += ag
                cnt += 1

        if(cnt == 0):
            return 0
        return tag / cnt

    global stats
    stats = ModelStats(config.trailing_avg_perc)                        
    stats.last_save_time = time.time()
    for idx,(block_pos_list,batch,targets) in enumerate(ms.wl):
        with util.DelayedKeyboardInterrupt(save_and_intr,eat_interrupt=True):
            for index,bp in enumerate(block_pos_list):
                if(bp == 0 and not config.random_batches):
                    ms.mdl.reset_memory_for_item_in_batch(index)
                    
            logits,loss,last_item_loss = ms.mdl(batch,targets)
            ms.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ms.mdl.update_memory_and_mem_params()
            mem_avg_grad = get_avg_abs_grad(ms.mdl.get_mem_params())
            out_avg_grad = get_avg_abs_grad(ms.mdl.get_out_params())
            memory_vec_lengths = [torch.norm(b.memory).item() for b in ms.mdl.transformer.h]
            avg_memory_vec_length = sum(memory_vec_lengths)/len(memory_vec_lengths)
            ms.optimizer.step() # TODO 2 make sure that optimizer has right params

            print_status(ms,config.print_status_time,
                         {'loss' : loss,
                          'last_item_loss' : last_item_loss,
                          'mem_avg_grad' : mem_avg_grad,
                          'out_avg_grad' : out_avg_grad,
                          'avg_mem_vec' : avg_memory_vec_length})
            
            if(time.time() - stats.last_save_time >= config.save_model_time):
                save_model_state(ms,stats)
                stats.last_save_time = time.time()
            if(time.time() - stats.last_est_loss_time >= config.est_loss_time):
                est_loss(ms,config)
                stats.last_est_loss_time = time.time()

def main():
    global stats

    parser = init_argparse()
    config = parser.parse_args()

    logging.config.fileConfig('logging.conf')

    enc = util.CharEncoder()
    config.vocab_size = enc.n_tokens
    if(config.load_model is not None):
        print(f"loading model {config.load_model}")
        (ms,stats) = load_model_state(config.load_model,enc,config)
    else:
        ms = create_model_state(enc,parser,config)

    print(sum(p.numel() for p in ms.mdl.parameters())/1e6, 'M parameters')
    if(config.bos):
        pdb.set_trace()
        
    if(config.command == "train"):
        run_training(config,enc,ms)
        save_model_state(ms,stats)
    elif(config.command == "imagine"):
        imagine_rest_of_story(enc,ms,config.zip_file,config.perc)

def myplot(x):
    plt.plot(x.detach().cpu())
    plt.show()
        
if __name__ == "__main__":
    main()
    
