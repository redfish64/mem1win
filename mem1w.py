import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional as af
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

import util
import gutenberg


#The idea of this project is that we have a long term memory organized in a different way than most do:
#  [----- mem output -----] [next char(s) pred]
#                 \              /
#         (mem out fn)          /
#         |        \           /
#         |    [ ----- core ----- ]
#         |        /           \
#  [----- mem input -----] [curr char(s)]
#

# How this module fits in with everything else:
#   This will work as a wrapper for another module. It's forward will expect the original
#   module's input. It will then append the memory to it from the last round as the first parameter
#   There will also be a output layer, which will accept the original model's output and generate
#   changes to memory values for the next round
# TODO add option to loop without consuming input if unsure (only updating memory)
# How to train with this module:
#   This module requires batches to be specially arraigned.
#   Each sample must come after the previous one in sequence according to the work (movie/book/etc.) be processed.

class M1WWrapper(nn.Module):
    # wrapped_module - this module takes an additional parameter in front, which is the
    #   input memory. The rest of parameters passed to this module are forwarded to it. In
    #   finetuning usage, you would typically take an original module, wrap it with
    #   another module that can take memory inputs, and and squish it down to what the
    #   original module expects. 
    # mem_out_model - A model which the input memory and output of wrapped_module and returns a new value for memory
    # n_memory - number of memory units
    # n_batch_size - number of samples in each batch. This is necessary so the right
    #   number of copies of independent memory can be allocated, to run in parallel (mostly for tuning)
    def __init__(self,wrapped_module,mem_out_model,n_memory,n_batch_size,decay):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.mem_out_model = mem_out_model
        self.n_memory = n_memory
        self.n_batch_size = n_batch_size
        self.decay = decay
        self.clear_memory()
        
    def clear_memory(self):
        #memory is a parameter so that we can get .grad of it to multiply with mem_grad from the previous cycle
        self.memory = torch.Parameter(torch.zeros(self.n_batch_size, self.n_memory))
        self.last_d_mem_out_params = torch.Parameter(torch.ones(self.n_batch_size, self.n_memory))

    def forward(self, *args, **kwargs):
        wm_out = self.wrapped_module(self.memory,*args,**kwargs)

        #saved for jaco_mat
        self.last_wm_out = wm_out

        return wm_out

    # this must be run after each cycle. It computes the memory gradiant and updates
    #  mem_out_model params
    def update_memory_nn(self):
        with torch.no_grad():

            #because forward was just run, we now have .grad populated in self.memory so we can use it here.
            d_complete_model_out_mem_in = self.memory.grad.clone()

            #optimize the nn for the memory
            self.mem_out_jm.optimize(d_complete_model_out_mem_in * self.last_d_mem_out_params)

            #update memory and update gradiant of d (mem / params)
            #PERF: we loop through each example one at a time because jacobian doesn't work
            #efficiently memorywise across large batches
            for i in range(self.n_batch_size):
                # return a function for an individual example
                jfn = self.mem_out_jm.nn_fn(self.wm_out[i])

                #we use jacobian here because we need a gradiant for each memory unit. pytorch.backwards
                #will only calculate the gradiant for a single value
                (d_mem_out_param,d_mem_out_mem_in) = af.jacobian(jfn,(self.params,self.mem_in[i]))
                self.last_d_mem_out_params[i] = d_mem_out_mem_in * self.last_d_mem_out_params[i] * self.decay + d_mem_out_param
                self.memory[i] = jfn(self.mem_out_jm.params,self.memory[i])


            



        

        
