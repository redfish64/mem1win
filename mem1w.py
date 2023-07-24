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

import util
import gutenberg


#The idea of this project is that we have a long term memory organized in a different way than most do:
#  [----- mem output -----] [next char(s) pred]
#     |              \           /
#    (+)         [ ----- core ----- ]
#     |              /           \
#  [----- mem input -----] [curr char(s)]
#
#  Lets assume a 1 char input window. So, in iteration 1, [mem input] is initialized to zero and included along with [curr char(s)]
#
#  Simple 2 iter example:
#
#  Iter 1:
#    Step 1: During forward pass use mem input and curr char to calculate [next char(s) pred] and do back prop.
#    Step 2:
#       let mem1_wrt_core1 = the derivative of [mem output] wrt all the weights in [core]
#
#  Iter 2:
#    Step 1: 
#       let out_wrt_core2 = derivative of [next char(s)] wrt weights in [core]
#       let out_wrt_mem1 = derivative of [next char(s)] wrt [mem input]
#       let mem2_wrt_core2 = the derivative of [mem output] wrt weights in [core]
#       let mem2_wrt_mem1 = the derivative of [mem output] wrt [mem input]
#       let mem2_wrt_both_cores = mem2_wrt_mem1 * mem1_wrt_core1 + mem2_wrt_core2
#       let out_wrt_both_cores = out_wrt_mem1 * mem1_wrt_core1 + out_wrt_core2
#    Step 2:
#       do backprop
#    
#
#   Full example:
#     Init:
#       let last_core_weight_grad = zeros_like(core_weights)
#       let memin_wrt_last_core = zeros_like(memory)
#       param c = constant indicating the relationship between distance of core weight and how much
#                 it effects memory
#       let memory = registered buffer with grad = true
#
#     Step 1: During forward pass use mem input and curr char to calculate [next char(s) pred]
#     Step 2: Do backprop from output_loss and mem_output at same time
#                 /\
#              we can do this by cat'ing the mem_output and the final loss into one tensor. Then
#              we run backwards against that like catted.backwards(ones_like_catted)
#              PERF: Not sure if this saves much time, since in the end we still have to calculate
#              individual gradiants for all memory locations and output.
#
#     Step 3:
#       let curr_core_weight_grad = length (aka absolute val) of gradient of output loss wrt core weight
#                                                   /\
#              TODO 2.5: should we really take the length here? If a weight moves forwards and backwards again
#                        it's grad would be less then if we used length. Maybe more appropriate?
#       let core_weight_grad = curr_core_weight_grad + last_core_weight_grad
#       let adj_memin_wrt_lastcore = memin_wrt_lastcore * c / core_weight_grad (for each core weight)
#                                                              /\
#             the idea here is that as the fastly moving core weights lose their ability to affect memory
#       let memout_wrt_all_cores = memout_wrt_core + memout_wrt_memin * adj_memin_wrt_lastcore
#       let out_wrt_all_cores = out_wrt_core + out_wrt_memin * adj_memin_wrt_lastcore
#     Step 5:
#       Save curr_core_weight_diff as last_core_weight_diff for next iter
#       Save memout_wrt_all_cores as memin_wrt_lastcore

# How this module fits in with everything else:
#   This will work as a wrapper for another module. It's forward will expect the original
#   module's input. It will then append the memory to it from the last round.
#   There will also be a output layer, which will accept the original model's output and generate
#   changes to memory values for the next round

# How to train with this module:
#   This module requires batches to be specially arraigned.
#   Each sample must come after the previous one in sequence according to the work (movie/book/etc.) be processed.


class M1WHead(nn.Module)
    # wrapped_module - this module takes an additional parameter in front, which is the
    #   input memory. The rest of parameters passed to this module are forwarded to it. In
    #   finetuning usage, you would typically take an original module, wrap it with
    #   another module that can take memory inputs, and squish it down to what the
    #   original module expects.
    # mem_out_module - this module takes the input memory and output of wrapped_module and returns a new value for memory
    # n_memory - number of memory units
    # n_items_per_batch - number of samples in each batch. This is necessary so the right amount of memory can be
    #   allocated
    def __init__(self,wrapped_module,mem_out_module,n_memory,n_items_per_batch):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.mem_out_module = mem_out_module
        self.n_memory = n_memory
        self.n_items_per_batch = n_items_per_batch

    def clear_memory():
        #we register memory as a buffer so it will be automatically saved during checkpoints
        self.memory = self.register_buffer('memory',torch.zeros(self.n_items_per_batch, self.n_memory))

    def forward(self, *args, **kwargs):
        wm_out = self.wrapped_module(self.memory,*args,**kwargs)
        mo_out = self.mem_out_module(self.memory,wm_out)

        
     

        
