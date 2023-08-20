import torch
import torch.nn as nn

class DummyMemoryModel(nn.Module):
    def __init__(self,contained_model):
        super().__init__()
        self.contained_model = contained_model

    def forward(self,*args, **kwargs):
        return self.contained_model.forward(*args,**kwargs)

    def save_memory(self):
        return None

    def restore_memory(self,mem):
        return

    def reset_memory(self,batch_size):
        return

    def reset_memory_for_item_in_batch(self,index):
        return
    
