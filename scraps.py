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
     
