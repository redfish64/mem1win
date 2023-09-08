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
     



        #target loss to immediate params (this runs through mem_in_layer and core_layer, but not mem_out params)
        imm_loss_to_sq_mem_in_grad = ag.grad((loss,),(squishing_param_tensors+mem_in),retain_graph=True)
        imm_loss_to_squishing_param_tensors_grad = grad.grad_from_list(loss_to_sq_mem_in_grad[0:len(squishing_param_tensors)])
        loss_to_mem_in_grad = grad.grad_from_list(loss_to_sq_mem_in_grad[len(squishing_param_tensors):])

        loss_to_past_squishing_params_grad = grad.combine_grads(loss_to_mem_in_grad,last_mem_in_to_squish_params_grad)
        loss_to_squishing_params_grad = imm_loss_to_squishing_param_tensors_grad + loss_to_past_squishing_params_grad

        loss_to_mem_out_params_grad = loss_to_mem_in_grad * last_mem_in_to_mem_out_grad


        mem_to_mem_grad = torch.full((n_batch_size,n_mem,n_mem),0.)
        mem_to_wrapper_params_grad = torch.full((n_batch_size,n_mem,len(wrapper_params)),0.)

        for bi in range(n_batch_size):
            for mi in range(n_mem):
                (mem_item_to_mem_grad,*mem_item_to_wrapper_params_grad) = ag.grad((mw.last_run_mem_out[bi,mi],),[mw.memory[bi]]+wrapper_params,retain_graph=True)
                pdb.set_trace()
                mem_to_mem_grad[bi,mi] = mem_item_to_mem_grad
                mem_to_wrapper_params_grad[bi,mi] = mem_item_to_wrapper_params_grad

        pdb.set_trace()
