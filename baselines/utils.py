import torch
from torch.autograd import Function


class AllGather(Function):
    """Function to gather data from all processes and backprop (accelerate gather does not backprop)
    taken from: # https://github.com/huggingface/accelerate/issues/76"""

    @staticmethod
    def forward(ctx, tensor):
        local_start = tensor.shape[0] * torch.distributed.get_rank()

        output_size = list(tensor.size())

        ctx.local_start = local_start
        ctx.local_length = tensor.shape[0]

        tensor = tensor.contiguous()

        out_len = tensor.shape[0] * torch.distributed.get_world_size()
        output_size[0] = out_len

        output = tensor.new_empty(output_size)
        gather_list = list(
            output.split([tensor.shape[0]] * torch.distributed.get_world_size(), dim=0)
        )

        torch.distributed.all_gather(gather_list, tensor)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.local_start
        length = ctx.local_length

        grad_input = grad_output.narrow(0, start, length)

        return (grad_input, None, None)


# Template for LlamaRec
INPUT_TEMPLATE = """### Instruction: Given user history in chronological order, recommend an item from the candidate pool with its index letter.

### Input: User history: {}; 

### Candidate pool:
{}
"""
