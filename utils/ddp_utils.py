import torch.distributed as dist


def is_main_rank():
    return dist.get_rank() == 0