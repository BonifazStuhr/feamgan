import random
import torch 

import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn

def initDist(local_rank, backend='nccl'):         
    if dist.is_available():
        if dist.is_initialized():
            return torch.cuda.current_device()   
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method='env://')

def getRank():    
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank

def setRandomSeed(seed, by_rank=False):
    if by_rank:
        seed += getRank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def initCudnn(deterministic, benchmark):
    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark

def dictionaryToCuda(dictionary):
    cuda_dictionary = {}
    for key in dictionary.keys():
        if isinstance(dictionary[key], dict):
            cuda_dictionary[key] = dictionaryToCuda(dictionary[key])
        else:
            if torch.is_tensor(dictionary[key]):
                cuda_dictionary[key] = dictionary[key].cuda() 
    return cuda_dictionary

def dictionaryRemoveNone(obj):
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(dictionaryRemoveNone(x) for x in obj if x is not None)
    elif isinstance(obj, dict):
        return type(obj)((dictionaryRemoveNone(k), dictionaryRemoveNone(v))
        for k, v in obj.items() if k is not None and v is not None)
    else:
        return obj

def getWorldSize():
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size

def isMaster():
    return getRank() == 0

def distAllGatherTensor(tensor):
    world_size = getWorldSize()
    if world_size < 2:
        return [tensor]
    tensor_list = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(tensor_list, tensor)
    return tensor_list