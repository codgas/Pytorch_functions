import torch

def tril_values(x):
    if len(x.shape) == 2:
        mask = torch.ones(x.shape[0], x.shape[0])
        return x[mask.triu()==1]
    elif len(x.shape) == 3:
        mask = torch.ones(x.shape[1], x.shape[1])
        return x[:,mask.triu()==1]
    else:
        raise ValueError("can't deal with this shape {}".format(x.shape)) 
