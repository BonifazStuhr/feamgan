import torch
from torch import autograd

def real_penalty(loss, real_img):
    ''' Compute penalty on real images. '''
    b = real_img.shape[0]
    grad_out = autograd.grad(outputs=loss, inputs=[real_img], create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
    reg_loss = torch.cat([g.pow(2).reshape(b, -1).sum(dim=1, keepdim=True) for g in grad_out if g is not None], 1).mean()
    return reg_loss

def tee_loss(x, y):
    return x+y, y.detach()