import torch
import torch.nn as nn
import numpy as np

from torch.nn import init

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', norm_val=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, norm_val)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('GroupNorm') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, norm_val)
                elif init_type == 'uniformSine':
                    num_input = m.weight.size(-1)
                    sine_variance = np.sqrt(6 / num_input) / norm_val
                    init.uniform_(m.weight.data, -sine_variance, sine_variance)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=norm_val)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=norm_val)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        with torch.no_grad():
            self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, norm_val)