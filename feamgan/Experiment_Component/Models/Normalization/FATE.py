import re
import torch
import torch.nn as nn

from apex import parallel

from feamgan.utils.visualizationUtils import wandbVisFATEAttention

class FATE(nn.Module):
    def __init__(self, config_text, norm_nc, in_nc, name):
        super().__init__()

        assert config_text.startswith('fade')
        parsed = re.search('fade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = parallel.SyncBatchNorm(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'group':
            self.param_free_norm = nn.GroupNorm(32, norm_nc, eps=1e-06, affine=False)
        elif param_free_norm_type == 'none':
            self.param_free_norm = nn.Identity()
        else:
            raise ValueError('%s is not a recognized param-free norm type in FATE'
                             % param_free_norm_type)

        pw = ks // 2
        self.gamma = nn.Conv2d(in_nc, norm_nc, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(in_nc, norm_nc, kernel_size=ks, padding=pw)
        self.attention = nn.Sequential(
            nn.Conv2d(in_nc*2, norm_nc, kernel_size=ks, padding=pw),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(norm_nc, norm_nc, kernel_size=ks, padding=pw),
            nn.Sigmoid()
        )

        self.frame_count = 0
        self.vis_frame_nr = 500
        self.name = name


    def forward(self, x, feat): 
        # Step 0. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
  
        # Step 1. Feature attentive modularization
        xf = torch.cat((x, feat), 1)
        a = self.attention(xf)

        # Step 1.1 Comment in to visualize the Attention for one step (quick hack), lots of data is generated
        #if self.frame_count == self.vis_frame_nr:
        #    wandbVisFATEAttention(a, self.frame_count, self.name)
        #self.frame_count += 1

        # Step 2. produce scale and bias conditioned on feature map
        gamma = self.gamma(feat)
        beta = self.beta(feat)

        # Step 4. apply attention
        gamma = gamma * a
        beta = beta * a

        # Step 4. apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
        