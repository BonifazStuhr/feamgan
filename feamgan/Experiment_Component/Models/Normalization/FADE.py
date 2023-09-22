import re
import torch.nn as nn

from apex import parallel

# Creates FADE normalization layer based on the given configuration from https://github.com/EndlessSora/TSIT
class FADE(nn.Module):
    def __init__(self, config_text, norm_nc, in_nc):
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
            raise ValueError('%s is not a recognized param-free norm type in FADE'
                             % param_free_norm_type)

        pw = ks // 2
        self.mlp_gamma = nn.Conv2d(in_nc, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(in_nc, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, feat):
        # Step 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Step 2. produce scale and bias conditioned on feature map
        gamma = self.mlp_gamma(feat)
        beta = self.mlp_beta(feat)

        # Step 3. apply scale and bias
        out = normalized * (1 + gamma) + beta
        
        return out