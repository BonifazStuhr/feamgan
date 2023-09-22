import torch.nn as nn
import torch.nn.functional as F

from apex import parallel

class StreamResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_S="spectralinstance", fmiddle=None):
        super().__init__()
        # attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = fmiddle if fmiddle else fin

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_S:
            self.conv_0 = nn.utils.spectral_norm(self.conv_0)
            self.conv_1 = nn.utils.spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = nn.utils.spectral_norm(self.conv_s)

        # define normalization layers
        subnorm_type = norm_S.replace('spectral', '')
        if subnorm_type == 'batch':
            self.norm_layer_in = nn.BatchNorm2d(fmiddle, affine=True)
            self.norm_layer_out= nn.BatchNorm2d(fout, affine=True)
            if self.learned_shortcut:
                self.norm_layer_s = nn.BatchNorm2d(fout, affine=True)
        elif subnorm_type == 'syncbatch':
            self.norm_layer_in = parallel.SyncBatchNorm(fmiddle, affine=True)
            self.norm_layer_out = parallel.SyncBatchNorm(fout, affine=True)
            if self.learned_shortcut:
                self.norm_layer_s = parallel.SyncBatchNorm(fout, affine=True)
        elif subnorm_type == 'instance':
            self.norm_layer_in = nn.InstanceNorm2d(fmiddle, affine=False)
            self.norm_layer_out= nn.InstanceNorm2d(fout, affine=False)
            if self.learned_shortcut:
                self.norm_layer_s = nn.InstanceNorm2d(fout, affine=False)
        elif subnorm_type == 'group':
            self.norm_layer_in = nn.GroupNorm(8, fmiddle, eps=1e-05, affine=False) 
            self.norm_layer_out= nn.GroupNorm(8, fout, eps=1e-05, affine=False)
            if self.learned_shortcut:
                self.norm_layer_s = nn.GroupNorm(8, fout, eps=1e-05, affine=False) 
        elif subnorm_type == 'groupA':
            self.norm_layer_in = nn.GroupNorm(8, fmiddle, eps=1e-05, affine=True) 
            self.norm_layer_out= nn.GroupNorm(8, fout, eps=1e-05, affine=True)
            if self.learned_shortcut:
                self.norm_layer_s = nn.GroupNorm(8, fout, eps=1e-05, affine=True) 
        elif subnorm_type == 'none':
            self.norm_layer_in = nn.Identity() 
            self.norm_layer_out= nn.Identity()
            if self.learned_shortcut:
                self.norm_layer_s = nn.Identity() 
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

            

    def forward(self, x):
        x_s = self.shortcut(x)

        dx = self.actvn(self.norm_layer_in(self.conv_0(x)))
        dx = self.actvn(self.norm_layer_out(self.conv_1(dx)))

        out = x_s + dx

        return out

    def shortcut(self,x):
        if self.learned_shortcut:
            x_s = self.actvn(self.norm_layer_s(self.conv_s(x)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)