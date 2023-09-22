import torch.nn as nn
import torch.nn.functional as F

from feamgan.Experiment_Component.Models.Normalization.FADE import FADE
from feamgan.Experiment_Component.Models.Normalization.FATE import FATE

# ResNet block that uses FADE.
# It differs from the ResNet block of SPADE in that
# it takes in the feature map as input, learns the skip connection if necessary.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was adpated from https://github.com/EndlessSora/TSIT
# and was extended with the FaDE implementation
class FADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G="spectralfadesyncbatch3x3", modularization="FADE"):
        super().__init__()
        # attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = fin

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = nn.utils.spectral_norm(self.conv_0)
            self.conv_1 = nn.utils.spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = nn.utils.spectral_norm(self.conv_s)

        # define normalization layers
        fade_config_str = norm_G.replace('spectral', '')
        if modularization == "FADE":
            self.norm_0 = FADE(fade_config_str, fin, fin)
            self.norm_1 = FADE(fade_config_str, fmiddle, fmiddle)
            if self.learned_shortcut:
                self.norm_s = FADE(fade_config_str, fin, fin)
        elif modularization == "FATE":
            self.norm_0 = FATE(fade_config_str, fin, fin, "fate1")
            self.norm_1 = FATE(fade_config_str, fmiddle, fmiddle, "fate1")
            if self.learned_shortcut:
                self.norm_s = FATE(fade_config_str, fin, fin, "fadeS")  
        else:
            raise ValueError('Unexpected modularization {}'.format(modularization))

    # Note the resnet block with FADE also takes in |feat|,
    # the feature map as input
    def forward(self, x, feat):
        x_s = self.shortcut(x, feat)

        dx = self.conv_0(self.actvn(self.norm_0(x, feat)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, feat)))

        out = x_s + dx

        return out

    def shortcut(self, x, feat):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, feat))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)