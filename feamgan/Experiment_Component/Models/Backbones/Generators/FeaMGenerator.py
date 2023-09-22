import torch
import torch.nn as nn
import torch.nn.functional as F

from feamgan.Experiment_Component.Models.BaseNetwork import BaseNetwork
from feamgan.Experiment_Component.Models.Blocks.FADEResnetBlock import FADEResnetBlock
from feamgan.Experiment_Component.Models.Blocks.StreamResnetBlock import StreamResnetBlock

from feamgan.Experiment_Component.Models.Streams.Stream import Stream 

# Inspired by the Generator from https://github.com/EndlessSora/TSIT
class FeaMGenerator(BaseNetwork):

    def __init__(self, backbone_config, content_nc, output_nc):
        super().__init__()
        nf = backbone_config["nrFirstLayerFilters"]
        modularization = backbone_config["modularization"]  
 
        norm_G = backbone_config["normG"]   
        norm_S = backbone_config["normS"]    
        norm_r = norm_G.replace('fade', '')[0:-3]

        fmiddle_first_layer = 8 * round(content_nc / 8. )
        self.content_stream = Stream(ngf=nf, semantic_nc=content_nc, norm_S=norm_S, fmiddle_first_layer=fmiddle_first_layer)
      
        # Downsampled Content
        self.embed_0 = nn.Conv2d(content_nc, 4 * nf, 3, stride=2, padding=1)
        self.embed_1 = nn.Conv2d(4 * nf, 8 * nf, 3, stride=2, padding=1)
        self.embed_2 = nn.Conv2d(8 * nf, 16 * nf, 3, padding=1)

        # Up Stream FADE Blocks
        self.up_0 = FADEResnetBlock(16 * nf, 16 * nf, norm_G, modularization)
        self.up_1 = FADEResnetBlock(16 * nf, 16 * nf, norm_G, modularization)
        self.up_2 = FADEResnetBlock(16 * nf, 16 * nf, norm_G, modularization)
        self.up_3 = FADEResnetBlock(16 * nf, 8 * nf, norm_G, modularization)
        self.up_4 = FADEResnetBlock(8 * nf, 4 * nf, norm_G, modularization)
        self.up_5 = FADEResnetBlock(4 * nf, 2 * nf, norm_G, modularization)
        self.up_6 = FADEResnetBlock(2 * nf, 1 * nf, norm_G, modularization)
        self.up_7 = FADEResnetBlock(1 * nf, 1 * nf, norm_G, modularization)

        # Up Stream ResnetBlock
        self.res_0 = StreamResnetBlock(1 * nf, 1  * nf, norm_r) 
        self.res_1 = StreamResnetBlock(1 * nf, 1  * nf, norm_r) 

        # Image Conv
        self.conv_img = nn.Conv2d(1 * nf, output_nc, 3, padding=1)

    def forward(self, input, _=None):
        ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7 = self.content_stream(input)

        x = self.embed_0(input)
        x = F.leaky_relu(x, 2e-1)
        x = self.embed_1(x)
        x = F.leaky_relu(x, 2e-1)
        x = F.interpolate(x, ft7.shape[2:])
        x = self.embed_2(x)
 
        x = self.up_0(x, ft7)
        x = F.interpolate(x, ft6.shape[2:])
        x = self.up_1(x, ft6)
        x = F.interpolate(x, ft5.shape[2:])
        x = self.up_2(x, ft5)
        x = F.interpolate(x, ft4.shape[2:])
        x = self.up_3(x, ft4)
        x = F.interpolate(x, ft3.shape[2:])
        x = self.up_4(x, ft3)
        x = F.interpolate(x, ft2.shape[2:])
        x = self.up_5(x, ft2)
        x = F.interpolate(x, ft1.shape[2:])
        x = self.up_6(x, ft1)
        x = self.res_0(x)
        x = F.interpolate(x, ft0.shape[2:])
        x = self.up_7(x, ft0)

        x = self.res_1(x)
        x = F.leaky_relu(x, 2e-1)
        x = self.conv_img(x)
        out = torch.tanh(x)
        return out
