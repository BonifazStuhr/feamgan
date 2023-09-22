import torch.nn as nn
import torch.nn.functional as F

from feamgan.Experiment_Component.Models.BaseNetwork import BaseNetwork
from feamgan.Experiment_Component.Models.Blocks.StreamResnetBlock import StreamResnetBlock 

class Stream(BaseNetwork):
    def __init__(self, ngf, semantic_nc, norm_S, fmiddle_first_layer):
        super().__init__()
        nf = ngf
        self.linear_0 = nn.Conv2d(semantic_nc, nf, kernel_size=1)
        self.linear_1 = nn.Conv2d(nf, nf, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        self.res_0 = StreamResnetBlock(nf, 1 * nf, norm_S, fmiddle_first_layer)  # 64-ch feature
        self.res_1 = StreamResnetBlock(1  * nf, 2  * nf, norm_S)   # 128-ch  feature
        self.res_2 = StreamResnetBlock(2  * nf, 4  * nf, norm_S)   # 256-ch  feature
        self.res_3 = StreamResnetBlock(4  * nf, 8  * nf, norm_S)   # 512-ch  feature
        self.res_4 = StreamResnetBlock(8  * nf, 16 * nf, norm_S)   # 1024-ch feature
        self.res_5 = StreamResnetBlock(16 * nf, 16 * nf, norm_S)   # 1024-ch feature
        self.res_6 = StreamResnetBlock(16 * nf, 16 * nf, norm_S)   # 1024-ch feature
        self.res_7 = StreamResnetBlock(16 * nf, 16 * nf, norm_S)   # 1024-ch feature

    def down(self, input):
        return F.interpolate(input, scale_factor=0.5)

    def forward(self,input):
        # assume that input shape is (n,c,256,512)
        input = self.act(self.linear_0(input))
        input = self.act(self.linear_1(input))

        x0 = self.res_0(input) # (n,64,256,512)

        x1 = self.down(x0)
        x1 = self.res_1(x1)    # (n,128,128,256)

        x2 = self.down(x1)
        x2 = self.res_2(x2)    # (n,256,64,128)

        x3 = self.down(x2)
        x3 = self.res_3(x3)    # (n,512,32,64)

        x4 = self.down(x3)
        x4 = self.res_4(x4)    # (n,1024,16,32)

        x5 = self.down(x4)
        x5 = self.res_5(x5)    # (n,1024,8,16)

        x6 = self.down(x5)
        x6 = self.res_6(x6)    # (n,1024,4,8)

        x7 = self.down(x6)
        x7 = self.res_7(x7)    # (n,1024,2,4)

        return [x0, x1, x2, x3, x4, x5, x6, x7]