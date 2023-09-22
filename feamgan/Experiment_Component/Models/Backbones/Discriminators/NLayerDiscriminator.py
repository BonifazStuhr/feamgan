import torch.nn as nn
import numpy as np

from feamgan.Experiment_Component.Models.BaseNetwork import BaseNetwork
from feamgan.Experiment_Component.Models.Normalization.utils import getNormLayer

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):

    def __init__(self, backbone_config, ganFeat_loss, input_nc):
        super().__init__()        
        self.ganFeat_loss = ganFeat_loss
        self.norm_D = backbone_config["normD"] 
        self.n_layers_D = backbone_config["nLayersD"] 
        nf = backbone_config["nrFirstLayerFilters"] 

        self.stride_1_layer= 3
        self.use_bias = True
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
                 
        activation = nn.LeakyReLU(0.2, False)

        norm_layer = getNormLayer(self.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw, bias=self.use_bias),
                     activation]]

        for n in range(1, self.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n >= self.stride_1_layer else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw, bias=self.use_bias)),
                          activation
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw, bias=self.use_bias)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = self.ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
