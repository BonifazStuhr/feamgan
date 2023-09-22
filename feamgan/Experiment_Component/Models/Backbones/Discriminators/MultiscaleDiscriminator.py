
import torch.nn.functional as F

from feamgan.Experiment_Component.Models.BaseNetwork import BaseNetwork
from feamgan.Experiment_Component.Models.Backbones.Discriminators.NLayerDiscriminator import NLayerDiscriminator
from feamgan.Experiment_Component.Models.Backbones.Discriminators.FeaMDiscriminator import FeaMDiscriminator

class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, backbone_config, ganFeat_loss, input_nc):
        super().__init__()
        self.num_D = backbone_config["numD"]
        self.netD_subarch = backbone_config["netDSubarch"]
        self.ganFeat_loss = ganFeat_loss

        for i in range(self.num_D):
            subnetD = self.createSingleDiscriminator(
                backbone_config, ganFeat_loss, input_nc)
            self.add_module('discriminator_%d' % i, subnetD)

    def createSingleDiscriminator(self, backbone_config, ganFeat_loss, input_nc):
        subarch = self.netD_subarch
        if subarch == 'NLayerDiscriminator':
            netD = NLayerDiscriminator(backbone_config, ganFeat_loss, input_nc)
        elif subarch == 'FeaMDiscriminator':
            netD = FeaMDiscriminator(backbone_config, ganFeat_loss, input_nc)
        else:
            raise ValueError(
                'unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size num_D x n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = self.ganFeat_loss
        for _, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)
        output = {"image": result}
        return output

