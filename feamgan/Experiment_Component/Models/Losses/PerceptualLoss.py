import os
import torch
import torch.nn as nn
import torchvision

from feamgan.Experiment_Component.Models.utils import modelUtils

# Perceptual loss that uses a pretrained VGG network
class PerceptualLoss(nn.Module):
    def __init__(self, model="vgg19"):
        super(PerceptualLoss, self).__init__()

        self.input_norm = modelUtils.applyImagenetNormalization
        self.weights = [1.0]
        if model=="vgg19":
            self.model = VGG19().cuda()
            self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.criterion = nn.L1Loss()
      
    def forward(self, x, y):
        x, y = self.input_norm(x), self.input_norm(y)
        x_model, y_model = self.model(x), self.model(y)
        loss = 0
        for x, y, w in zip(x_model, y_model, self.weights):
            loss += w * self.criterion(x, y.detach())
        return loss

class PerceptualNetwork(nn.Module):
    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, nn.Sequential)
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = []
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers: output.append(x)
        return output

def vgg16(layers):
    os.environ['TORCH_HOME'] = 'models/vgg16'
    network = torchvision.models.vgg16(pretrained=True).features
    layer_name_mapping = {3: 'relu_1_2',
                          8: 'relu_2_2',  
                          15: 'relu_3_3',
                          22: 'relu_4_3',
                          29: 'relu_5_3',
                          30: 'mp_5_3'}
    return PerceptualNetwork(network, layer_name_mapping, layers)

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        os.environ['TORCH_HOME'] = 'models/vgg19'
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out