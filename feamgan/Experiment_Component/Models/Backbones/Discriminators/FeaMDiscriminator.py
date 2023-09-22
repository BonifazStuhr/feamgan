import torch
import torch.nn as nn
import torch.nn.functional as F

from feamgan.Experiment_Component.Models.BaseNetwork import BaseNetwork
from feamgan.Experiment_Component.Models.Normalization.utils import getNormLayer

# Inspired by the Feature-Pyramid Semantics Embedding Discriminator from https://github.com/xh-liu/CC-FPSE
class FeaMDiscriminator(BaseNetwork):
    def __init__(self, backbone_config, ganFeat_loss, input_nc):
        super().__init__()
        self.ganFeat_loss = ganFeat_loss
        self.norm_D = backbone_config["normD"] 
        nf = backbone_config["nrFirstLayerFilters"] 
        self.input_nc = input_nc
        image_nc = 3
        label_nc = input_nc -3 

        norm_layer = getNormLayer(self.norm_D)

        # bottom-up pathway
        self.enc1 = nn.Sequential(
                norm_layer(nn.Conv2d(image_nc, nf, kernel_size=3, stride=2, padding=1)), 
                nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(
                norm_layer(nn.Conv2d(nf, nf*2, kernel_size=3, stride=2, padding=1)), 
                nn.LeakyReLU(0.2, True))
        self.enc3 = nn.Sequential(
                norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=2, padding=1)), 
                nn.LeakyReLU(0.2, True))
        self.enc4 = nn.Sequential(
                norm_layer(nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=2, padding=1)), 
                nn.LeakyReLU(0.2, True))
        self.enc5 = nn.Sequential(
                norm_layer(nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=2, padding=1)), 
                nn.LeakyReLU(0.2, True))

        # top-down pathway
        self.lat2 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*2, nf*4, kernel_size=1)), 
                    nn.LeakyReLU(0.2, True))
        self.lat3 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*4, nf*4, kernel_size=1)), 
                    nn.LeakyReLU(0.2, True))
        self.lat4 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1)), 
                    nn.LeakyReLU(0.2, True))
        self.lat5 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*8, nf*4, kernel_size=1)), 
                    nn.LeakyReLU(0.2, True))
                
        # final layers
        self.final2 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)), 
                    nn.LeakyReLU(0.2, True))
        self.final3 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)), 
                    nn.LeakyReLU(0.2, True))
        self.final4 = nn.Sequential(
                    norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1)), 
                    nn.LeakyReLU(0.2, True))
    
        # true/false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf*2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf*2, nf*2, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, nf*2, kernel_size=1)

    def forward(self, input):
        segmap = input[:,0:self.input_nc-3]
        fake_and_real_img = input[:,-3:]

        # bottom-up pathway
        feat11 = self.enc1(fake_and_real_img)
        feat12 = self.enc2(feat11)
        feat13 = self.enc3(feat12)
        feat14 = self.enc4(feat13)
        feat15 = self.enc5(feat14)
        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)

        feat24 = F.interpolate(feat25, feat14.shape[2:], mode='bilinear') + self.lat4(feat14)
        feat23 = F.interpolate(feat24, feat13.shape[2:], mode='bilinear') + self.lat3(feat13)
        feat22 = F.interpolate(feat23, feat12.shape[2:], mode='bilinear') + self.lat2(feat12)

        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)
        # Patch-based True/False prediction
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        # intermediate features for discriminator feature matching loss
        feats = [feat12, feat13, feat14, feat15]

        # segmentation map embedding
        segemb = self.embedding(segmap)
        segemb2 = F.interpolate(segemb, seg2.shape[2:], mode='bilinear') 
        segemb3 = F.interpolate(segemb, seg3.shape[2:], mode='bilinear') 
        segemb4 = F.interpolate(segemb, seg4.shape[2:], mode='bilinear') 

        # semantics embedding discriminator score
        pred2 += torch.mul(segemb2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segemb3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segemb4, seg4).sum(dim=1, keepdim=True)

        # concat results from multiple resolutions
        b = pred2.shape[0]
        results = [pred2.view(b, -1), pred3.view(b, -1), pred4.view(b, -1)]
        feats.append(torch.cat(results, dim=1))
        results = feats

        get_intermediate_features = self.ganFeat_loss
        if get_intermediate_features:
            return results
        else:
            return results[-1]
