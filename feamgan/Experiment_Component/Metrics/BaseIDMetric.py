import os
import torch
import numpy as np

from torch.nn import functional as F
from torchvision.models import inception_v3

from feamgan.utils.distUtils import distAllGatherTensor, isMaster
from feamgan.Experiment_Component.Models.Losses.PerceptualLoss import vgg16
from feamgan.Experiment_Component.Models.utils import modelUtils

class BaseIDMetric:
    def __init__(self, is_video, model_dir, dataset_name, dis_model_name):
        self.is_video = is_video
        self.dis_model_name = dis_model_name
        self.save_path = f"{model_dir}/mean_conv"
        self.file_name = f"{dataset_name}_{dis_model_name}.npz"
        self.meters = {}

        self.model = None
        if dis_model_name == "inception_v3":
            os.environ['TORCH_HOME'] = 'models/inception_v3'
            self.model = inception_v3(pretrained=True, transform_input=False, init_weights=False) 
            self.model.fc = torch.nn.Sequential()          
        elif dis_model_name == "vgg16_f":
            os.environ['TORCH_HOME'] = 'models/vgg16'
            self.model = vgg16(layers=['relu_1_2','relu_2_2', 'relu_3_3','relu_4_3','relu_5_3'])  
        elif dis_model_name == "vgg16_f_1":
            os.environ['TORCH_HOME'] = 'models/vgg16'
            self.model = vgg16(layers=['relu_1_2'])  
        elif dis_model_name == "vgg16_f_2":
            os.environ['TORCH_HOME'] = 'models/vgg16'
            self.model = vgg16(layers=['relu_2_2'])  
        elif dis_model_name == "vvgg16_f_3":
            os.environ['TORCH_HOME'] = 'models/vgg16'
            self.model = vgg16(layers=['relu_3_3'])  
        elif dis_model_name == "vgg16_f_4":
            os.environ['TORCH_HOME'] = 'models/vgg16'
            self.model = vgg16(layers=['relu_4_3'])  
        elif dis_model_name == "vgg16_f_5":
            os.environ['TORCH_HOME'] = 'models/vgg16'
            self.model = vgg16(layers=['relu_5_3'])  
        elif dis_model_name == "vgg16_f_45":
            os.environ['TORCH_HOME'] = 'models/vgg16'
            self.model = vgg16(layers=['relu_4_3', 'relu_5_3'])  
        elif dis_model_name == "vgg16_f_ll":
            os.environ['TORCH_HOME'] = 'models/vgg16'
            self.model = vgg16(layers=['relu_5_3'])  
        else:
            raise ValueError(f"{dis_model_name} is not a recognized model name")
  
        self.model = self.model.to('cuda')
        self.model.eval()   

    @torch.no_grad()
    def forwardBatch(self, real_data, fake_data, mode, save_real_prefix=None): 
        assert real_data.shape == fake_data.shape
        if self.is_video: assert len(real_data.shape) == 5
        else: assert len(real_data.shape) == 4

        meter_key = f"{save_real_prefix}_{mode}"
        if meter_key not in self.meters:
            self.meters[meter_key] = {"real": [], "fake": []}

        if save_real_prefix and os.path.exists(f"{self.save_path}/{save_real_prefix}_{mode}_{self.file_name}") and (mode != "train"):
            real_activations = None
        else:
            real_activations = self._getActivations(real_data)  
            self.meters[meter_key]["real"].append(real_activations.cpu())

        fake_activations = self._getActivations(fake_data)
        self.meters[meter_key]["fake"].append(fake_activations.cpu())

    @torch.no_grad()
    def reduceBatches(self, mode, save_real_prefix=None): 
        meter_key = f"{save_real_prefix}_{mode}"
        path = f"{self.save_path}/{save_real_prefix}_{mode}_{self.file_name}"
        if meter_key in self.meters:
            if save_real_prefix and os.path.exists(path) and (mode != "train"):
                print('Load FID real mean and cov from {}'.format(path))
                npz_file = np.load(path)
                real_mean = npz_file['mean']
                real_cov = npz_file['cov']
            else:
                if self.meters[meter_key]["real"]:
                    real_activations = self.meters[meter_key]["real"]
                    real_activations = torch.cat(real_activations)
                    real_activations = distAllGatherTensor(real_activations)
                    if isMaster():
                        real_activations = torch.cat(real_activations).cpu().data.numpy()
                        real_mean, real_cov = self._computeMeanCov(real_activations)
                    if save_real_prefix:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        if isMaster():
                            np.savez(path, mean=real_mean, cov=real_cov)
                else:
                    return None

            if self.meters[meter_key]["fake"]:
                fake_activations = self.meters[meter_key]["fake"]
                fake_activations = torch.cat(fake_activations)
                fake_activations = distAllGatherTensor(fake_activations)
                fid = None
                if isMaster():
                    fake_activations = torch.cat(fake_activations).cpu().data.numpy()
                    fake_mean, fake_cov = self._computeMeanCov(fake_activations)
                    fid = self._calculateDistance(fake_mean, fake_cov, real_mean, real_cov)
            else:
                return None
                
            self.meters[meter_key]["real"] = []
            self.meters[meter_key]["fake"] = []
            return fid
        else: 
            return None

    def _computeMeanCov(self, y):
        m = np.mean(y, axis=0)
        s = np.cov(y, rowvar=False)
        return m, s

    def setModelDir(self, model_dir):
        self.save_path = f"{model_dir}/mean_conv/"

    @torch.no_grad()
    def _getActivations(self, images):
        shape = images.shape 
        activations = []
        if self.is_video:
            images = images.view(shape[0]*shape[1], shape[2], shape[3], shape[4])

        self.model.eval()
        # Clamp the image for models that do not set the output to between
        # -1, 1. For models that employ tanh, this has no effect.
        images.clamp_(-1, 1)
        images = modelUtils.applyImagenetNormalization(images)
        if self.dis_model_name == "inception_v3":      
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=True)
            activations = self.model(images)
            activations = torch.flatten(activations, start_dim=1)
        elif "vgg16_f" in self.dis_model_name:
            activations = self.model(images)
            activations = [torch.flatten(act, start_dim=1) for act in activations]

        return activations

    def _calculateDistance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        raise NotImplementedError('Not implemented')