import torch
import os
import numpy as np

from torch.nn import functional as F

from feamgan.utils.distUtils import distAllGatherTensor, isMaster
from feamgan.Experiment_Component.Metrics.KidMetric import KidMetric

class SkvdMetric(KidMetric):
    def __init__(self, is_video, model_dir, dataset_name, dis_model_name="vgg16_f_ll"):
        super(SkvdMetric, self).__init__(is_video, model_dir, dataset_name, dis_model_name)
        self.nr_no_matching_pairs = 0
        self.save_path = f"{model_dir}/skvd_real_act_des"
        self.random_gen = np.random.default_rng(seed=42)
        
    @torch.no_grad()
    def forwardBatch(self, real_data, fake_data, mode, save_real_prefix=None): 
        assert real_data[0].shape == fake_data[0].shape 
        assert real_data[1].shape == fake_data[1].shape
        if self.is_video: 
            assert len(real_data[0].shape) == 5
            assert len(real_data[1].shape) == 5
        else: 
            assert len(real_data[0].shape) == 4
            assert len(real_data[1].shape) == 4
      
        if save_real_prefix and os.path.exists(f"{self.save_path}/{save_real_prefix}_{mode}_{self.file_name}") and (mode != "train"):
            real_semantic_description = None
            real_activations = None
        else:
            real_img_patched, real_seg_patched = self._oneEightCrop(real_data[0], real_data[1])
            real_activations = self._getActivations(real_img_patched)
            real_semantic_description = self._getSemanticDescriptions(real_seg_patched)
              
        fake_img_patched, fake_seg_patched = self._oneEightCrop(fake_data[0], fake_data[1])
        fake_activations = self._getActivations(fake_img_patched)
        fake_semantic_description = self._getSemanticDescriptions(fake_seg_patched)

        meter_key = f"{save_real_prefix}_{mode}"
        if meter_key not in self.meters:
            self.meters[meter_key] = {"real_des": [], "real_act": [[] for _ in range(len(fake_activations))], "fake_des": [], "fake_act": [[] for _ in range(len(fake_activations))]}

        if not(save_real_prefix and os.path.exists(f"{self.save_path}/{save_real_prefix}_{mode}_{self.file_name}") and (mode != "train")):
            self.meters[meter_key]["real_des"].append(real_semantic_description.cpu())
            for i, real_act in enumerate(real_activations):
                self.meters[meter_key]["real_act"][i].append(real_act.cpu())

        self.meters[meter_key]["fake_des"].append(fake_semantic_description.cpu())
        for i, fake_act in enumerate(fake_activations):
            self.meters[meter_key]["fake_act"][i].append(fake_act.cpu())
        

    def _oneEightCrop(self, img, seg):
        img_patches = []
        seg_patches = []

        h = img.shape[-2]
        w = img.shape[-1]
        p = int(np.floor(np.sqrt((w*h)/8.0)))   
        p2 = p//2 

        for i in range(img.shape[0]):
            hr = self.random_gen.integers(low=p2, high=h-p2)
            wr = self.random_gen.integers(low=p2, high=w-p2)
            if self.is_video:
                img_patches.append(img[[i],:,:,hr-p2: hr+p2, wr-p2:wr+p2])
                seg_patches.append(seg[[i],:,:,hr-p2: hr+p2, wr-p2:wr+p2])
            else:
                img_patches.append(img[[i],:,hr-p2:hr+p2, wr-p2:wr+p2])
                seg_patches.append(seg[[i],:,hr-p2:hr+p2, wr-p2:wr+p2])
        img_patches = torch.cat(img_patches)
        seg_patches = torch.cat(seg_patches)

        return img_patches,seg_patches

    def _getSemanticDescriptions(self, seg_img):
        shape = seg_img.shape 
        if self.is_video:
            seg_img = seg_img.view(shape[0]*shape[1], *shape[2:])

        seg_des = F.interpolate(seg_img.byte(), size=(16, 16), mode='nearest')
        seg_des = torch.flatten(seg_des, start_dim=1)
        return seg_des

    def _calcSimilatiry(self, seg_des_1, seg_des_2):
        return np.mean(np.equal(seg_des_1, seg_des_2).astype(float), axis=1) 
    
    def _getMatchingPairs(self, fake_des, fake_act, real_des, real_act, sim_threshold=0.5):
        self.nr_no_matching_pairs = 0
        fake_activation_paired = [[] for i in range(len(fake_act))]
        real_activation_paired = [[] for i in range(len(fake_act))]

        for f in range(fake_des.shape[0]):
            found_index = -1
            sims = self._calcSimilatiry(real_des, fake_des[f])
            nearest_neighbour_index = np.argmax(sims)
            if sims[nearest_neighbour_index] > sim_threshold:
                found_index = nearest_neighbour_index 
   
            if found_index > -1:
                for a in range(len(fake_act)):
                    fake_activation_paired[a].append(np.expand_dims(fake_act[a][f], axis=0))
                    real_activation_paired[a].append(np.expand_dims(real_act[a][found_index], axis=0))
            else:
                self.nr_no_matching_pairs += 1
                
        print(f"No matching pairs found for {self.nr_no_matching_pairs} samples")       
                
        for a in range(len(fake_activation_paired)): 
            fake_activation_paired[a] = np.concatenate(fake_activation_paired[a])
            real_activation_paired[a] = np.concatenate(real_activation_paired[a])
        
        return fake_activation_paired, real_activation_paired

    @torch.no_grad()
    def reduceBatches(self, mode, save_real_prefix=None): 
        meter_key = f"{save_real_prefix}_{mode}"
        path = f"{self.save_path}/{save_real_prefix}_{mode}_{self.file_name}"
        if meter_key in self.meters:
            if save_real_prefix and os.path.exists(path) and (mode != "train"):
                print('Load sKVD seg_des and activations from {}'.format(path))
                npz_file = np.load(path)
                real_des = npz_file['real_des']
                real_act = npz_file['real_act']
            else:
                if self.meters[meter_key]["real_des"]:
                    real_des = self.meters[meter_key]["real_des"]
                    real_des = torch.cat(real_des)
                    real_des = distAllGatherTensor(real_des)
           
                    real_act = self.meters[meter_key]["real_act"]
                    real_act = [torch.cat(act) for act in real_act]
                    real_act = [distAllGatherTensor(act) for act in real_act]
                    if isMaster():
                        real_des = torch.cat(real_des).cpu().data.numpy()
                        real_act = [torch.cat(act).cpu().data.numpy() for act in real_act]
                    if save_real_prefix:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        if isMaster():
                            np.savez(path, real_des=real_des, real_act=real_act)
                else:
                    return None

            if self.meters[meter_key]["fake_des"]:
                fake_des = self.meters[meter_key]["fake_des"]
                fake_des = torch.cat(fake_des)
                fake_des = distAllGatherTensor(fake_des)
 
                fake_act = self.meters[meter_key]["fake_act"]
                fake_act = [torch.cat(act) for act in fake_act] 
                fake_act = [distAllGatherTensor(act) for act in fake_act] 
                kids = None
                if isMaster():
                    fake_des = torch.cat(fake_des).cpu().data.numpy()
                    fake_act = [torch.cat(act).cpu().data.numpy() for act in fake_act] 
                    fake_activations_paired, real_activations_paired = self._getMatchingPairs(fake_des, fake_act, real_des, real_act)
                    kids = {}
                    i = 1
                    for fake_activations, real_activations in zip(fake_activations_paired, real_activations_paired):
                        if fake_activations.size != 0:
                            mmd, mmd_vars = self._polynomial_mmd_averages(fake_activations, real_activations)
                            kids[i] = mmd.mean()
                        else:
                            kids[i] = None
                        i+=1
            else:
                return None
                     
            self.meters[meter_key] = {"real_des": [], "real_act": [[] for _ in range(len(self.meters[meter_key]["fake_act"]))], 
                                      "fake_des": [], "fake_act": [[] for _ in range(len(self.meters[meter_key]["fake_act"]))]}
            return kids
        else: 
            return None

