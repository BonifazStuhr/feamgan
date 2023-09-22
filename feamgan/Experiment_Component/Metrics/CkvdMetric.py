import torch
import os
import pandas as pd
import numpy as np

from torch.nn import functional as F
from utils.distUtils import distAllGatherTensor, isMaster
from Experiment_Component.Metrics.KidMetric import KidMetric

class CkvdMetric(KidMetric):
    def __init__(self, is_video, model_dir, dataset_name, dis_model_name="vgg16_f_ll"):
        super(CkvdMetric, self).__init__(is_video, model_dir, dataset_name, dis_model_name)
        self.nr_no_matching_pairs = {}
        self.save_path = f"{model_dir}/ckvd_real_cls_acts"
        self.random_gen = np.random.default_rng(seed=42)
        # Ids and class names can be foud here:
        # https://github.com/mseg-dataset/mseg-api/blob/master/mseg/class_remapping_files/MSeg_master.tsv#L100
        # True id is id_form_table-2
        """
        self.class_id_to_name = {
            31:"tunnel", 32:"bridge", 33:"building-parent", 35:"building", 36:"ceiling-merged",
            94:"gravel",95:"platform",97:"railroad",98:"road",100:"pavement-merged",101:"ground",102:"terrain", 
            125:"person", 126:"rider_other", 127:"bicyclist", 128:"motorcyclist",
            130:"streetlight",131:"road_barrier",132:"mailbox",133:"cctv_camera",134:"junction_box",135:"traffic_sign",136:"traffic_light",137:"fire_hydrant",138:"parking_meter",139:"bench",140:"bike_rack",141:"billboard",
            142:"sky",
            144:"fence",146:"guard_rail",
            147:"mountain-merged", 148:"rock-merged",
            174:"vegetation", 
            175:"bicycle",176:"car",177:"autorickshaw", 178:"motorcycle",179:"airplane",180:"bus",181:"train",182:"truck",183:"trailer", 184:"boat", 185:"slow_wheeled_object",
            186:"river",187:"sea",188:"water-other",189:"water_1",190:"water_2",
            191:"wall",
            192:"window-other", 193:"window-blind",
            194:"unlabeled"
        }
        """
        # Due to compositional limitations, we use only a subset of the classes
        self.name_to_class_ids = {
            "sky": [142],
            "ground": [94,95,97,100,101], 
            "road": [98], 
            "terrain": [102],
            "vegetation": [174], 
            "roadside-obj.": [130,131,132,133,134,135,136,137,138,139,140,141],
            "building-parent": [31,32,33,35,36],
            "person-parent": [125,126,127,128],        
            "vehicle": [175,176,177,178,180,181,182,183,185],
        }

        
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
            real_class_activations = None
        else:
            real_img_patched, real_seg_patched = self._oneEightCrop(real_data[0], real_data[1])
            real_class_imgs = self._getClassImages(real_img_patched, real_seg_patched, p="r")
            real_class_activations = self._getClassActivations(real_class_imgs)
            real_class_semantic_descriptions = self._getClassSemanticDescriptions(real_class_imgs, real_seg_patched)

        fake_img_patched, fake_seg_patched = self._oneEightCrop(fake_data[0], fake_data[1])   
        fake_class_imgs = self._getClassImages(fake_img_patched, fake_seg_patched, p="f")
        fake_class_activations = self._getClassActivations(fake_class_imgs)
        fake_class_semantic_descriptions = self._getClassSemanticDescriptions(fake_class_imgs, fake_seg_patched)

        meter_key = f"{save_real_prefix}_{mode}"
        if meter_key not in self.meters:
            self.meters[meter_key] = {"fake_cls_des":{}, "fake_cls_acts":{}, "real_cls_des":{}, "real_cls_acts":{}}

        for cls in fake_class_activations:
            if cls not in self.meters[meter_key]["fake_cls_des"]:
                self.meters[meter_key]["fake_cls_des"].update({cls: []})
            if cls not in self.meters[meter_key]["fake_cls_acts"]:
                self.meters[meter_key]["fake_cls_acts"].update({cls: [[] for _ in range(len(fake_class_activations[cls]))]})
        for cls in real_class_activations:
            if cls not in self.meters[meter_key]["real_cls_des"]:
                self.meters[meter_key]["real_cls_des"].update({cls: []})
            if cls not in self.meters[meter_key]["real_cls_acts"]:
                self.meters[meter_key]["real_cls_acts"].update({cls: [[] for _ in range(len(real_class_activations[cls]))]})
           
        if not(save_real_prefix and os.path.exists(f"{self.save_path}/{save_real_prefix}_{mode}_{self.file_name}") and (mode != "train")):
            for cls in real_class_activations:
                for i, real_act in enumerate(real_class_activations[cls]):
                    self.meters[meter_key]["real_cls_acts"][cls][i].append(real_act.cpu())
                self.meters[meter_key]["real_cls_des"][cls].append(real_class_semantic_descriptions[cls].cpu())

        for cls in fake_class_activations:
            for i, fake_act in enumerate(fake_class_activations[cls]):
                self.meters[meter_key]["fake_cls_acts"][cls][i].append(fake_act.cpu())
            self.meters[meter_key]["fake_cls_des"][cls].append(fake_class_semantic_descriptions[cls].cpu())
    
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
    
    def _getClassSemanticDescriptions(self, class_images, seg):
        class_seg_des = {}
        for cls in class_images:    
            # Per image we only need this once, therefore this could be implementend more efficently
            class_seg_des[cls] = self._getSemanticDescription(seg) 
        return class_seg_des

    def _getSemanticDescription(self, seg_img):
        shape = seg_img.shape 
        if self.is_video:
            seg_img = seg_img.view(shape[0]*shape[1], *shape[2:])

        seg_des = F.interpolate(seg_img.byte(), size=(16, 16), mode='nearest')
        seg_des = torch.flatten(seg_des, start_dim=1)
        return seg_des
    
    def _getClassActivations(self, class_images):
        class_activations = {}
        for cls in class_images:
            class_activations[cls] = self._getActivations(class_images[cls])
        return class_activations

    def _getClassImages(self, img, seg, threshold=0.05, p=""):
        class_images = {}
        
        overall_mask = None
        for name in self.name_to_class_ids:
            mask = None
            for id in self.name_to_class_ids[name]:
                mask = torch.eq(seg, id) if mask is None else mask+torch.eq(seg, id)
            overall_mask = mask if overall_mask is None else overall_mask+mask    
            if torch.mean(mask.float()) > threshold:
                #if name is "person-parent": print(f"{p}_{name}:{torch.mean(mask.float())}") 
                #if name is "roadside-obj.": print(f"{p}_{name}:{torch.mean(mask.float())}") 
                masked_image = img * mask
                class_images[name] = masked_image     
            #from torchvision.utils import save_image
            #save_image(masked_image, f'{name}_img.png')

        class_images["rest"] = img * ~overall_mask
        #from torchvision.utils import save_image
        #save_image(class_images["rest"], 'rest_img.png')
        #save_image(img * overall_mask, 'ckvd_measured_image.png')
        return class_images
    
    def _calcSimilatiry(self, seg_des_1, seg_des_2):
        return np.mean(np.equal(seg_des_1, seg_des_2).astype(float), axis=1) 
    
    def _getMatchingPairs(self, fake_des, fake_act, real_des, real_act, cls, sim_threshold=0.5):
        self.nr_no_matching_pairs[cls] = 0
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
                self.nr_no_matching_pairs[cls] += 1

        print(f"No matching pairs found for {self.nr_no_matching_pairs[cls]} samples for class {cls} ")       
                
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
                print('Load cKVD activations from {}'.format(path))
                npz_file = np.load(path)
                real_cls_des = npz_file['real_cls_des']
                real_cls_act = npz_file['real_cls_acts']
            else:
                if self.meters[meter_key]["real_cls_acts"]:           
                    real_cls_des = self.meters[meter_key]["real_cls_des"]
                    real_cls_act = self.meters[meter_key]["real_cls_acts"]
                    for cls in real_cls_act:
                        real_cls_des[cls] = torch.cat(real_cls_des[cls])
                        real_cls_des[cls] = distAllGatherTensor(real_cls_des[cls])

                        real_cls_act[cls] = [torch.cat(act) for act in real_cls_act[cls]]
                        real_cls_act[cls] = [distAllGatherTensor(act) for act in real_cls_act[cls]]
                        if isMaster():
                            real_cls_des[cls] = torch.cat(real_cls_des[cls]).cpu().data.numpy()
                            real_cls_act[cls] = [torch.cat(act).cpu().data.numpy() for act in real_cls_act[cls]]
                    if save_real_prefix:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        if isMaster():
                            np.savez(path, real_cls_act=real_cls_act)
                else:
                    return None

            if self.meters[meter_key]["fake_cls_acts"]:
                fake_cls_des = self.meters[meter_key]["fake_cls_des"]
                fake_cls_act = self.meters[meter_key]["fake_cls_acts"]

                class_kids = {}
                for cls in fake_cls_act:  
                    print(f"Reducing batch for class: {cls}")
                    fake_cls_des[cls] = torch.cat(fake_cls_des[cls])
                    fake_cls_des[cls] = distAllGatherTensor(fake_cls_des[cls])

                    fake_cls_act[cls] = [torch.cat(act) for act in fake_cls_act[cls]] 
                    fake_cls_act[cls] = [distAllGatherTensor(act) for act in fake_cls_act[cls]] 

                    if isMaster():
                        fake_cls_des[cls] = torch.cat(fake_cls_des[cls]).cpu().data.numpy()
                        fake_cls_act[cls] = [torch.cat(act).cpu().data.numpy() for act in fake_cls_act[cls]]
                        fake_activations_paired, real_activations_paired = self._getMatchingPairs(fake_cls_des[cls], fake_cls_act[cls], real_cls_des[cls], real_cls_act[cls], cls)
                        class_kids[cls] = {}
                        i = 1
                        if fake_activations_paired and real_activations_paired:
                            for fake_activations, real_activations in zip(fake_activations_paired, real_activations_paired):
                                if fake_activations.size != 0:
                                    mmd, mmd_vars = self._polynomial_mmd_averages(fake_activations, real_activations)
                                    class_kids[cls][i] = mmd.mean()
                                else:
                                    class_kids[cls][i] = None
                                i+=1
            else:
                return None  
            if isMaster():
                df = pd.DataFrame(class_kids.values())
                class_kids["all"] = dict(df.mean())

            self.meters[meter_key] = {"real_cls_des": {}, "real_cls_acts": {}, 
                                      "fake_cls_des": {}, "fake_cls_acts": {}}
            
            return class_kids
        else: 
            return None