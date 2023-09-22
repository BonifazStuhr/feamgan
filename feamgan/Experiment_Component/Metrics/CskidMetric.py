import torch
import os
import pandas as pd
import numpy as np

from torch.nn import functional as F

from feamgan.utils.distUtils import distAllGatherTensor, isMaster
from feamgan.Experiment_Component.Metrics.KidMetric import KidMetric

class CskidMetric(KidMetric):
    def __init__(self, is_video, model_dir, dataset_name, dis_model_name="vgg16_f_ll"):
        super(CskidMetric, self).__init__(is_video, model_dir, dataset_name, dis_model_name)
        self.nr_no_matching_pairs = 0
        self.p = None 
        self.save_path = f"{model_dir}/cskid_real_cls_acts"

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
            147:"mountain-merged", 147:"rock-merged",
            174:"vegetation", 
            175:"bicycle",176:"car",177:"vehicle", 178:"motorcycle",179:"airplane",180:"bus",181:"train",182:"truck",183:"trailer", 184:"boat", 185:"slow_wheeled_object",
            186:"river",187:"sea",188:"water-other",189:"water_1",190:"water_2",
            191:"wall",
            192:"window-other", 193:"window-blind",
            194:"unlabeled"
        }
      
        #self.class_id_to_name = {class_id:class_id_to_name[class_id]}
        #print(self.class_id_to_name)
       
        self.categories = {
            "building-parent": ["tunnel","bridge","building-parent","building","ceiling-merged"],
            "ground": ["gravel","platform","railroad","road","pavement-merged","ground","terrain"], 
            "person-parent": ["person","rider_other","bicyclist","motorcyclist"],
            "roadside-obj.": ["streetlight","road_barrier","mailbox","cctv_camera","junction_box","traffic_sign","traffic_light","fire_hydrant","parking_meter","bench","bike_rack","billboard"],
            "sky":["sky"],
            "structural":["fence","guard_rail"],
            "mountain":["mountain-merged","rock-merged"],
            "vegetation":["vegetation"], 
            "vehicle":["bicycle","car","vehicle","motorcycle","airplane","bus","train","truck","trailer","boat","slow_wheeled_object"],
            "water":["river","sea","water-other","water_1","water_2"],
            "wall":["wall"],
            "window":["window-other","window-blind"],
            "unlabeled":["unlabeled"]
        }
        """
        self.name_to_class_ids = {
            "building-parent": [31, 32, 33, 35, 36],
            "road": [98], 
            "ground": [94,95,97,100,101,102], 
            "person-parent": [125, 126, 127, 128],
            "roadside-obj": [130,131,132,133,134,135,136,137,138,139,140,141],
            "sky": [142],
            "structural": [144,146,191],
            "mountain": [147,147],
            "vegetation": [174], 
            "vehicle": [175,176,177,178,179,180,181,182,183,184,185],
            "water": [186,187,188,189,190],
        }

        # Only use a subset of the above:
        #125:"person", 
        """
        self.class_id_to_name = {
            35:"building",
            98:"road", 101:"ground",102:"terrain", 
            142:"sky",
            147:"mountain-merged",
            174:"vegetation", 
            175:"bicycle",176:"car", 178:"motorcycle",180:"bus",
            194:"unlabeled"
        }
        
        #"person-parent": ["person"],
        self.categories = {
            "building-parent": ["building"],
            "ground": ["road","ground","terrain"], 
            "sky":["sky"],
            "mountain":["mountain-merged"],
            "vegetation":["vegetation"], 
            "vehicle":["bicycle","car","motorcycle","bus"],
            "unlabeled":["unlabeled"],
            "rest":["rest"]
        }
        """
        
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
            real_class_imgs = self._getClassImages(real_data[0], real_data[1])
            real_class_activations = self._getClassActivations(real_class_imgs)
              
        fake_class_imgs = self._getClassImages(fake_data[0], fake_data[1])
        fake_class_activations = self._getClassActivations(fake_class_imgs)

        meter_key = f"{save_real_prefix}_{mode}"
        if meter_key not in self.meters:
            self.meters[meter_key] = {"fake_cls_acts":{}, "real_cls_acts":{}}

        for cls in fake_class_activations:
            if cls not in self.meters[meter_key]["fake_cls_acts"]:
                self.meters[meter_key]["fake_cls_acts"].update({cls: [[] for _ in range(len(fake_class_activations[cls]))]})
        for cls in real_class_activations:
            if cls not in self.meters[meter_key]["real_cls_acts"]:
                self.meters[meter_key]["real_cls_acts"].update({cls: [[] for _ in range(len(real_class_activations[cls]))]})

        if not(save_real_prefix and os.path.exists(f"{self.save_path}/{save_real_prefix}_{mode}_{self.file_name}") and (mode != "train")):
            for cls in real_class_activations:
                for i, real_act in enumerate(real_class_activations[cls]):
                    self.meters[meter_key]["real_cls_acts"][cls][i].append(real_act.cpu())

        for cls in fake_class_activations:
            for i, fake_act in enumerate(fake_class_activations[cls]):
                self.meters[meter_key]["fake_cls_acts"][cls][i].append(fake_act.cpu())
            
    def _getClassActivations(self, class_images):
        class_activations = {}
        for cls in class_images:
            class_activations[cls] = self._getActivations(class_images[cls])
        return class_activations

    def _getClassImagesOld(self, img, seg):
        class_images = {}

        overall_erase_mask = None
        for id in self.class_id_to_name:
            erase_mask = torch.eq(seg, id)
            masked_image = img * erase_mask
            overall_erase_mask = erase_mask if overall_erase_mask is None else overall_erase_mask+erase_mask
            class_images[self.class_id_to_name[id]] = masked_image
            from torchvision.utils import save_image
            save_image(masked_image, f'{self.class_id_to_name[id]}.png')

        class_images["rest"] =  img * ~overall_erase_mask.bool()
        from torchvision.utils import save_image
        save_image(class_images["rest"], 'rest.png')
        save_image(img * overall_erase_mask.bool(), 'cskid_measured_image.png')
        return class_images

    def _getClassImages(self, img, seg):
        class_images = {}

        overall_mask = None
        for name in self.name_to_class_ids:
            mask = None
            for id in self.name_to_class_ids[name]:
                mask = torch.eq(seg, id) if mask is None else mask+torch.eq(seg, id)
            masked_image = img * mask
            overall_mask = mask if overall_mask is None else overall_mask+mask
            class_images[name] = masked_image      
            #from torchvision.utils import save_image
            #save_image(masked_image, f'{name}.png')

        class_images["rest"] =  img * ~overall_mask
        #from torchvision.utils import save_image
        #save_image(class_images["rest"], 'rest.png')
        #save_image(img * overall_mask, 'cskid_measured_image.png')
        return class_images
    
    @torch.no_grad()
    def reduceBatches(self, mode, save_real_prefix=None): 
        meter_key = f"{save_real_prefix}_{mode}"
        path = f"{self.save_path}/{save_real_prefix}_{mode}_{self.file_name}"
        if meter_key in self.meters:
            if save_real_prefix and os.path.exists(path) and (mode != "train"):
                print('Load CSKID activations from {}'.format(path))
                npz_file = np.load(path)
                real_cls_act = npz_file['real_cls_acts']
            else:
                if self.meters[meter_key]["real_cls_acts"]:           
                    real_cls_act = self.meters[meter_key]["real_cls_acts"]
                    for cls in real_cls_act:
                        real_cls_act[cls] = [torch.cat(act) for act in real_cls_act[cls]]
                        real_cls_act[cls] = [distAllGatherTensor(act) for act in real_cls_act[cls]]
                        if isMaster():
                            real_cls_act[cls] = [torch.cat(act).cpu().data.numpy() for act in real_cls_act[cls]]
                    if save_real_prefix:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        if isMaster():
                            np.savez(path, real_cls_act=real_cls_act)
                else:
                    return None

            if self.meters[meter_key]["fake_cls_acts"]:
                fake_cls_act = self.meters[meter_key]["fake_cls_acts"]
                class_kids = {}
                for cls in fake_cls_act:
                    fake_cls_act[cls] = [torch.cat(act) for act in fake_cls_act[cls]] 
                    fake_cls_act[cls] = [distAllGatherTensor(act) for act in fake_cls_act[cls]] 
                    class_kids[cls] = {}
                    if isMaster():
                        fake_cls_act[cls] = [torch.cat(act).cpu().data.numpy() for act in fake_cls_act[cls]] 
                        i = 1
                        if fake_cls_act[cls] and real_cls_act[cls]:
                            for fake_activations, real_activations in zip(fake_cls_act[cls], real_cls_act[cls]):
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
                """
                class_kids["categories"] = {}
                for cat in self.categories: 
                    cat_kids = []
                    for cls in self.categories[cat]:
                        if cls in class_kids:
                            cat_kids.append(class_kids[cls])
                    df = pd.DataFrame(cat_kids)
                    class_kids["categories"][cat] = dict(df.mean())
                """

            self.meters[meter_key]["fake_cls_acts"] = {}
            self.meters[meter_key]["real_cls_acts"] = {}

            return class_kids
        else: 
            return None

