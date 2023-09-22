import torch 
import random

import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from torch.nn import functional as F

from feamgan.Experiment_Component.Models.utils import modelUtils


@torch.no_grad()
def _similarityCropVgg(image1, image2, seg1, seg2, shape, height, width, sim_threshold, sim_model):
    
    best_top1 = torch.randint(0, shape[-2]-height,(1,))
    best_left1 = torch.randint(0, shape[-1]-width,(1,))

    crops1 = image1[:,:,best_top1:best_top1+height, best_left1:best_left1+width]
    comp_crop1 = modelUtils.applyImagenetNormalization(crops1) 

    vgg1 = sim_model(comp_crop1)[-1]
    vgg1 = torch.mean(vgg1, dim=[2,3]) # "avg pool"
    vgg1 = vgg1.view(1,-1) # "flatten for seq sim"
    vgg1 = F.normalize(vgg1, p=2, dim=1) # For efficiency
    
    best_sim = 0
    top = best_top1
    left = best_left1
    best_top2 = top
    best_left2 = left
    for i in range(0, 1000):
        crops2 = image2[:,:,top:top+height, left:left+width]
        crops2 = modelUtils.applyImagenetNormalization(crops2) 
        vgg2 = sim_model(crops2)[-1]
        vgg2 = torch.mean(vgg2, dim=[2,3]) # "avg pool"
        vgg2 = vgg2.view(1,-1) # "flatten for seq sim"
        #vgg2 = F.normalize(vgg2, p=2, dim=1)
        sim = F.cosine_similarity(vgg1, vgg2)
        if sim > sim_threshold:
            best_sim = sim
            best_top2 = top
            best_left2 = left
            break
        else:
            if sim>best_sim:
                best_sim = sim
                best_top2 = top
                best_left2 = left
            top = torch.randint(0, shape[-2]-height,(1,))
            left = torch.randint(0, shape[-1]-width,(1,))
    return best_top1, best_left1, best_top2, best_left2


@torch.no_grad()
def _similarityCropSegSim(image1, image2, seg1, seg2, shape, height, width, sim_threshold, sim_model):   
    sim_content = torch.eq(seg1, seg2)

    top = torch.randint(0, shape[-2]-height,(1,))
    left = torch.randint(0, shape[-1]-width,(1,))

    best_sim = 0
    best_top = top
    best_left = left
    for i in range(0, 1000):
        crop_content_sim = sim_content[:,:,top:top+height, left:left+width] 
        sim = torch.mean(crop_content_sim.float()) 
        if sim > sim_threshold:
            best_sim = sim
            best_top = top
            best_left = left
            break
        else:
            if sim>best_sim:
                best_sim = sim
                best_top = top
                best_left = left
            top = torch.randint(0, shape[-2]-height,(1,))
            left = torch.randint(0, shape[-1]-width,(1,))
            
    return best_top, best_left, best_top, best_left

@torch.no_grad()
def _similarityCropSegSimRandomResize(image1, image2, seg1, seg2, shape, height, width, sim_threshold, sim_model):   
    sim_content = torch.eq(seg1, seg2)

    scale = random.uniform(0.5, 1.0)
    h = int(float(height)*scale)
    w = int(float(width)*scale)

    top = torch.randint(0, shape[-2]-h,(1,))
    left = torch.randint(0, shape[-1]-w,(1,))

    best_sim = 0
    best_top = top
    best_left = left
    best_h = h
    best_w = w

    for i in range(0, 1000):
        crop_content_sim = sim_content[:,:,top:top+h, left:left+w] 
        sim = torch.mean(crop_content_sim.float()) 
        if sim > sim_threshold:
            best_sim = sim
            best_top = top
            best_left = left
            best_h = h
            best_w = w
            break
        else:
            if sim>best_sim:
                best_sim = sim
                best_top = top
                best_left = left
                best_h = h
                best_w = w
            scale = random.uniform(0.5, 1.0)
            h = int(float(height)*scale)
            w = int(float(width)*scale)
            top = torch.randint(0, shape[-2]-h,(1,))
            left = torch.randint(0, shape[-1]-w,(1,))
            
    return best_top, best_left, best_top, best_left, best_h, best_w


@torch.no_grad()
def _randomCrop(image1, image2, seg1, seg2, shape, height, width, sim_threshold, sim_model):   
    top = torch.randint(0, shape[-2]-height,(1,))
    left = torch.randint(0, shape[-1]-width,(1,))
    return top, left, top, left


def _similarityCropPerconSim(image1, image2, seg1, seg2, shape, height, width, sim_threshold, sim_model):      
    sim_content = torch.eq(seg1, seg2)
    top = torch.randint(0, shape[-2]-height,(1,))
    left = torch.randint(0, shape[-1]-width,(1,))
    best_sim = 0
    best_top = top
    best_left = left
    for i in range(0, 1000):
        crop_content_sim = sim_content[:,:,top:top+height, left:left+width] 

        crop_img_1 = image1[:,:,top:top+height, left:left+width]
        crop_img_2 = image2[:,:,top:top+height, left:left+width]

        vgg1 = modelUtils.applyImagenetNormalization(crop_img_1) 
        vgg1 = sim_model(vgg1)[-1]
        vgg1 = torch.mean(vgg1, dim=[2,3]) # "avg pool"
        vgg1 = vgg1.view(1,-1) # "flatten seq dim"
   
        vgg2 = modelUtils.applyImagenetNormalization(crop_img_2) 
        vgg2 = sim_model(vgg2)[-1]
        vgg2 = torch.mean(vgg2, dim=[2,3]) # "avg pool"
        vgg2 = vgg2.view(1,-1) # "flatten seq dim"

        preceptual_sim = F.cosine_similarity(vgg1, vgg2)
        content_sim = torch.mean(crop_content_sim.float()) 
        sim = (preceptual_sim+content_sim)/2.0

        if sim > sim_threshold:
            best_sim = sim
            best_top = top
            best_left = left
            break
        else:
            if sim>best_sim:
                best_sim = sim
                best_top = top
                best_left = left
            top = torch.randint(0, shape[-2]-height,(1,))
            left = torch.randint(0, shape[-1]-width,(1,))  

    return best_top, best_left, best_top, best_left


class SimilarityCrop(nn.Module):
    def __init__(self, height, width, similarity_type="vgg", **kwargs): # sim_threshold=0.5, sim_model
        super().__init__()
        self.h = height
        self.w = width
        self.kwargs = kwargs
        self.similarity_type = similarity_type

        if similarity_type == "random":
            self.sim_fn = _randomCrop
        elif similarity_type == "vgg":
            self.sim_fn = _similarityCropVgg
        elif similarity_type == "segSim":
            self.sim_fn = _similarityCropSegSim
        elif similarity_type == "segSimRS":
            self.sim_fn = _similarityCropSegSimRandomResize
            self.resize_frames = transforms.Resize(size=(self.h, self.w), interpolation=InterpolationMode.BICUBIC)
            self.resize_other = transforms.Resize(size=(self.h, self.w), interpolation=InterpolationMode.NEAREST)
        elif similarity_type == "perconSim":
            self.sim_fn = _similarityCropPerconSim
        else:
            raise ValueError(f"{similarity_type} is not a recognized similarity type")

    @torch.no_grad()
    def forward(self, inputs1, inputs2):
        shape = inputs1[0].shape
        crops1 = [[] for _ in range(len(inputs1))]
        crops2 = [[] for _ in range(len(inputs2))]
        for b in range(shape[0]):  
            if self.similarity_type == "segSimRS":
                best_top1, best_left1, best_top2, best_left2, best_h, best_w = self.sim_fn(inputs1[0][b], inputs2[0][b], inputs1[1][b], inputs2[1][b], 
                                                                       shape, self.h, self.w, **self.kwargs)
                f_img = inputs1[0][b,:,:,best_top1:best_top1+best_h, best_left1:best_left1+best_w] 
                crops1[0].append(torch.unsqueeze(self.resize_frames(f_img),0))
                for i in range(1,len(inputs1)):
                    o_img = inputs1[i][b,:,:,best_top1:best_top1+best_h, best_left1:best_left1+best_w]
                    crops1[i].append(torch.unsqueeze(self.resize_other(o_img),0))

                f_img = inputs2[0][b,:,:,best_top1:best_top1+best_h, best_left1:best_left1+best_w]
                crops2[0].append(torch.unsqueeze(self.resize_frames(f_img),0))
                for i in range(1,len(inputs2)):
                    o_img = inputs2[i][b,:,:,best_top1:best_top1+best_h, best_left1:best_left1+best_w]
                    crops2[i].append(torch.unsqueeze(self.resize_other(o_img),0))
            else:
                best_top1, best_left1, best_top2, best_left2 = self.sim_fn(inputs1[0][b], inputs2[0][b], inputs1[1][b], inputs2[1][b], 
                                                                       shape, self.h, self.w, **self.kwargs)
                for i, img in enumerate(inputs1):
                    crops1[i].append(torch.unsqueeze(img[b,:,:,best_top1:best_top1+self.h, best_left1:best_left1+self.w],0))
                for i, img in enumerate(inputs2):
                    crops2[i].append(torch.unsqueeze(img[b,:,:,best_top2:best_top2+self.h, best_left2:best_left2+self.w],0)) 

        for i in range(len(inputs1)):
            crops1[i] = torch.cat(crops1[i],0)
        for i in range(len(inputs2)):
            crops2[i] = torch.cat(crops2[i],0)
        return crops1, crops2


def similarityCropSmallPatches(inputs1, inputs2, erase_mask, crop_factor_h=8, crop_factor_w=8, sim_threshold=0.5, max_batch_size=32):
    shape = erase_mask.shape
    w = shape[-1]//crop_factor_w
    h = shape[-2]//crop_factor_h
    
    erase_mask = erase_mask.unfold(-2, h, h).unfold(-2, w, w).permute(1,0,2,3,4,5).reshape(1, shape[0]*crop_factor_w*crop_factor_h,h,w).permute(1,0,2,3)
 
    select = torch.mean(erase_mask, dim=[1,2,3])
    select[select<sim_threshold] = 0
    select = select.nonzero(as_tuple=True)

    inputs1 = [x.unfold(-2, h, h).unfold(-2, w, w).permute(1,0,2,3,4,5).reshape(x.shape[1], shape[0]*crop_factor_w*crop_factor_h,h,w).permute(1,0,2,3) for x in inputs1]
    inputs2 = [x.unfold(-2, h, h).unfold(-2, w, w).permute(1,0,2,3,4,5).reshape(x.shape[1], shape[0]*crop_factor_w*crop_factor_h,h,w).permute(1,0,2,3) for x in inputs2]

    inputs1 = [x[select[0:max_batch_size]] for x in inputs1]
    inputs2 = [x[select[0:max_batch_size]] for x in inputs2]
    
    return inputs1, inputs2