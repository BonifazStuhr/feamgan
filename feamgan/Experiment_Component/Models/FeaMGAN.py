import torch   
import wandb
import os
import copy

from feamgan.utils.visualizationUtils import formatVideo
from feamgan.utils.semanticSegmentationUtils import labelMapToOneHot, getNumSemanticLabelIds, labelMapToColor
from feamgan.Experiment_Component.Models.utils import modelUtils
from feamgan.Experiment_Component.Models.Sampling.discriminatorMask import discriminatorMask
from feamgan.Experiment_Component.Models.Sampling.SimilarityCrop import SimilarityCrop, similarityCropSmallPatches
from feamgan.Experiment_Component.Models.BaseModel import BaseModel
from feamgan.Experiment_Component.Models.Backbones.Generators.FeaMGenerator import FeaMGenerator
from feamgan.Experiment_Component.Models.Backbones.Discriminators.MultiscaleDiscriminator import MultiscaleDiscriminator
from feamgan.Experiment_Component.Models.Losses.GANLoss import GANLoss
from feamgan.Experiment_Component.Models.Losses.PerceptualLoss import PerceptualLoss
from feamgan.Experiment_Component.Models.Losses.PerceptualLoss import vgg16
from feamgan.Experiment_Component.Models.Losses.gradientPenalty import tee_loss, real_penalty

def disabled_train(self, mode=True):
    return self

class FeaMGAN(BaseModel):

    def __init__(self, model_config, dataset_config, input_shape, output_shape, sequence_length):   
        
        self.is_train = model_config["isTrain"]  
        self.verbose = model_config["verbose"]  

        # Shapes
        self.input_shape_in = input_shape
        self.output_shape_in = output_shape
        if self.is_train: 
            # Sampling Shapes and Threshold
            self.sim_crop_type = model_config["optimization"]["simCropType"]
            self.sim_crop_height = model_config["optimization"]["simCropHeight"]
            self.sim_crop_width = model_config["optimization"]["simCropWidth"]
            self.sim_crop_threshold = model_config["optimization"]["simCropThreshold"]
            self.use_sim_crop_dis = model_config["optimization"]["useSimCropDis"]
            self.lambdaRealPenalty = model_config["optimization"]["lambdaRealPenalty"]
            if self.use_sim_crop_dis:
                self.crop_factor_h = model_config["optimization"]["cropFactorW"]
                self.crop_factor_w = model_config["optimization"]["cropFactorH"]
                self.sim_crop_dis_max_batch_size = model_config["optimization"]["simCropDisMaxBatchSize"]

            input_shape = [3, self.sim_crop_height, self.sim_crop_width]
            output_shape = [3, self.sim_crop_height, self.sim_crop_width]
        
        super(FeaMGAN, self).__init__(model_config, dataset_config, input_shape, output_shape, sequence_length)

        if self.is_train: 
            # Initialization 
            self.init_type = self.model_config["initializationType"]
            self.init_variance = self.model_config["initializationVariance"]

            # Optimization
            self.optimizer_type = model_config["optimization"]["optimizerType"] 
            self.lr_scheduler = model_config["optimization"]["lrScheduler"]["name"]
            if self.lr_scheduler == "stepLR":
                self.niter = model_config["optimization"]["lrScheduler"]["learningRateDecayStartingStep"] 
                self.niter_decay = model_config["optimization"]["lrScheduler"]["learningRateDecayAfterSteps"] 
            self.gan_mode = model_config["optimization"]["ganMode"]
            self.lr_g = model_config["optimization"]["learningRateG"] 
            self.lr_d = model_config["optimization"]["learningRateD"] 
            self.beta1 = model_config["optimization"]["beta1"] 
            self.beta2 = model_config["optimization"]["beta2"] 
            self.tTUR = model_config["optimization"]["TTUR"] 

            # Optimization - Perceptual Loss
            self.lambda_perceptual = model_config["optimization"]["lambdaPerceptual"] 
            self.perceptual_loss_type = model_config["optimization"]["perceptualLossType"]

            # Optimization - Discriminators
            self.c_discriminator = model_config["optimization"]["cDiscriminator"]
            self.use_discriminator_mask = model_config["optimization"]["useDiscriminatorMask"]
            self.lambda_gan_feat = model_config["optimization"]["lambdaGanFeat"]
            self.FloatTensor = torch.cuda.FloatTensor 

            self.dis_input_nc = 3
            if self.c_discriminator:
                self.dis_input_nc += getNumSemanticLabelIds("mseg", are_train_ids=False)

            self.dis_names = ["main"]
            if self.use_sim_crop_dis: 
                self.dis_names.append("sim_crop")
 
        # Generator Input Dimensions
        self.instance = dataset_config["hasInstance"]
        self.gen_output_nc = 3
        self.gen_style_nc = 3
        self.gen_content_nc = 0
        for input_type in self.input_types:
            if input_type == "frames":
                self.gen_content_nc += 3
            elif input_type == "segmentations":
                self.gen_content_nc += getNumSemanticLabelIds(dataset_config["nameOfDataset"], are_train_ids=False)
                if self.instance:
                    self.gen_content_nc += 1

        # Indicies of the Segmentations in the Input
        self.seg_index = 0
        if 1 < len(self.input_types) and "segmentations" in self.input_types:
            self.seg_index = 1
        if 2 < len(self.input_types) and "segmentations" in self.input_types:
            self.seg_index = 2
            
        self.netG, self.netD, self.netDSim, self.netS = self.defineNetworks()
       
        # Create Sampling 
        if self.is_train and self.sim_crop_type:
            self.simCrop = SimilarityCrop(self.sim_crop_height, self.sim_crop_width, self.sim_crop_type, sim_threshold=self.sim_crop_threshold, sim_model=self.netS)
        
        # Create all Criteria
        self.use_video_loss, self.criterionGAN, self.criterionFeat, self.criterionPerceptual = self.defineLoss()
        self.gan_factor = 0.5 if self.use_video_loss else 1.0

    def defineLoss(self): 
        use_video_loss = None
        criterionGAN = None
        criterionFeat = None
        criterionPerceptual = None 
        if self.is_train:
            use_video_loss = False
            criterionGAN = GANLoss(self.gan_mode)
            if self.lambda_gan_feat:
                criterionFeat = torch.nn.L1Loss()
            if self.lambda_perceptual:
                criterionPerceptual = PerceptualLoss(self.perceptual_loss_type)
 
        return use_video_loss, criterionGAN, criterionFeat, criterionPerceptual

    def defineNetworks(self):
        backbone_name = self.model_config["generators"]["backboneName"]
        if backbone_name == "FeaMGenerator": 
            netG = FeaMGenerator(self.model_config["generators"][backbone_name], self.gen_content_nc, self.gen_output_nc)
        else: 
            raise NotImplementedError(f"Generator with model name {backbone_name} could not be recognized")
   
        netD = None
        netDSim = None
        netS = None
        if self.is_train:
            backbone_name = self.model_config["discriminators"]["backboneName"]

            if backbone_name == "MultiscaleDiscriminator":
                netD = MultiscaleDiscriminator(self.model_config["discriminators"][backbone_name], self.lambda_gan_feat, self.dis_input_nc)
            else:
                raise NotImplementedError(f"Generator with model name {backbone_name} could not be recognized")

            if self.use_sim_crop_dis:
                d_config = copy.deepcopy(self.model_config["discriminators"][backbone_name])
                d_config["numD"] = d_config["numD"] - 1 if d_config["numD"] > 1 else 1
                netDSim = MultiscaleDiscriminator(d_config, self.lambda_gan_feat, self.dis_input_nc)
               
            if self.sim_crop_type:
                if self.sim_crop_type != "segSim":
                    os.environ['TORCH_HOME'] = 'models/vgg16'
                    netS = vgg16(["mp_5_3"]) 

        return netG, netD, netDSim, netS 

    def initOptimization(self):
        if self.is_train:
            G_params = list(self.netG.parameters())
            D_params = list(self.netD.parameters())
            if self.use_sim_crop_dis:
                D_params += list(self.netDSim.parameters())

            if not self.tTUR:
                beta1, beta2 = self.beta1, self.beta2
                G_lr, D_lr = self.lr_g, self.lr_d
            else:
                beta1, beta2 = 0, 0.9
                G_lr, D_lr = self.lr_g / 2, self.lr_d * 2

            if self.optimizer_type == "AdamW":
                self.optimizer_G = torch.optim.AdamW(G_params, lr=G_lr, betas=(beta1, beta2), weight_decay=0.0001) 
                self.optimizer_D = torch.optim.AdamW(D_params, lr=D_lr, betas=(beta1, beta2), weight_decay=0.0001)
            else:
                self.optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), weight_decay=0.0001)
                self.optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2), weight_decay=0.0001)

            self.optimizers = [self.optimizer_G, self.optimizer_D] 
            self.initMetrics()

    def initWeights(self):
        self.netG.init_weights(self.init_type, self.init_variance)
        if self.netD:
            self.netD.init_weights(self.init_type, self.init_variance)
        if self.netDSim:
            self.netDSim.init_weights(self.init_type, self.init_variance)
     
    def updateLearningRate(self, step):  
        if self.lr_scheduler == "stepLR":
            if step > self.niter and (step%self.niter_decay) == 0:
                if self.lr_g / 2 < 0.0000125:
                    new_lr_g = self.lr_g
                    new_lr_d = self.lr_d
                else:
                    new_lr_g = self.lr_g / 2
                    new_lr_d = self.lr_d / 2
            else:
                new_lr_g = self.lr_g
                new_lr_d = self.lr_d

            if new_lr_g != self.lr_g:
                if not self.tTUR:
                    new_lr_G = new_lr_g
                    new_lr_D = new_lr_d
                else:
                    new_lr_G = new_lr_g / 2
                    new_lr_D = new_lr_d * 2

                for param_group in self.optimizer_G.param_groups:
                    param_group['lr'] = new_lr_G
                for param_group in self.optimizer_D.param_groups:
                    param_group['lr'] = new_lr_D

                self.lr_g = new_lr_g
                self.lr_d = new_lr_d
        
    def getContent(self, data, index=0, to_train_ids=False):    
        d = data[index][self.input_types[index]]
        if self.input_types[index] == 'segmentations':
            inst = data[index+1][self.input_types[index+1]] if self.instance else None
            d = labelMapToOneHot(d, self.dataset_config["nameOfDataset"], to_train_ids=to_train_ids, inst_map=inst)
        return d

    def getStyle(self, data, index=0):    
        d = data[self.output_index+index][self.output_types[index]]
        return d

    def val(self, data):

        content = [self.getContent(data, index=0)]
        style = [self.getStyle(data, index=0)] 
    
        if self.c_discriminator:  # append mseg segmentations
            content.append(self.getContent(data, index=1)) 
            style.append(self.getStyle(data, index=1)) 

        if self.seg_index: # append segmentations
            content.append(self.getContent(data, index=self.seg_index)) 

        content = [c.view(-1, *c.shape[2:]) for c in content]
        style = [s.view(-1, *s.shape[2:]) for s in style]
                
        if self.c_discriminator:
            content[1] = labelMapToOneHot(content[1], "mseg", to_train_ids=False) 
            style[1] = labelMapToOneHot(style[1], "mseg", to_train_ids=False) 
  
        content_g = torch.cat([content[0], content[self.seg_index]], axis=1) if self.seg_index else content[0] 
        fake_image_batch = self.generateFake(content_g, None)

        return {"fake_B_batch": fake_image_batch, "content": content, "style": style}
    

    def inference(self, data):
        content = [self.getContent(data, index=0)] 
        if self.seg_index: content.append(self.getContent(data, index=self.seg_index)) 
        style = [self.getStyle(data, index=0)] 

        content = [c.view(-1, *c.shape[2:]) for c in content]  
        style = [s.view(-1, *s.shape[2:]) for s in style]

        content_g = torch.cat(content, axis=1) if self.seg_index else content[0] 
        fake_image_batch = self.generateFake(content_g, None)  

        # only needed for metrics calculation later on
        if "segmentations_mseg" in self.input_types: 
            content.append(self.getContent(data, index=1)) 

        return {"fake_B_batch": fake_image_batch, "content": content, "style": style}
    
    def forward(self, data, fake_data=[], sub_nets_name=[]):
        if sub_nets_name == 'generators':
            return self.forwardGenerators(data)
        elif sub_nets_name == 'discriminators':
            return self.forwardDiscriminators(data, fake_data)

    def forwardGenerators(self, data):
        content = [self.getContent(data, index=0)]
        style = [self.getStyle(data, index=0)] 
    
        if self.c_discriminator:  # append mseg segmentations
            content.append(self.getContent(data, index=1)) 
            style.append(self.getStyle(data, index=1)) 

        if self.seg_index: # append segmentations
            content.append(self.getContent(data, index=self.seg_index)) 

        if self.sim_crop_type:
            with torch.no_grad():
                content, style = self.simCrop(content, style)

        content = [c.view(-1, *c.shape[2:]) for c in content]
        style = [s.view(-1, *s.shape[2:]) for s in style]
                
        if self.c_discriminator:
            content[1] = labelMapToOneHot(content[1], "mseg", to_train_ids=False) 
            style[1] = labelMapToOneHot(style[1], "mseg", to_train_ids=False) 
  
        content_g = torch.cat([content[0], content[self.seg_index]], axis=1) if self.seg_index else content[0]
        style_g = style[0]
            
        fake_image_batch = self.generateFake(content_g, style_g)

        out = {"valid":{}} 

        content_condition = content[1] if self.c_discriminator else None
        style_condition = style[1] if self.c_discriminator else None
        content_sim_crop = None
        style_sim_crop = None    
        
        main_valid = 1
        if self.use_discriminator_mask:
            fake_earse, style_earse, earse_mask = discriminatorMask([fake_image_batch, content_condition], [style[0], style_condition])
            main_valid = torch.max(earse_mask).item()

            if self.use_sim_crop_dis:
                content_sim_crop, style_sim_crop = similarityCropSmallPatches([fake_image_batch, content_condition], [style[0], style_condition], earse_mask, crop_factor_h=self.crop_factor_h, crop_factor_w=self.crop_factor_w, max_batch_size=self.sim_crop_dis_max_batch_size)   
                out.update({"fake_sim_crop": content_sim_crop, "style_sim_crop": style_sim_crop})
                out["valid"].update({"sim_crop":style_sim_crop[0].shape[0]})
            
            pred_fake, pred_real = self.discriminate(fake_earse, style_earse, fake_data_sim_d=content_sim_crop, real_data_sim_d=style_sim_crop)
            out.update({"fake_erase": fake_earse, "style_erase": style_earse, "earse_mask": earse_mask})
        else:
            if self.use_sim_crop_dis:
                _, _, earse_mask = discriminatorMask([fake_image_batch, content_condition], [style[0], style_condition])
                content_sim_crop, style_sim_crop = similarityCropSmallPatches([fake_image_batch, content_condition], [style[0], style_condition], earse_mask, crop_factor_h=self.crop_factor_h, crop_factor_w=self.crop_factor_w, max_batch_size=self.sim_crop_dis_max_batch_size)   
                out.update({"fake_sim_crop": content_sim_crop, "style_sim_crop": style_sim_crop})
                out["valid"].update({"sim_crop":style_sim_crop[0].shape[0]})
            pred_fake, pred_real = self.discriminate([fake_image_batch, content_condition], [style[0], style_condition], fake_data_sim_d=content_sim_crop, real_data_sim_d=style_sim_crop)

        out.update({"fake_B_batch": fake_image_batch, "content":content, "style": style,
            "discrimiator_pred_fake_B": pred_fake,
            "discrimiator_pred_real_B": pred_real,
            })
        out["valid"].update({"main":main_valid})
        return out
  
    def generatorLosses(self, data, pred): 
        pred_fake = pred["discrimiator_pred_fake_B"]
        pred_real = pred["discrimiator_pred_real_B"]

        fake_image = pred["fake_B_batch"]
        target = pred["content"][0] 

        losses = {
            "genarators":{
                "AB": {    
                },
            }     
        } 
        for pred_f, pred_r, key_name in zip(pred_fake, pred_real, self.dis_names):
            GAN_loss_image = self.criterionGAN(pred_f[0], True, for_discriminator=False) * self.gan_factor
            losses["genarators"]["AB"][f"GAN_loss_image_{key_name}"] = GAN_loss_image 

            if self.lambda_gan_feat:
                for k, key in enumerate(["image"]):
                    if k < len(pred_f):
                        num_D = len(pred_f[k])
                        GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                        for i in range(num_D):  # for each discriminator
                            # last output is the final prediction, so we exclude it
                            num_intermediate_outputs = len(pred_f[k][i]) - 1
                            for j in range(num_intermediate_outputs):  # for each layer output
                                unweighted_loss = self.criterionFeat(
                                    pred_f[k][i][j], pred_r[k][i][j].detach())
                                GAN_Feat_loss += unweighted_loss * self.lambda_gan_feat / num_D
                        losses["genarators"]["AB"][f"GAN_feat_loss_{key}_{key_name}"] = GAN_Feat_loss[0] * self.gan_factor

        if self.lambda_perceptual:
            perceptual_loss = self.criterionPerceptual(target.contiguous(), fake_image.contiguous()) if "lpips_vgg" in self.perceptual_loss_type else self.criterionPerceptual(fake_image, target) 
            perceptual_loss = perceptual_loss * self.lambda_perceptual
            losses["genarators"]["AB"]["Perceptual_loss"] = perceptual_loss

        genarators_loss = sum(losses["genarators"]["AB"].values()).mean()

        losses["loss"] = genarators_loss
        losses["genarators"]["AB"]["loss"] = genarators_loss

        return losses, min(pred["valid"].values())

    def forwardDiscriminators(self, data, pred_data):   

        if self.use_discriminator_mask:
            style = pred_data["style_erase"]
            content = pred_data["fake_erase"]
            with torch.no_grad():
                fake_image = content[0]   
        else:
            style = pred_data["style"]
            content = pred_data["content"]
            with torch.no_grad():
                fake_image = pred_data["fake_B_batch"] 

        with torch.no_grad():    
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
            style[0] = style[0].detach()
            style[0].requires_grad_()

        style_return = [style[0]]

        fake_sim_crop = None
        style_sim_crop = None
        if self.use_sim_crop_dis:
            fake_sim_crop = pred_data["fake_sim_crop"]
            style_sim_crop = pred_data["style_sim_crop"]
            with torch.no_grad():    
                fake_sim_crop[0] = fake_sim_crop[0].detach()
                fake_sim_crop[0].requires_grad_()
                style_sim_crop[0] = style_sim_crop[0].detach()
                style_sim_crop[0].requires_grad_()
            style_return.append(style_sim_crop[0])

        content_condition = content[1] if self.c_discriminator else None
        style_condition = style[1] if self.c_discriminator else None

        pred_fake, pred_real = self.discriminate([fake_image, content_condition], [style[0], style_condition], fake_data_sim_d=fake_sim_crop, real_data_sim_d=style_sim_crop)
        return [pred_fake, pred_real, style_return, pred_data["valid"]]
    
    def discriminatorLosses(self, pred):
        pred_fake = pred[0]
        pred_real = pred[1]
        real_imgs = pred[2]
        valid = pred[3]

        losses = {
            "discriminators":{
                "B": {    
                },
            }     
        } 
        for pred_f, pred_r, real_img, key_name in zip(pred_fake, pred_real, real_imgs, self.dis_names):
            pred_fake_loss_image = self.criterionGAN(pred_f[0], False, for_discriminator=True) 
            pred_real_loss_image = self.criterionGAN(pred_r[0], True, for_discriminator=True) 
            losses["discriminators"]["B"][f"fake_loss_image_{key_name}"] = pred_fake_loss_image 
            losses["discriminators"]["B"][f"real_loss_image_{key_name}"] = pred_real_loss_image 

            if self.lambdaRealPenalty > 0 and valid[key_name]:
                pred_real_loss_image_penalty, _ = tee_loss(0, real_penalty(pred_real_loss_image, real_img))
                losses["discriminators"]["B"][f"real_loss_image_{key_name}_penalty"] = self.lambdaRealPenalty * pred_real_loss_image_penalty 

        loss_is_valid = min(valid.values())
        discriminators_loss = sum(losses["discriminators"]["B"].values()).mean()
        losses["loss"] = discriminators_loss
        losses["discriminators"]["B"]["loss"] = discriminators_loss
        return losses, loss_is_valid

    def generateFake(self, input_semantics, real_image):
        fake_image = self.netG(input_semantics, real_image)
        return fake_image

    def _discriminator_discriminate(self, netD, fake_image, real_image, fake_condition=None, real_condition=None):
        fake_concat = torch.cat([fake_condition, fake_image], dim=1) if self.c_discriminator else fake_image
        real_concat = torch.cat([real_condition, real_image], dim=1) if self.c_discriminator else real_image
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = netD(fake_and_real)
        pred_fake, pred_real = self.dividePred(discriminator_out)
        return pred_fake, pred_real
  
    def discriminate(self, fake_data_main_d, real_data_main_d, fake_data_sim_d=None, real_data_sim_d=None): 
        pred_fake_main, pred_real_main = self._discriminator_discriminate(self.netD, fake_data_main_d[0], real_data_main_d[0], fake_data_main_d[1], real_data_main_d[1])
        pred_fake = [pred_fake_main]
        pred_real = [pred_real_main]
        if self.use_sim_crop_dis and fake_data_sim_d[0].shape[0]:
            pred_fake_sim, pred_real_sim = self._discriminator_discriminate(self.netDSim, fake_data_sim_d[0], real_data_sim_d[0], fake_data_sim_d[1], real_data_sim_d[1])
            pred_fake.append(pred_fake_sim)
            pred_real.append(pred_real_sim)       

        return pred_fake, pred_real
    
    def dividePred(self, pred):
        # Take the prediction of fake and real images from the combined batch
        # the prediction contains the intermediate outputs of multi-scale GAN,
        # so it's usually a list
        fake_list = []
        real_list = []
        for key in ["image"]:
            if key in pred:
                if type(pred[key]) == list:
                    fake = []
                    real = []
                    for p in pred[key]:
                        fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                        real.append([tensor[tensor.size(0) // 2:] for tensor in p])
                else:
                    fake = pred[key][:pred[key].size(0) // 2]
                    real = pred[key][pred[key].size(0) // 2:]
                fake_list.append(fake)
                real_list.append(real)
        return fake_list, real_list

    def printModel(self):
        modelUtils.printNetwork(self.netG)
        if self.is_train:
            modelUtils.printNetwork(self.netD)
            if self.netDSim:
                modelUtils.printNetwork(self.netDSim)

    def getDiscriminatorOptimizers(self):
        return [self.optimizer_D]

    def getGeneratorOptimizers(self):
        return [self.optimizer_G]

    def getDiscriminators(self):
        return [[d for d in [self.netD, self.netDSim] if d is not None]] 

    def getGenerators(self):
        return [self.netG]

    def getImages(self, data, pred, mode): 
        if data and pred:
            shape_in = self.input_shape if (mode=="train") else self.input_shape_in
            shape_out = self.output_shape if (mode=="train") else self.output_shape_in
            fake_b = pred["fake_B_batch"].view(-1, self.seq_lenght, *shape_out)

            out = { f"real_{self.input_types[0]}_A": pred["content"][0].view(-1, self.seq_lenght, *shape_in), 
                    f"real_{self.output_types[0]}_B": pred["style"][0].view(-1, self.seq_lenght, *shape_out),
                    f"fake_{self.output_types[0]}_B": fake_b}  
        
            if "segmentations" in self.input_types:
                d = pred["content"][1]
                if self.instance: 
                    d = d[:,:-1]
                    i = d[:,-1:]
                    i = i.view(-1, self.seq_lenght, 1, *shape_in[1:])
                    out.update({ f"real_instance_A": i})      
                seg_A = torch.argmax(d, keepdim=True, dim=-3).view(-1, self.seq_lenght, 1, *shape_in[1:])
                seg_A = labelMapToColor(seg_A, self.dataset_config["nameOfDataset"], are_train_ids=False) 
                out.update({ f"real_segmentations_A": seg_A})      

            if "segmentations_mseg" in self.input_types:
                out.update({ f"real_segmentations_mseg_A": pred["content"][-1]})      

            return out

    def getSummary(self, data, model_values, pred, mode):   
        summary = super().getSummary(data, model_values, pred, mode)
        if data:
            shape_in = self.input_shape if (mode=="train") else self.input_shape_in
            if self.verbose:
                for i, input_type in enumerate(self.input_types): 
                    if 'instance' in input_type:     
                        continue
                    if 'segmentations' in input_type:
                        d = pred["content"][i] 
                        if self.instance: 
                            d = d[:,:-1]
                            i = d[:,-1:] * 255
                            i = i.view(-1, self.seq_lenght, 1, shape_in[-2], shape_in[-1])[0].byte().cpu()   
                            summary.update({f"input_content_instance_A":wandb.Video(i, fps=1, format="gif")})

                        d = torch.argmax(d, keepdim=True, dim=-3)    
                        d = d.view(-1, self.seq_lenght, 1, shape_in[-2], shape_in[-1])[0].byte().cpu()      
                    else:
                        d = pred["content"][i].view(-1, self.seq_lenght, *shape_in)
                        d = formatVideo(d, input_type, vid_index=0, dataset_name=self.dataset_config["nameOfDataset"])
                    summary.update({f"input_content_{input_type}_A":wandb.Video(d, fps=1, format="gif")})

                shape_out = self.output_shape if (mode=="train") else self.output_shape_in
                for i, output_type in enumerate(self.output_types):
                    if 'instance' in input_type:     
                        continue
                    if 'segmentations' in output_type:
                        d = pred["style"][i]
                        if self.instance: 
                            d = d[:,:-1]
                            i = d[:,-1:] * 255
                            i = i.view(-1, self.seq_lenght, 1, shape_out[-2], shape_out[-1])[0].byte().cpu()   
                            summary.update({f"input_style_instance_B":wandb.Video(i, fps=1, format="gif")})

                        d = torch.argmax(d, keepdim=True, dim=-3)  
                        d = d.view(-1, self.seq_lenght, 1, shape_out[-2], shape_out[-1])[0].byte().cpu()
                    else:
                        d = pred["style"][i].view(-1, self.seq_lenght, *shape_out)      
                        d = formatVideo(d, output_type, vid_index=0, dataset_name=self.dataset_config["nameOfDataset"])
                    summary.update({f"input_style_{output_type}_B":wandb.Video(d, fps=1, format="gif")})
            else:
                d = pred["content"][0].view(-1, self.seq_lenght, *shape_in)  
                d = formatVideo(d, "frames", vid_index=0, dataset_name=self.dataset_config["nameOfDataset"])
                summary.update({f"input_content_frames_A":wandb.Video(d, fps=1, format="gif")})
                
        if pred:
            shape = self.output_shape if (mode=="train") else self.output_shape_in
            if mode == "train":
                fake_b = pred["fake_B_batch"].view(-1, self.seq_lenght, *shape)
                out_pred = {"output_fake_frames_B":wandb.Video(formatVideo(fake_b, self.output_types[0], vid_index=0, dataset_name=self.dataset_config["nameOfDataset"]), fps=1, format="gif")}
                summary.update(out_pred)
            else:
                for i in range(0, len(pred["fake_B_batch"])):
                    fake_b = pred["fake_B_batch"][i].view(-1, self.seq_lenght, *shape)
                    out_pred = {f"output_fake_frames_B.{i}":wandb.Video(formatVideo(fake_b, self.output_types[0], vid_index=0, dataset_name=self.dataset_config["nameOfDataset"]), fps=1, format="gif")}
                    summary.update(out_pred)

            if self.verbose and (mode == "train"):
                if self.use_discriminator_mask:
                    erase_shape = pred["fake_erase"][0].shape[-3:]
                    fake_b_erase = pred["fake_erase"][0].view(-1, self.seq_lenght, *erase_shape)
                    style_b_erase = pred["style_erase"][0].view(-1, self.seq_lenght, *erase_shape)
                    out_pred = {"output_fake_frames_B_erase":wandb.Video(formatVideo(fake_b_erase, self.output_types[0], vid_index=0, dataset_name=self.dataset_config["nameOfDataset"]), fps=1, format="gif"),
                                "input_style_frames_B_erase":wandb.Video(formatVideo(style_b_erase, self.input_types[0], vid_index=0, dataset_name=self.dataset_config["nameOfDataset"]), fps=1, format="gif")}
                    summary.update(out_pred)

        if mode == "train":
            lrs={
                "opt_generators_lr":self.optimizer_G.param_groups[0]['lr'],
                "opt_discriminator_lr":self.optimizer_D.param_groups[0]['lr'],
                }
            summary.update(lrs)

        return summary
    