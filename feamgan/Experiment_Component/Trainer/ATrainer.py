
import torch

from apex import amp

from feamgan.utils import distUtils
from feamgan.Experiment_Component.ITrainer import ITrainer

class ATrainer(ITrainer):

    def __init__(self):
        super(ATrainer, self).__init__()
        self.loss_index = 0
        self.is_inference = False
        self.mode = "train"

    def startOfEpoch(self, current_epoch, model):
       pass

    def startOfStep(self, data, current_step, model):

        torch.autograd.set_detect_anomaly(True)

        self.loss_index = 0
        
        for i in range(len(data)):
            data[i] = distUtils.dictionaryToCuda(data[i])
        return data

    def discriminator(self, data, fake_data, model):
    
        if self.mode == "train":
            discriminators_opts = model.module.getDiscriminatorOptimizers()
            discriminators = model.module.getDiscriminators()
            generators = model.module.getGenerators()
            for dis in discriminators: 
                for d in dis :self.toggleGrad(d, True)
            for gen in generators: self.toggleGrad(gen, False)

            for opt in discriminators_opts: opt.zero_grad()
          
        pred = model(data, fake_data, "discriminators")
        losses, loss_is_valid = model.module.discriminatorLosses(pred)
      
        if self.mode == "train":
            loss = model.module.discriminatorLossesToTrainList(losses)
      
            if None in loss:
                return

            for l, opt, dis in zip(loss, discriminators_opts, discriminators):#, discriminators): 
                with amp.scale_loss(l, opt, loss_id=self.loss_index) as scaled_loss:
                    scaled_loss.backward()  

                opt.step() if loss_is_valid else opt.zero_grad()
                self.loss_index+=1
                
        return losses, pred

    def generator(self, data, model):

        if self.mode == "train":
            generator_opts = model.module.getGeneratorOptimizers()
            discriminators = model.module.getDiscriminators()
            generators = model.module.getGenerators()
            for dis in discriminators:
                for d in dis :self.toggleGrad(d, False)
            for gen in generators: self.toggleGrad(gen, True)

            for opt in generator_opts: opt.zero_grad()

        pred = model(data, None, "generators")
        losses, loss_is_valid = model.module.generatorLosses(data, pred)

        if self.mode == "train":
            loss = model.module.generatorLossesToTrainList(losses)
 
            if None in loss:
                return

            for opt, l, gen in zip(generator_opts, loss, generators):#, generators):
                with amp.scale_loss(l, opt, loss_id=self.loss_index) as scaled_loss:
                    scaled_loss.backward()

                opt.step() if loss_is_valid else opt.zero_grad()
                self.loss_index+=1

        return losses, pred

    def endOfStep(self, data, current_epoch, next_step, model):
        if self.mode == "train":
            model.module.updateLearningRate(next_step) 
        return data

    def endOfEpoch(self, data, next_epoch, next_step, model):
        pass

    def getDiscriminatorTrainingSteps(self):
        return 1

    def getGeneratorTrainingSteps(self):
        return 1

    def getSummary(self):
        summary = {}
        return summary

    def train(self):
        self.mode = "train"
    
    def val(self):
        self.mode = "val"

    def toggleGrad(self, model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)