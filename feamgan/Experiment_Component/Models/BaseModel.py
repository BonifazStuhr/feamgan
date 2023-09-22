import torch
import abc

from feamgan.Experiment_Component.Metrics.FidMetric import FidMetric
from feamgan.Experiment_Component.Metrics.KidMetric import KidMetric
from feamgan.Experiment_Component.Metrics.SkvdMetric import SkvdMetric

class BaseModel(torch.nn.Module):

    def __init__(self, model_config, dataset_config, input_shape, output_shape, sequence_length):
        super().__init__()
        self.model_dir = None
        self.model_config = model_config
        self.dataset_config = dataset_config
        
        self.paired = True if "paired" in dataset_config["nameOfDataset"] else False
        self.input_types = model_config["inputDataTypes"]
        self.output_types = model_config["outputDataTypes"]
        self.seq_lenght = sequence_length
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_index = len(self.input_types)
        self.fid_metric_frame = None
        self.kid_metric_frame = None
        self.skvd_metric_frame = None

    @abc.abstractmethod
    def defineLoss(self):
        pass

    def initTraining(self):
        pass

    @abc.abstractmethod
    def initOptimization(self):
        pass

    def initMetrics(self):  
        self.fid_metric_frame = FidMetric(is_video=False, model_dir=self.model_dir, dataset_name=self.dataset_config["nameOfDataset"], dis_model_name="inception_v3")
        self.kid_metric_frame = KidMetric(is_video=False, model_dir=self.model_dir, dataset_name=self.dataset_config["nameOfDataset"], dis_model_name="inception_v3")
        self.skvd_metric_frame = SkvdMetric(is_video=False, model_dir=self.model_dir, dataset_name=self.dataset_config["nameOfDataset"], dis_model_name="vgg16_f_ll")

    @abc.abstractmethod
    def initWeights(self):
        pass

    def setModelDir(self, model_dir):
        self.model_dir = model_dir
        if self.fid_metric_frame:
            self.fid_metric_frame.setModelDir(model_dir)
        if self.kid_metric_frame:
            self.kid_metric_frame.setModelDir(model_dir)
        if self.skvd_metric_frame:
            self.skvd_metric_frame.setModelDir(model_dir)

    @abc.abstractmethod
    def forward(self, data, fake_data=[], sub_nets_name=[]):
        pass

    @abc.abstractmethod
    def generatorLosses(self, data, pred):
        pass

    @abc.abstractmethod
    def discriminatorLosses(self, pred):
        pass

    def discriminatorLossesToTrainList(self, losses):
        return [losses["loss"]]

    def generatorLossesToTrainList(self, losses):
        return [losses["loss"]]

    @abc.abstractmethod
    def updateLearningRate(self, epoch):
        pass

    @abc.abstractmethod
    def printModel(self):
        pass

    def loadOptimizerStateDict(self, state_dicts):
        for optimizer, state_dict in zip(self.optimizers, state_dicts):
            optimizer.load_state_dict(state_dict)

    def getOptimizerStateDict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    @abc.abstractmethod
    def getDiscriminatorOptimizers(self):
        pass

    @abc.abstractmethod
    def getGeneratorOptimizers(self):
        pass
    
    @abc.abstractmethod
    def getDiscriminators(self):
        pass

    @abc.abstractmethod
    def getGenerators(self):
        pass
  
    def updateMetrics(self, data, pred, mode, extended_validation=False): 
        metrics_values = []
        if data and pred:
            fake_b = pred["fake_B_batch"]
            real_b = pred["style"][0]
            self.fid_metric_frame.forwardBatch(real_b, fake_b, mode)  
            if mode is "eval":
                fake_seg_b = pred["content"][1]
                real_seg_b = pred["style"][1]
                self.kid_metric_frame.forwardBatch(real_b, fake_b, mode)  
                self.skvd_metric_frame.forwardBatch([fake_b, fake_seg_b], [real_b, real_seg_b], mode)

        return metrics_values

    @abc.abstractmethod
    def reduceMetrics(self, metrics_values, mode, num_gpus, extended_validation=False): 
        fid_B_frame = self.fid_metric_frame.reduceBatches(mode)
        if mode is "eval":
            kid_B_frame = self.kid_metric_frame.reduceBatches(mode)
            skvd_frame = self.skvd_metric_frame.reduceBatches(mode)
            return {"fid_B_frame": fid_B_frame, "kid_B_frame": kid_B_frame, "skvd_frame": skvd_frame}

        return {"fid_B_frame": fid_B_frame}

    @abc.abstractmethod
    def getInferenceImages(self, data, pred, mode): 
        pass         

    def getSummary(self, data, model_values, pred, mode):   
        return model_values
    
    @abc.abstractmethod
    def getGraphInformation(self):
        pass
