from abc import ABCMeta, abstractmethod

class ITrainer(metaclass=ABCMeta):
    """
    The ITrainer provides the interface for trainer classes, such as a multi gpu trainer.
    """
    
    @abstractmethod
    def startOfEpoch(self, current_epoch):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def startOfStep(self, data, current_step):
        #return data
        raise NotImplementedError('Not implemented')
   
    @abstractmethod
    def discriminator(self, data):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def generator(self, data):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def endOfStep(self, data, current_epoch, next_step):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def endOfEpoch(self, data, next_epoch, next_step):
        raise NotImplementedError('Not implemented')
    
    @abstractmethod
    def getDiscriminatorTrainingSteps(self):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def getGeneratorTrainingSteps(self):
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def getSummary(self):
        raise NotImplementedError('Not implemented')
