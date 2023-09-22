from abc import ABCMeta, abstractmethod

class IExperiment(metaclass=ABCMeta):
    """
    The IExperiment provides the interface for experiment classes.
    """
    
    @abstractmethod
    def execute(self, config):
        """
        Interface Method: Executes the experiment.
        :param config: (Dictionary) The configuration of the experiment.
        """
        raise NotImplementedError('Not implemented')

