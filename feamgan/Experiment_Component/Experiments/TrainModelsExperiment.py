import traceback
import wandb
import torch

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler
from feamgan.Experiment_Component.IExperiment import IExperiment
from feamgan.Experiment_Component.Experiments.experimentFunctions import trainValInferModel
from feamgan.utils import distUtils

class TrainModelsExperiment(IExperiment):
    """
    The experiment trains each model for each given dataset and saves logs and checkpoints.

    :Attributes:
        __config:       (Dictionary) The config of the experiment, containing all models parameters. Refer to the config
                        trainModelsExperiment.json for an example.
        __logger:       (Logger) The logger for the experiment.
        __local_rank:   (Integer) The local_rank (device id).
        __distributed:  (Boolean) Is this run distributed over multiple gpus/devices?
        __seed:         (Integer) The random seed used in the pipeline.
        __num_gpus:     (Integer) The number of GPUs to use.
        __num_threads:  (Integer) The number of CPU threads to use.
        __wandb_config: (Dictionary) The wandb config: inclues e.g. the wandb project to log training stuff.
    """
    def __init__(self, config, controller_config):
        """
        Constructor, initialize member variables.
        :param config: (Dictionary) The config of the experiment, containing all models parameters. Refer to the config
                        trainModelsExperiment.json for an example.
        :param controller_config: (Dictionary) The config of the controller.
        """
        self.__config = config
        self.__logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)
        self.__local_rank = controller_config["localRank"]
        self.__distributed = controller_config["distributed"]
        self.__seed = None
        self.__num_threads = controller_config["hardware"]["numCPUCores"]
        self.__wandb_config = controller_config["modelLogging"]["wandb"]
        self.__use_wandb = controller_config["modelLogging"]["usewandb"]
       
    def execute(self):
        """
        Executes the experiment with the given config.

        The experiment trains each model for each given dataset for the defined training steps.
        """
        for model_config in self.__config["modelConfigs"]:
            self.__seed = model_config["seed"]
            # We first set the random seed to be the same so that we initialize each
            # copy of the network in exactly the same way so that they have the same
            # weights and other parameters. The true seed will be the seed.
            distUtils.setRandomSeed(self.__seed, by_rank=False)
            distUtils.initCudnn(bool(model_config["cudnn"]["deterministic"]), bool(model_config["cudnn"]["benchmark"]))
            
            world_size = 1
            if self.__distributed:
                distUtils.initDist(self.__local_rank)
                if self.__local_rank == 0:
                    print('__CUDA Initialized:', torch.cuda.is_initialized())
                world_size = torch.distributed.get_world_size() # For a single machine world_size == num_gpus
            assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

            model_name = model_config["modelName"]
            if (self.__local_rank == 0) and self.__use_wandb:
                wandb.init(entity=self.__wandb_config["entity"], name=model_config["modelName"], 
                        allow_val_change=True, project=self.__wandb_config["project"], 
                        force=self.__wandb_config["forceLogin"], resume=False, 
                        config=model_config, save_code=True)
                wandb.watch_called = False 

            try:
                for dataset_config in self.__config["datasetConfigs"]:

                    # Only train the model if a batch size for the dataset is given
                    if not dataset_config["nameOfDataset"] in model_config["batchSizes"].keys():
                        continue

                    # and for each xFold iteration
                    for repeat_training_step in model_config["repeatTrainingSteps"]:

                        # If the dataset contains different sizes we want to test, we save them in seperat directories 
                        dataset_dir_name = dataset_config["nameOfDataset"]       
                        if "trainDatasetSize" in dataset_config.keys():
                            dataset_dir_name = dataset_dir_name + "_" + str(dataset_config["trainDatasetSize"])

                        # Construct the model Name
                        model_dir = "/" + model_name + "/" + dataset_dir_name + "/repeatTrainingStep_" + str(repeat_training_step)

                        # Train the model
                        self.__logger.info("Starting to train: " + model_dir, "TrainModelsExperiment:execute")
                        trainValInferModel(model_config, model_dir, dataset_config, self.__local_rank, world_size, self.__num_threads, self.__seed, self.__use_wandb)
                        self.__logger.info("Finished to train: " + model_dir, "TrainModelsExperiment:execute")
                           
            except:
                print(traceback.format_exc())




