import os.path
import importlib
import copy
import nvidia.dali.types as types
from nvidia.dali.types import DALIImageType

from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from feamgan.Input_Component.DataPipelines.daliPipelines import dali_sequence_pipeline
from feamgan.Input_Component.inputUtils import createPipelines
from feamgan.Experiment_Component.AModelSuit import AModelSuit

def createModel(model_config, dataset_config):
    """
    Function which creates the model for the given model_config and dataset_config.
    :param model_config: (Dictionary) The configuration of the model containing all hyperparameters.
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :return: model: (torch.nn.Module) The model.
    """
    input_module = importlib.import_module("feamgan.Experiment_Component.Models." + model_config["modelClassName"])
    model = getattr(input_module, model_config["modelClassName"])(model_config, dataset_config,
            input_shape=dataset_config["augmentations"]["inputDataShapeA"], output_shape=dataset_config["augmentations"]["inputDataShapeB"], sequence_length=model_config["sequenceLengths"][dataset_config["nameOfDataset"]])  
    return model

def createTrainer(model_config):
    """
    Function which creates the trainer, which is used in the training loop.
    :param model_config: (Dictionary) The configuration of the model containing all hyperparameters.
    :return: trainer: (ITrainer) The trainer.
    """
    input_module = importlib.import_module("feamgan.Experiment_Component.Trainer." + model_config["trainerName"])
    trainer = getattr(input_module, model_config["trainerName"])() 
    return trainer

def loadInputPipeline(dataset_config, batch_size, sequence_length, device_id, shard_id, num_shards, num_threads, dali_cpu, seed, val_dataset_name):
    """
    Function which loads the input pipeline of the dataset specified in the given dataset_config.

    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :param batch_size: (Integer) The batch size to use.
    :param sequence_length: (Integer) The lenght of one sequence (V2V).
    :param device_id: (Integer) The DALI id of the device to use for computation.
    :param shard_id: (Integer) Id defining the shard of the current device. To perform sharding the dataset is divided into multiple parts or shards, and each GPU gets its own shard to process.
    :param num_shards: (Integer) The overall number of shards.
    :param num_threads: (Integer) The number of threads to use.
    :param dali_cpu: (Boolean) True if Dali should use CPU only.
    :param seed: (Integer) The random seed used in the pipeline.
    :param val_dataset_name: (String) The name of the validation dataset (used to switch between val and a val subset, e.g. val2000).
    :return: dataset: (Dictionary) Dictionary containing the input pipelines in form {"train":[pipe0,...], "eval":[pipe0,...], "test":[pipe0,...]}
    """
    # Set parameters 
    data_root = os.path.dirname(os.path.abspath("feamgan"))   
    pipe_type = dali_sequence_pipeline
    domains = [dataset_config["datasetATypes"], dataset_config["datasetBTypes"]]
    domain_relative_paths = dataset_config["datasetDirs"]
    steps_in = dataset_config["steps"]
    strides_in = dataset_config["strides"]
    unpaired_domains = False if "paired" in dataset_config["nameOfDataset"] else True

    # Specify input
    train_dataset_dirs = []
    eval_dataset_dirs = []
    data_types = []
    paired_to_prev = []  
    steps = []
    strides = []
    s = 0
    for domain, domain_relative_path in zip(domains, domain_relative_paths):
        first = True
        for data_type in domain:
            train_dataset_dirs.append(os.path.join(data_root, domain_relative_path, 'train', data_type))
            eval_dataset_dirs.append(os.path.join(data_root, domain_relative_path, val_dataset_name, data_type))
            data_types.append(data_type)
            steps.append(steps_in[s])
            strides.append(strides_in[s])
            if first and unpaired_domains:
                paired_to_prev.append(False)
                first = False
            else:
                paired_to_prev.append(True) 
        s += 1

    # Create Augmentations
    augmentation_configs = []
    for data_type in data_types:
        aug_conf = copy.deepcopy(dataset_config["augmentations"])
        aug_conf["resize"] = True 
        if data_type == "frames":
            aug_conf["method"] = types.INTERP_CUBIC
            aug_conf["normalize"] = True
            aug_conf["daliImageType"] = DALIImageType.RGB  
        elif (data_type == "segmentations") or (data_type == "segmentations_mseg"):
            aug_conf["method"] = types.INTERP_NN
            aug_conf["normalize"] = False
            aug_conf["daliImageType"] = DALIImageType.GRAY 
        augmentation_configs.append(aug_conf)

    # Create input pipelines
    dataset = {}
    data_names = None

    train_pipes = createPipelines(train_dataset_dirs, augmentation_configs, "train", pipe_type, batch_size, sequence_length, device_id, shard_id, num_shards, num_threads, dali_cpu, seed, steps, strides, paired_to_prev)
    for pipe in train_pipes: pipe.build()
    dataset["train"] = [DALIGenericIterator(train_pipe, [data_type], reader_name="train_reader", last_batch_policy=LastBatchPolicy.DROP) for train_pipe, data_type in zip(train_pipes, data_types)]

    eval_pipes = createPipelines(eval_dataset_dirs, augmentation_configs, "eval", pipe_type, batch_size, sequence_length, device_id, shard_id, num_shards, num_threads, dali_cpu, seed, steps, strides, paired_to_prev)
    for pipe in eval_pipes: pipe.build()
    dataset["eval"] = [DALIGenericIterator(eval_pipe, [data_type], reader_name="eval_reader", last_batch_policy=LastBatchPolicy.DROP) for eval_pipe, data_type in zip(eval_pipes, data_types)]

    return dataset, data_names


def trainValInferModel(model_config, model_dir, dataset_config, local_rank, num_gpus, num_threads, seed, use_wandb):
    """
    Trains and additionally evaluates the model defined in the model_config with the given dataset on num_gpus gpus
    and saves the model to the path model_dir.
    :param model_config: (Dictionary) The configuration of the model containing all hyperparameters.
    :param model_dir: (String) The path to the directory in which the model is saved.
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :param local_rank: (Integer) The local_rank (device id).
    :param num_gpus: (Integer) The number of gpus to train and val with.
    :param num_threads: (Integer) The number of threads to use.
    :param seed: (Integer) The random seed used in the pipeline.
    """
    batch_size = model_config["batchSizes"][dataset_config["nameOfDataset"]]
    sequence_length = model_config["sequenceLengths"][dataset_config["nameOfDataset"]]
    
    dataset, data_names = loadInputPipeline(dataset_config, batch_size, sequence_length, local_rank, local_rank, num_gpus, num_threads, 
                                model_config["pipeline"]["daliUseCPU"], seed, dataset_config["valDatasetName"])

    model_config.update({"inputDataTypes":dataset_config["datasetATypes"], "outputDataTypes":dataset_config["datasetBTypes"]})

    model = createModel(model_config, dataset_config)
    trainer = createTrainer(model_config)

    model_suit = AModelSuit(model=model, 
                            trainer=trainer, 
                            dataset=dataset, 
                            batch_size=batch_size, 
                            num_gpus=num_gpus,
                            model_dir=model_dir,
                            model_config=model_config,
                            local_rank=local_rank,
                            seed=seed,
                            save_checkpoint_steps=model_config["saveCheckpointSteps"],
                            save_checkpoint_epochs=model_config["saveCheckpointEpochs"],
                            log_steps=model_config["logSteps"],
                            log_epochs=model_config["logEpochs"],
                            save_summary_steps=model_config["saveSummarySteps"],
                            save_summary_epochs=model_config["saveSummaryEpochs"],
                            verbose=model_config["verbose"],
                            use_wandb=use_wandb,
                            dict_keys=data_names)

    if model_config["doTraining"]:
        model_suit.doTraining(model_config["trainingSteps"], model_config["evalSteps"], 
                            model_config["trainingEpochs"], model_config["evalEpochs"])

    if model_config["doInference"]:
        model_suit.doInference("eval")
    
    if model_config["createVideoFromInference"]:
        model_suit.createVideoFromInference("eval", dataset_config, num_frames=9000)

