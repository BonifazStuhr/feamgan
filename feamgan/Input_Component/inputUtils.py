def createPipelines(dataset_dirs, augmentation_configs, mode, pipe_type, batch_size, sequence_length, device_id, shard_id, num_shards, num_threads, dali_cpu, seed, steps, strides, paired_to_prev):
    """"
    Function which creates pipelines for the given dataset_dirs and mode.
    :param dataset_dirs: (List) A list of dataset dirs to create pipelines for.
    :param augmentation_configs: (LDictionary) Augmentations pipelines config for the dataset.
    :param mode: (String) The subset of the dataset ("train", "eval" or "test).
    :param pipe_type: (DaliPipeline) The Pipeline to construct.
    :param batch_size: (Integer) The batch size to use.
    :param sequence_length: (Integer) The lenght of one sequence (V2V).
    :param device_id: (Integer) The DALI id of the device to use for computation.
    :param shard_id: (Integer) Id defining the shard of the current device. To perform sharding the dataset is divided into multiple parts or shards, and each GPU gets its own shard to process.
    :param num_shards: (Integer) The overall number of shards.
    :param num_threads: (Integer) The number of threads to use.
    :param dali_cpu: (Boolean) True, if Dali should use CPU only.
    :param seed: (Integer) The random seed used in the pipeline.
    :param steps: (Integer) Distance between first frames of consecutive sequences.
    :param strides: (Integer) Distance between consecutive frames in a sequence.
    :param paired_to_prev: (List of Boolean) If true, the dataset will be read in in paired mode with respect to the last folder (to allign two domains this requires paird examples of both domains).
    :return: pipes: List of Pipelines: The created pipelines.
    """
    random_shuffle=False
    if mode is "train":
        random_shuffle=True
    dali_device = "cpu" if dali_cpu else "gpu"

    pipes = []
    i = 0
    for d, aug_conf, step, stride, paired in zip(dataset_dirs, augmentation_configs, steps, strides, paired_to_prev):
        if not paired:
            seed += i
        pipes.append(
            pipe_type(data_dir=d,
                    augmentation_config=aug_conf, 
                    batch_size=batch_size, 
                    sequence_length=sequence_length, 
                    random_shuffle=random_shuffle,
                    pad_last_batch=True,
                    device_id=device_id, 
                    shard_id=shard_id,
                    num_shards=num_shards,
                    num_threads=num_threads, 
                    seed=seed+device_id,
                    reader_seed=seed+device_id,
                    step=step,
                    stride=stride,
                    mode=mode,
                    reader_name=f"{mode}_reader",
                    dali_device=dali_device
                   )
        )
        i+=1
    return pipes