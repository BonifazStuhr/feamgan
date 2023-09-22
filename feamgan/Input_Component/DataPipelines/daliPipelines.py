from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn

@pipeline_def
def dali_sequence_pipeline(data_dir, augmentation_config, sequence_length, random_shuffle, pad_last_batch, shard_id, num_shards, reader_seed, step, stride, mode, reader_name, dali_device="cpu"):
    images = fn.readers.sequence(file_root=data_dir, 
                                sequence_length=sequence_length,
                                shard_id=shard_id,
                                num_shards=num_shards,
                                initial_fill=1024,
                                prefetch_queue_depth=100,
                                random_shuffle=random_shuffle,
                                pad_last_batch=pad_last_batch,
                                read_ahead=True,
                                step=step,
                                stride=stride,
                                seed=reader_seed,
                                name=reader_name,
                                image_type=augmentation_config["daliImageType"]) 
    if dali_device == "gpu":                               
        images = images.gpu()
        
    if augmentation_config["resize"]:
        if 'resize' in augmentation_config["preprocessMode"]: 
            images = fn.resize(images, device=dali_device, size=augmentation_config["dataLoadSize"], interp_type=augmentation_config["method"])
        elif 'scale_width' in augmentation_config["preprocessMode"]:
            images = fn.resize(images, device=dali_device, resize_x=augmentation_config["dataLoadSize"], interp_type=augmentation_config["method"])
        elif 'scale_height' in augmentation_config["preprocessMode"]:
            images = fn.resize(images, device=dali_device, resize_y=augmentation_config["dataLoadSize"], interp_type=augmentation_config["method"])
        elif 'scale_not_smaller' in augmentation_config["preprocessMode"]:
            images = fn.resize(images, device=dali_device, size=augmentation_config["dataLoadSize"], interp_type=augmentation_config["method"], mode="not_smaller")
    
    if 'crop' in augmentation_config["preprocessMode"]:    
        min_crop_top_pos = 0.1 # Cityscapes has sometimes some recording artifacts at the top of the image, so we do not crop them
        max_crop_buttom_pos = 0.2 # Viper/PDF and Cityscapes have differtent car bodies, we reduce the apperance of them # 0.6
        if mode=="train":
            crop_pos_x = fn.random.uniform(device=dali_device, range=(0.0, 1.0))
            if 'top_crop' in augmentation_config["preprocessMode"]:
                crop_pos_y = fn.random.uniform(device=dali_device, range=(min_crop_top_pos, max_crop_buttom_pos))
            else:
                crop_pos_y = fn.random.uniform(device=dali_device, range=(0.0, 1.0))
        else:
            crop_pos_x = 0.5
            if 'top_crop' in augmentation_config["preprocessMode"]:
                crop_pos_y = min_crop_top_pos 
            elif 'random_crop' in augmentation_config["preprocessMode"]:
                crop_pos_x = fn.random.uniform(device=dali_device, range=(0.0, 1.0))
                crop_pos_y = fn.random.uniform(device=dali_device, range=(0.0, 1.0))
            else: 
                crop_pos_y = 0.5

        images = fn.crop(images,
                    crop=augmentation_config["cropSize"],
                    crop_pos_x=crop_pos_x,
                    crop_pos_y=crop_pos_y,
                    device=dali_device)

    if augmentation_config["flip"] and mode=="train":   
        images = fn.flip(images, horizontal=fn.random.coin_flip(device=dali_device, probability=0.5), device=dali_device)
 
    if augmentation_config["normalize"]:
        images = fn.normalize(images, mean=127.5, stddev=127.5, device=dali_device) 

    images = fn.transpose(images, perm=[0,3,1,2], device=dali_device)

    return images