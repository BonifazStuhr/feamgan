# Original Implementation of the Paper "Masked Discriminators for Content-Consistent Unpaired Image-to-Image Translation"
This is the original implementation of the paper 
[“Masked Discriminators for Content-Consistent Unpaired Image-to-Image Translation”](https://arxiv.org/abs/2309.13188).

This repository contains all the code used to create the paper, as well as links to results and weights:

&emsp; :rocket: Links to inferred images of the validation set for easy comparison with our models<br/>
&emsp; :crystal_ball: Links to the weights of our pretrained models<br/>
&emsp; :microscope: Code and tutorial for configuring, training, and validating own models<br/>
&emsp; :scroll: Code and tutorial for inference with our pretrained models<br/>

![alt text](resources/FeaMGAN_results.png "Results of different translation tasks")

## FeaMGAN Overview
![alt text](resources/FeaMGAN_method_overview.svg "FeaMGAN overview")

## Images for easy comparison
We provide inferred images of the validation set for our models for easy comparison with our models:

| Model | Translation Task | FID | KID | sKVD | cKVD |
| :---: | :---: | :---: | :---: |  :---: |  :---: | 
| [Small](https://drive.google.com/file/d/1eH3k18SfPj7EIkKMXQBuoUoWS8qNMLx2/view?usp=sharing) | PFD → Cityscapes| 43.27 | 32.50 | 12.98 | 40.23 | 
| [Small](https://drive.google.com/file/d/1DNEUmFZbqpKVAOnu5D7lrP6C-UsUd-Es/view?usp=sharing) | VIPER → Cityscapes| 50.00 | 32.74 | 13.62 | 44.60 | 
| [Small](https://drive.google.com/file/d/1JOyH5xaz1EtODSMBJpfvM80wbEY5BIqK/view?usp=sharing) | BDDday → BDDnight| 66.29 | 46.05 | 13.21 | 45.26 | 
| [Small](https://drive.google.com/file/d/1ZrF6RWOIlOmANU4vhUEej0QLwJ0TVulf/view?usp=sharing) | BDDclear → BDDsnowy| 56.25 | 15.83 | 12.09 | 38.91 | 
| [Big](https://drive.google.com/file/d/17Dar1NldAemZlT_2v4Tdb9QpDYSvKo9N/view?usp=sharing) | PFD → Cityscapes| 40.32 | 28.59 | 12.94 | 40.02 | 
| [Big](https://drive.google.com/file/d/1E1qTliRF91S4VOcvqTjt1_CvZj2fQi4J/view?usp=sharing) | VIPER → Cityscapes| 48.00 | 29.16 | 13.58 | 47.52 | 
| [Big](https://drive.google.com/file/d/1Cqfw-GUAPkSeldFIsg8IyUP0WkHO_-GF/view?usp=sharing) | BDDday → BDDnight| 64.40 | 43.74 | 11.89 | 45.34 | 
| [Big](https://drive.google.com/file/d/1brr1FdD3LktOTUVZks4SMZwRX6LjpHAB/view?usp=sharing) | BDDclear → BDDsnowy| 56.47 | 13.18 | 10.93 | 39.97 | 


## Requirements
NVIDIA Graphic Card Drivers, Docker, and the NVIDIA Container Toolkit are required. 

- Docker Installation Guide for Ubuntu can be found [here](https://docs.docker.com/engine/install/ubuntu/).
- NVIDIA Container Toolkit Installation Guides can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Since we use the NVIDIA Container Toolkit and NVIDIA Dali, we do not support Windows operating systems.

## Building the docker images
In the following, we provide an installation guide of our framework based on docker.
None-Docker users need to install the requirements (pytorch, pip, and others) specified in the docker files by hand and run the scripts without docker. 

### Main docker image 
This docker image is used for inference, training and to extract and reformat the PFD, VIPER, Cityscapes, and BDD datasets.

1. From the project root ```feamgan``` navigate to the directory containing the docker images: 
   ```console
   cd feamgan/Docker
   ```
2. Build the docker image ```Dockerfile```: 
   ```console
   docker image build -f Dockerfile -t feamgan_docker .
   ```

### MSeg docker image 
For model training, we create segmentations with [mseg](https://github.com/mseg-dataset/mseg-semantic).
These segmentations are used by the discriminators.

1. From the project root ```feamgan``` navigate to the directory containing the docker images: 
   ```console
   cd feamgan/Docker
   ```
2. Build the docker image ``Dockerfile_MSeg``: 
   ```console
   docker image build -f Dockerfile_MSeg -t feamgan_mseg_docker .
   ```
3. Download the ```mseg-3m.pth``` model [here](https://drive.google.com/file/d/1BeZt6QXLwVQJhOVd_NTnVTmtAO1zJYZ-/view). 
4. Move the file ```mseg-3m.pth```  to the directory ```feamgan/models```. If the directory does not exist, create the directory.


## Prepare datasets
We provide scripts to automatically extract and reformat the datasets we used in our experiments.  
These scripts format each dataset into the following structure:
```
feamgan
   data
      DATASETNAME
         zips
            data1.zip
            data2.zip
            ...
         sequencens
            train
               frames
                  sequence1
                     sequence1_frame1_info.png
                     sequence1_frame2_info.png
                     ...
                  sequence2
                     sequence2_frame1_info.png
                     sequence2_frame2_info.png
                     ...
                  ...
               segmentations
                  ...
               ...
            val
               ...
```

### Cityscapes
1. Log in to the [Cityscapes Dataset Homepage](https://www.cityscapes-dataset.com/login/) and make sure that your account has access to the following data: ```leftImg8bit_sequence_trainvaltest```.
   If you do not have access to this data, follow the instructions on the Cityscapes website and request access.
2. Run the following command from the project root ```feamgan``` and follow the instructions:  
   ```console
   docker run -it --rm --cpus 255 -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -m feamgan.datasetPreperation.downloadAndExtractCityscapes
   ```

### Playing for Data (PFD)
1. Navigate to the [PFD download website](https://download.visinf.tu-darmstadt.de/data/from_games/), scroll down, and download the entire dataset (Images and Labels):
2. Move all downloaded files to the directory ```feamgan/data/PFD/zips```. If the directory does not exist, create the directory.
3. Navigate to the [EPE baseline download website](https://drive.google.com/u/0/uc?id=1FXKa7PrtQgkv_C_Egz2YLXHwyNc4CHnK&export=download) and download the baseline images from the EPE model (we need the images infreend by EPE to construct a dataset for a fair comparison).
4. Move the downloaded baseline images .zip file (ours_pfd2cs_jpg.zip) to the directory ```feamgan/data/Baselines/from_epe/epe/```, if the directory does not exist, create the directory.
5. Run ```mkdir -p frames/0``` and ```unzip ours_pfd2cs_jpg.zip -d frames/0```
6. Run the following command from the project root ```feamgan```: 
   ```console
   docker run -it --rm --cpus 255 -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -m feamgan.datasetPreperation.extractPFD
   ```
7. Run the following command from the project root ```feamgan``` to format the segmentations of the dataset to grayscale: 
   ```console
   docker run -it --rm --cpus 255 -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -m feamgan.datasetPreperation.convertSegmentations --save_path "/data/PFD" --subset "train"

   docker run -it --rm --cpus 255 -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -m feamgan.datasetPreperation.convertSegmentations --save_path "/data/PFD" --subset "val"
   ```

### VIPER
1. Navigate to the [VIPER Download Website](https://playing-for-benchmarks.org/download/) and download the following files for the training and validation set:
   ```
      Modality: Image
         - Every 10th frame, lossless compression.
         - The next frame, needed for e.g., optical flow, lossless compression.
         - All other frames for dense video, lossless compression
      Modality: Semantic class labels
         - Every 10th frame.
         - The next frame, needed for e.g., optical flow.
         - All other frames for dense video.
   ```
2. Move all downloaded files to the directory ```feamgan/data/VIPER/zips```. If the directory does not exist, create the directory.
3. Run the following command from the project root ```feamgan```: 
   ```console
   docker run -it --rm --cpus 255 -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -m feamgan.datasetPreperation.extractVIPER
   ```
4. Run the following commands from the project root ```feamgan``` to format the segmentations of the dataset to grayscale: 
   ```console
   docker run -it --rm --cpus 255 -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -m feamgan.datasetPreperation.convertSegmentations --save_path "/data/VIPER" --subset "train"

   docker run -it --rm --cpus 255 -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -m feamgan.datasetPreperation.convertSegmentations --save_path "/data/VIPER" --subset "val"
   ```

### BDD100k
1. Navigate to the [BDD100k Website](https://bdd-data.berkeley.edu/), press download dataset, create an account and download ```Labels (press Labels button under BDD100K)```
2. Navigate to the [dl.yf.io BDD100k Page](http://dl.yf.io/bdd100k/video_parts/) and download the files ```bdd100k_videos_train_00.zip``` and ```bdd100k_videos_val_00.zip```.
3. Move all downloaded files to the directory ```feamgan/data/BDD100k/zips```. If the directory does not exist, create the directory.
4. Run the following command from the project root ```feamgan```: 
   ```console
   docker run -it --rm --cpus 255 -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -m feamgan.datasetPreperation.extractBDD
   ```
5. This script is inefficient and may take some days.
6. Run the following command to construct the subsets used in the paper:
   ```console
   docker run -it --rm --cpus 255 -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python python -m feamgan.datasetPreperation.createBDDSubsets
   ```

## Create mseg segmentations for a given dataset
To train our model on a translation task, we need to create segmentations for the entire translation task (both domains). We provide a script that automates this process.
The script is inefficient but can be used in parallel on multiple GPUs to speed up the process a little.

Specify the DATASETNAME (e.g., PFD) and run the createMSegSegmentations from the project root ```feamgan```.

- One 1 GPU: 
   ```console
   docker run -it --rm --gpus device=0 --cpus 255 --shm-size 64G --ipc=host -v $PWD:/root/feamgan -w /root/feamgan feamgan_mseg_docker python -u -m feamgan.datasetPreperation.createMSegSegmentations --rank 0 --num_gpus 1 --dataset_path "/data/DATASETNAME" 
   ```
   
- Alternatively, you can run the script in parallel on multiple GPUs, e.g., 2:
   ```console
   docker run -it --rm --gpus device=0 --cpus 255 --shm-size 64G --ipc=host -v $PWD:/root/feamgan -w /root/feamgan feamgan_mseg_docker python -u -m feamgan.datasetPreperation.createMSegSegmentations --rank 0 --num_gpus 2 --dataset_path "/data/DATASETNAME"
   ```
   ```console
   docker run -it --rm --gpus device=1 --cpus 255 --shm-size 64G --ipc=host -v $PWD:/root/feamgan -w /root/feamgan feamgan_mseg_docker python -u -m feamgan.datasetPreperation.createMSegSegmentations --rank 1 --num_gpus 2 --dataset_path "/data/DATASETNAME"
   ```
   
This script is inefficient and may take some days.

## Create a validation subset for training
For training, we recommend randomly sampling a small subset of the validation data (e.g., 2000 samples) to partially evaluate the model during training without wasting too much computational time:
   ```console
   docker run -it --rm --cpus 255 -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -m feamgan.datasetPreperation.createDatasetSubsetSubset --dataset_path "/data/DATASETNAME" --dataset_subset "val" --subset_size 2000
   ```

## Inference with pretrained models 

### 1. Setup Configuration: ```controllerConfig.json```
Here you can configure if the logs should be synchronized with your [Weights&Biases](https://wandb.ai/site) project to log the inference online:
```json
{ 
   "executeExperiments": 1,
   "usewandb": 1,
   "modelLogging":{
      "wandb":{
         "apiKey": "Your Weights&Biases API Key",
         "project": "Your Weights&Biases Project",
         "entity": "Your Weights&Biases Entity (Profile)",
         "forceLogin": 1 
      }
   }
}
```

### 2. Download pretrained models 
Download a pretrained model of your choice, and make sure the corresponding datasets were extracted and reformated correctly.
   | Model | Translation Task | FID | KID | sKVD | cKVD |
   | :---: | :---: | :---: | :---: |  :---: |  :---: | 
   | [Small](https://drive.google.com/file/d/1XCevGVnrTPQExVRizayPxcxoIDEWVQlp/view?usp=sharing) | PFD → Cityscapes| 43.27 | 32.50 | 12.98 | 40.23 | 
   | [Small](https://drive.google.com/file/d/1SXazkntDuye3pgp1id7CjSc9R90ulHZj/view?usp=sharing) | VIPER → Cityscapes| 50.00 | 32.74 | 13.62 | 44.60 | 
   | [Small](https://drive.google.com/file/d/1hiTMZbuajb_Kawscqg2KMRhqhqOYLeY0/view?usp=sharing) | BDDday → BDDnight| 66.29 | 46.05 | 13.21 | 45.26 | 
   | [Small](https://drive.google.com/file/d/1TB8hXumlVVw4UamR6mg2ujo0rk56t4GM/view?usp=sharing) | BDDclear → BDDsnowy| 56.25 | 15.83 | 12.09 | 38.91 | 
   | [Big](https://drive.google.com/file/d/1mw19mbZFU3iii86rr4pxs0x-zGoOiTrW/view?usp=sharing) | PFD → Cityscapes| 40.32 | 28.59 | 12.94 | 40.02 | 
   | [Big](https://drive.google.com/file/d/1lJyrhimZH1F62MD8gtlVNOp1wE5wbgvY/view?usp=sharing) | VIPER → Cityscapes| 48.00 | 29.16 | 13.58 | 47.52 | 
   | [Big](https://drive.google.com/file/d/1fmJu3mW49Fe3wunH722hDVLnP3t_CwKT/view?usp=sharing) | BDDday → BDDnight| 64.40 | 43.74 | 11.89 | 45.34 | 
   | [Big](https://drive.google.com/file/d/1Nzp2vXBdtvu_BRgujj4PS47NmQsTyHLl/view?usp=sharing) | BDDclear → BDDsnowy| 56.47 | 13.18 | 10.93 | 39.97 | 

Move the ```MODEL_NAME.pt```  to the directory ```feamgan/pretrainedModels/MODEL_NAME/checkpoints```.

### 3. Register the InferenceExperiment in the schedule
Open ```feamgan/feamgan/experimentSchedule``` and configure the schedule as follows:
```json
{
   "mode": "sequential",
   "experimentsToRun": ["InferenceExperiment"],
   "experimentConfigsToRun": {"InferenceExperiment": ["pretrainedModels/val/MODEL_NAME.json"]}
}
```

### 4. Execute the experiment
Execute the following command from the project root ```feamgan```: 
```console
docker run -it --rm --gpus device=0 --cpus 255 --shm-size=32G --ipc=host -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -u -m feamgan.main
```

## Train and evaluate your own model
There are two .json configuration files: ```controllerConfig.json``` and ```experimetSchedule.json``` to configure a run. 
The default settings execute the training of our best PFD → Cityscapes model.

### 1. Setup configuration: ```controllerConfig.json```
Here you can configure if the logs should be synchronized with your [Weights&Biases](https://wandb.ai/site) project to log your training online:
```json
{ 
   "executeExperiments": 1,
   "modelLogging":{
      "usewandb": 1,
      "wandb":{
         "apiKey": "Your Weights&Biases API Key",
          "project": "Your Weights&Biases Project",
         "entity": "Your Weights&Biases Entity (Profile)",
         "forceLogin": 1 
      }
   }
}
```
### 2. Choose the experiment
Experiment configurations for training can be found in ```feamgan/feamgan/Experiment_Component/ExperimentConfigs/train```
Choose the .json training configuration file to execute and register it in the experimentSchedule.json file:
```json
{
   "mode": "sequential",
   "experimentsToRun": ["TrainModelsExperiment"],
   "experimentConfigsToRun": {"TrainModelsExperiment": ["train/path/to/trainExperiment.json"]}
}
```

### 3. Execute the experiment
Execute the following command from the project root ```feamgan```. This trains the model specified in the configuration on the first GPU (device 0): 
```console
docker run -it --rm --gpus device=0 --cpus 255 --shm-size=32G --ipc=host -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -u -m feamgan.main
```

### 4. Inference and evaluation with the model
#### 4.1  Change the TrainModelsExperiment to the corresponding evaluation configuration
Experiment configurations for evaluation can be found in ```feamgan/feamgan/Experiment_Component/ExperimentConfigs/train```
Choose the .json corresponding evaluation configuration file to execute and register it in the experimentSchedule.json file:
```json
{
   "mode": "sequential",
   "experimentsToRun": ["TrainModelsExperiment"],
   "experimentConfigsToRun": {"TrainModelsExperiment": ["val/path/to/trainExperiment.json"]}
}
```

#### 4.2 Inference with the model
Execute the following command from the project root ```feamgan```. This uses the trained model to create images for the given dataset specified in the configuration on the first GPU (device 0): 
```console
docker run -it --rm --gpus device=0 --cpus 255 --shm-size=32G --ipc=host -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -u -m feamgan.main
```

#### 4.3 Evaluation of the model: FID and KID
Execute the following command from the project root ```feamgan```. This command uses the inferred images to calculate metrics such as FID  and KID. ```MODEL_NAME, DATASET_A_NAME, DATASET_B_NAME``` are specified in the experiment's configuration.
```console
docker run -it --rm --gpus device=0 --cpus 255 --shm-size=32G --ipc=host -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -u -m feamgan.eval.quickEval --model_name MODEL_NAME --dataset_A_name "DATASET_A_NAME" --dataset_B_name "DATASET_B_NAME"
```
| Translation Task | Config | 
| :---: | :---: | 
| PFD → Cityscapes | --dataset_A_name "pfd" --dataset_B_name "Cityscapes" | 
| VIPER → Cityscapes| --dataset_A_name "viper" --dataset_B_name "Cityscapes" | 
| BDDday → BDDnight | --dataset_A_name "BDDdaytimeSubset" --dataset_B_name "BDDnightSubset" | 
| BDDclear → BDDsnowy | --dataset_A_name "BDDclearSubset" --dataset_B_name "BDDsnowySubset" | 


#### 4.4 Evaluation of the model: sKVD and cKVD
Execute the following command from the project root ```feamgan```. This command uses the inferred images to calculate metrics such as FID and KID. ```MODEL_NAME, DATASET_A_NAME, DATASET_B_NAME, CROP_SIZE_H, CROP_SIZE_W``` are speciefied in the experiment's configuration. 
```console
docker run -it --rm --gpus device=0 --cpus 255 --shm-size=32G --ipc=host -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -u -m feamgan.eval.kvdEval --model_name MODEL_NAME --dataset_A_name "DATASET_A_NAME" --dataset_B_name "DATASET_B_NAME" --data_load_height HEIGHT--crop_size_w CROP_SIZE_W --crop_size_h CROP_SIZE_H"
```
The sKVD and cKVD implementations are memory-consuming. You may need to specify other parameters defined in ```kvdEval.py``` like ```--every_x_steps``.

Configurations for the translation tasks:

| Translation Task | sKVD Config | 
| :---: | :---: | 
| PFD → Cityscapes | --dataset_A_name "viper" --dataset_B_name "Cityscapes" --data_load_height 526 --crop_size_h 526 --crop_size_w 957 --metric "sKVD" | 
| VIPER → Cityscapes | --dataset_A_name "pfd" --dataset_B_name "Cityscapes" --data_load_height 526 -crop_size_h 526 --crop_size_w 935 --metric "sKVD" --every_x_steps 2 | 
| BDDday → BDDnight |--dataset_A_name "BDDdaytimeSubset" --dataset_B_name "BDDnightSubset" --data_load_height 526 -crop_size_h 526 --crop_size_w 935 --metric "sKVD" --every_x_steps 2 | 
| BDDclear → BDDsnowy | --dataset_A_name "BDDclearSubset" --dataset_B_name "BDDsnowySubset" --data_load_height 526 -crop_size_h 526 --crop_size_w 935 --metric "sKVD" --every_x_steps 2 | 

| Translation Task | cKVD Config | 
| :---: | :---: | 
| PFD → Cityscapes | --dataset_A_name "viper" --dataset_B_name "Cityscapes" --data_load_height 526 --crop_size_h 526 --crop_size_w 957 --metric "cKVD" --every_x_steps 4| 
| VIPER → Cityscapes | --dataset_A_name "pfd" --dataset_B_name "Cityscapes" --data_load_height 526 -crop_size_h 526 --crop_size_w 935 --metric "cKVD" --every_x_steps 8 | 
| BDDday → BDDnight |--dataset_A_name "BDDdaytimeSubset" --dataset_B_name "BDDnightSubset" --data_load_height 526 -crop_size_h 526 --crop_size_w 935 --metric "cKVD" --every_x_steps 8 | 
| BDDclear → BDDsnowy | --dataset_A_name "BDDclearSubset" --dataset_B_name "BDDsnowySubset" --data_load_height 526 -crop_size_h 526 --crop_size_w 935 --metric "cKVD" --every_x_steps 8 | 

#### 4.5 Creating a video from the inferred frames for VIPER or BDD
In the evaluation configuration of your model (e.g. ```evalMODELNAMEExperiment.json```). Make the following changes:
```json
   ...
      "doInference":0,
      "createVideoFromInference":1
   ...
```
then run the following command:
```console
docker run -it --rm --gpus device=0 --cpus 255 --shm-size=32G --ipc=host -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -u -m feamgan.main
```
The video will be saved in [Weights&Biases](https://wandb.ai/site).

#### 4.6 To share the results in the same way as in previous work, convert the inference directory to the same structure as EPE for the PFD → Cityscapes task.
Run the following command:
```console
docker run -it --rm --cpus 255 -v $PWD:/root/feamgan -w /root/feamgan feamgan_docker python -u -m feamgan.eval.convertPFDResults --model_name "MODEL_NAME"
```
The ```MODEL_NAME``` is specified in the configuration of the experiment. 
The inference directory is located in ```feamgan/experimentResults/MODEL_NAME/DATASET_NAME/repeatTrainingStep_0/```

## Experiment Configuration
For each experiment, .json configuration files must be defined to configure the models and their 
training parameters. You can use the original configuration files of the experiments from the paper as a template to 
configure your own experiments and register your configuration file in the ```experimetSchedule.json``` file.

## License
[MIT](https://choosealicense.com/licenses/mit/)
