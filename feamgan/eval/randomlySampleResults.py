import os
import glob
import shutil

from tqdm import tqdm
import numpy as np

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler

# INFO: This script is for results not converted with convertPFDResults. 
# For results convertPFDResults change:
#        input_files_1 = sorted(glob.glob(f"{input_dir_1}/*/*.jpg"))
#        input_files_2 = sorted(glob.glob(f"{input_dir_2}/*/*.jpg"))
#   to
#        input_files_1 = sorted(glob.glob(f"{input_dir_1}/*.jpg"))
#        input_files_2 = sorted(glob.glob(f"{input_dir_2}/*.jpg"))
# in the function below
def randomlySampleResults(nr_random_samples, model_names, dataset_name, save_folder, repetition=0, baseline="epe", include_baseline=False):   
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
    logger.info("Randomly sampling results...", ":randomlySampleResults")

    first_run = True
    project_path = os.path.dirname(os.path.abspath("feamgan"))

    baseline_dir = f"{project_path}/data/Baselines/from_epe/{baseline}/frames/0"
    baseline_files = sorted(glob.glob(f"{baseline_dir}/*.jpg"))
    random_index = np.random.randint(0, high=len(baseline_files)-1, size=nr_random_samples) 

    for model_name in model_names:
        path = f"{project_path}/experimentResults/{model_name}/{dataset_name}/repeatTrainingStep_{repetition}/inference/eval"
        input_dir_1 = f"{path}/fake_frames_B"
        input_dir_2 = f"{path}/real_frames_A"
        input_files_1 = sorted(glob.glob(f"{input_dir_1}/*/*.jpg"))
        input_files_2 = sorted(glob.glob(f"{input_dir_2}/*/*.jpg"))
        #input_files_1 = sorted(glob.glob(f"{input_dir_1}/*.jpg"))
        #input_files_2 = sorted(glob.glob(f"{input_dir_2}/*.jpg"))

        save_path = os.path.join(project_path, save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            if first_run: os.makedirs(os.path.join(save_path, "source"))
            if include_baseline: os.makedirs(os.path.join(save_path, baseline))
        os.makedirs(os.path.join(save_path, model_name))
        
        for i, r_i in tqdm(zip(range(len(random_index)), random_index), desc=f'Randomly sampling results for {model_name} ...'):            
            shutil.copy(input_files_1[r_i], os.path.join(save_path, model_name, f"{i}.jpg"))
            if first_run:
                shutil.copy(input_files_2[r_i], os.path.join(save_path, "source",  f"{i}.jpg"))
                if include_baseline:
                    shutil.copy(baseline_files[r_i], os.path.join(save_path, baseline, f"{i}.jpg"))                              
        first_run = False
  
    logger.info("Randomly sampling results complete.", ":randomlySampleResults")

if __name__ == "__main__":
    nr_random_samples = 500
    model_names = ["FeaMGAN_PFD_to_CS_Crop352_Full_r1"]
    dataset_name= "pfd"
    save_folder = "random_results"
    randomlySampleResults(nr_random_samples, model_names, dataset_name, save_folder)