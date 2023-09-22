import argparse
import os
import scipy.io
import glob
import shutil

from tqdm import tqdm

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler
from feamgan.datasetPreperation.utils.datasetUtils import extractData

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="the name of the model to evaluate (default: 'model')",
                        nargs='?', default="FeaMGAN_PFD_to_CS_Crop352_Full_r1", const="FeaMGAN_PFD_to_CS_Crop352_Full_r1")
    parser.add_argument("--repetition", type=int, help="the training repetition to evalaute (default: 0)",
                        nargs='?', default=0, const=0)
    args = parser.parse_args()
    return args

def convertPFDResults(model_name, repetition):   
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
    logger.info("Reformating PFD results...", ":convertPFDResults")

    project_path = os.path.dirname(os.path.abspath("feamgan"))
    path = f"{project_path}/experimentResults/{model_name}/pfd/repeatTrainingStep_{repetition}/inference/eval"
    input_dir_1 = f"{path}/fake_frames_B"
    input_dir_2 = f"{path}/real_frames_A"
    baseline_dir = f"{project_path}/data/Baselines/from_epe/epe/frames/0"

    input_files_1 = sorted(glob.glob(f"{input_dir_1}/*/*.jpg"))
    input_files_2 = sorted(glob.glob(f"{input_dir_2}/*/*.jpg"))
    baseline_files_2 = sorted(glob.glob(f"{baseline_dir}/*.jpg"))

    for in_1, in_2, base in tqdm(zip(input_files_1,input_files_2,baseline_files_2), desc='Reformating pfd results'):
        new_name_1_split = in_1.split("/")
        new_name_2_split = in_2.split("/")
        
        new_name_1 = '/'.join(new_name_1_split[:-2]) + "/" + base.split("/")[-1]
        new_name_2 = '/'.join(new_name_2_split[:-2]) + "/" +  base.split("/")[-1]
 
        shutil.move(in_1, new_name_1)
        shutil.move(in_2, new_name_2)

        os.rmdir('/'.join(new_name_1_split[:-1]))
        os.rmdir('/'.join(new_name_2_split[:-1]))
  
    logger.info("Reformating PFD results complete.", ":convertPFDResults")

if __name__ == "__main__":
    args = parseArguments()
    model_name = args.model_name
    repetition = args.repetition
    convertPFDResults(model_name, repetition)