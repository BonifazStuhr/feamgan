import os
import glob
import subprocess
import argparse
import shutil

import numpy as np

from tqdm import tqdm

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="the dataset to create the mseg segmetnations for",
                        nargs='?', default="/data/Cityscapes", const="/data/Cityscapes")
    parser.add_argument("--rank", type=int, help="the rank of the current proccess (default: 0)",
                    nargs='?', default=0, const=0)
    parser.add_argument("--skip_if_exists", type=bool, help="skipps the creation of segmentations if they already exist (default: 1)",
                    nargs='?', default=True, const=True)
    parser.add_argument("--num_gpus", type=int, help="the number of gpus to devide the process (default: 1)",
                    nargs='?', default=1, const=1)
    parser.add_argument("--prefix", type=str, help="the prefix for the split eg. 'daytime/' for the daytime folder in BDD100K (default: "")",
                    nargs='?', default="", const="")
    args = parser.parse_args()
    return args

def createMSegSegmentations(dataset_path, rank, num_gpus, skip_if_exists, prefix):
    root_dir = os.path.dirname(os.path.abspath("feamgan"))
    model_name= "mseg-3m"
    model_path= f"{root_dir}/models/mseg-3m.pth"
    config="/workspace/unique_for_mseg_semantic/mseg-semantic/mseg_semantic/config/test/default_config_360_ms.yaml" # predictions are often visually better when we feed in test images at 360p resolution.

    splits = [f"{prefix}train", f"{prefix}val", f"{prefix}test"]
    for split in splits:
        print(f"{dataset_path}/sequences/{split}")
        if not os.path.exists(f"{dataset_path}/sequences/{split}"):
            continue     
        file_paths = [x for x in sorted(glob.glob(f"{dataset_path}/sequences/{split}/frames/*"))] # For now we do it file by file because the e gpu runs out of memmory in the demo script if we use it on a entire folder
        file_paths = np.array_split(file_paths, num_gpus)

        for file_path in tqdm(file_paths[rank], total=len(file_paths[rank]), desc="Creating semantic segmentations"):  
            dir_path = f"{dataset_path}/sequences/{split}/{rank}/gray"
            seq_id = file_path.split("/")[-1]
            save_dir = f"{dataset_path}/sequences/{split}/segmentations_mseg/{seq_id}"
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
            if skip_if_exists and os.path.exists(save_dir) and os.path.isdir(save_dir):
                print(f"Skipping seq {seq_id}. {save_dir} already exists.")
                continue
            subprocess.check_call([f"python -u /workspace/unique_for_mseg_semantic/mseg-semantic/mseg_semantic/tool/universal_demo.py --config={config} model_name {model_name} model_path {model_path} input_file {file_path} save_folder {dataset_path}/sequences/{split}/{rank}"], shell=True)          
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  
            for file_name in os.listdir(dir_path):
                shutil.move(f"{dir_path}/{file_name}", save_dir)
   
if __name__ == "__main__":
    args = parseArguments()
    dataset_path = os.path.dirname(os.path.abspath("feamgan")) + args.dataset_path
    rank = args.rank
    num_gpus = args.num_gpus
    skip_if_exists = args.skip_if_exists
    prefix = args.prefix
    createMSegSegmentations(dataset_path, rank, num_gpus, skip_if_exists, prefix)