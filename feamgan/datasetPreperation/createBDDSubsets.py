import os
import shutil

from tqdm import tqdm

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler


def createBDDSubsets(project_path, bdd_subset, dataset_defintion_txt_path_train, dataset_defintion_txt_path_val):   
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
    logger.info(f"Creating BDD100k subsets for {bdd_subset} ...", ":createBDDSubsets")

    for dataset_definition_path_txt, split in zip([dataset_defintion_txt_path_train, dataset_defintion_txt_path_val], ["train", "val"]):
        
        print(f"Creating subsets for {split} ...")

        new_path = f"{project_path}/data/BDD{bdd_subset}Subset/sequences/{split}"
        old_path = f"{project_path}/data/BDD100k/sequences/{bdd_subset}/{split}"
        
        with open(os.path.join(project_path, dataset_definition_path_txt)) as s_list:
            subset_paths = s_list.read().splitlines()
        
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        for subset_path in tqdm(subset_paths):
            subset_path_frame = f"frames/{subset_path}" 
            subset_path_seg = f"segmentations_mseg/{subset_path}".replace(".jpg",".png")  
            new_frame_file_path = f"{new_path}/{subset_path_frame}"
            new_seg_file_path = f"{new_path}/{subset_path_seg}"

            frame_dir = "/"+ os.path.join(*new_frame_file_path.split("/")[:-1])
            seg_dir = "/"+ os.path.join(*new_seg_file_path.split("/")[:-1])
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
            if not os.path.exists(seg_dir):
                os.makedirs(seg_dir)

            shutil.copy(f"{old_path}/{subset_path_frame}", new_frame_file_path)     
            shutil.copy(f"{old_path}/{subset_path_seg}", new_seg_file_path)      
   

    logger.info(f"Creating BDD100k subsets for {bdd_subset} complete.", ":createBDDSubsets")

if __name__ == "__main__":
    for bdd_subset in ["daytime", "night", "clear", "snowy"]:
        dataset_defintion_txt_path_train = f"feamgan/datasetPreperation/BDDSubsets/bdd_{bdd_subset}_train.txt"
        dataset_defintion_txt_path_val = f"feamgan/datasetPreperation/BDDSubsets/bdd_{bdd_subset}_val.txt"
        project_path = os.path.dirname(os.path.abspath("feamgan"))
        createBDDSubsets(project_path, bdd_subset, dataset_defintion_txt_path_train, dataset_defintion_txt_path_val)