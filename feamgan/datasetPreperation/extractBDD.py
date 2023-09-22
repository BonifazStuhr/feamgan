import argparse
import os
import shutil
import contextlib
import json
import glob
import skvideo
import skvideo.io

from pathlib import Path
from tqdm import tqdm

from feamgan.datasetPreperation.utils.datasetUtils import extractData, checkZipDownload
from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="the relative save path of the data (default: '/data/BDD100k')",
                        nargs='?', default="/data/BDD100k", const="/data/BDD100k")
    parser.add_argument("--split_into_tw_subsets", type=bool, help="specify the shoudl be split into the subsets: [rainy|snowy|clear|overcast|partly cloudy|foggy] and [daytime|night|dawn/dusk]. 'True' by default.",
                        nargs='?', default=True, const=True)
    parser.add_argument("--skip_if_exists", type=bool, help="skipps the formating of sequences if they already exist (default: 1)",
                    nargs='?', default=True, const=True)
    args = parser.parse_args()
    return args

def getDatasetParameters(split_into_tw_subsets):
    splits = ["train", "val"]
    data_type_names = ["frames"] 
    data_num = "100k"
    
    nr_no_label = [137, 0]
    if split_into_tw_subsets:
        nr_no_condition_files_time = [137, 172]
        nr_no_condition_files_weather = [8119, 9276]

    data_types = ["videos"] 
    bdd_package_list = ["bdd100k_videos_train_00.zip", "bdd100k_videos_val_00.zip", "bdd100k_labels_release.zip"]
    bdd_package_list_md5 = None

    return splits, data_types, data_type_names, bdd_package_list, bdd_package_list_md5, data_num, nr_no_label, nr_no_condition_files_time, nr_no_condition_files_weather

def cleanDirs(save_path):
    with contextlib.suppress(FileNotFoundError): 
        shutil.rmtree(f"{save_path}")

def reformateDataset(save_path, data_types, data_type_names, splits, nr_no_condition_files_time, nr_no_condition_files_weather, nr_no_label, data_num, split_into_tw_subsets, skip_if_exists, logger):
    data_types_bdd_package_pbar = tqdm(zip(data_types, data_type_names), total=len(data_types), desc='Reformating datatypes')
    for data_type, data_type_name in data_types_bdd_package_pbar:
        # For each split
        undefined_time_conditions = 0
        undefined_weather_conditions = 0
        for split, nr_no_cond_t, nr_no_cond_w, nr_no_l in tqdm(zip(splits, nr_no_condition_files_time, nr_no_condition_files_weather, nr_no_label), total=len(splits), desc='Reformating subsets'):

            with open(f"{save_path}/bdd100k/labels/bdd{data_num}_labels_images_{split}.json") as json_file:
                    labels = json.load(json_file)
            if data_type == "videos":
                frames_dir = f"{save_path}/bdd100k/{data_type}/{split}"         
            else:
                frames_dir = f"{save_path}/bdd100k/{data_num}/{data_type}/{split}"

            if split_into_tw_subsets:
                sequenze_index = {"rainy":1, "snowy":1, "clear":1, "overcast":1, "partlycloudy":1, "foggy":1, "daytime":1, "night":1, "dawndusk":1}
                current_seq_id = {"rainy":None, "snowy":None, "clear":None, "overcast":None, "partlycloudy":None, "foggy":None, "daytime":None, "night":None, "dawndusk":None}
            else:
                sequenze_index = {"all":1}
                current_seq_id = {"all":None}

            data_types_bdd_package_pbar.write(f"Moving sequences from {frames_dir} to {save_path} ...")
            
            for label in tqdm(labels, desc='Reformating entries'):       
                 
                f = label["name"]
                if data_type == "videos":
                    f = f.replace(".jpg", ".mov")
                
                if split_into_tw_subsets:
                    frame_condition_time = label["attributes"]["timeofday"]
                    frame_condition_weather = label["attributes"]["weather"]

                scene = label["attributes"]["scene"].replace(" ", "") # Just for additional information in the file name

                if split_into_tw_subsets:
                    if frame_condition_time == "undefined":
                        undefined_time_conditions +=1
                    if frame_condition_weather == "undefined":
                        undefined_weather_conditions += 1

                    if (frame_condition_time == "undefined") and (frame_condition_weather == "undefined"):
                        data_types_bdd_package_pbar.write("No weather/daytime conditions given for this frame. Skipping and removing this frame.")
                        if data_type == "videos":
                            if os.path.exists(f"{frames_dir}/{f}"):
                                os.remove(f"{frames_dir}/{f}")
                        else:
                            os.remove(f"{frames_dir}/{f}")
                        continue
                    
                    frame_condition_time = frame_condition_time.replace("/", "")
                    frame_condition_time = frame_condition_time.replace(" ", "")
                    frame_condition_weather = frame_condition_weather.replace("/", "")
                    frame_condition_weather = frame_condition_weather.replace(" ", "")

                else:
                    frame_condition_time = "all"
                    frame_condition_weather = "undefined"

              
                prev = None
                for frame_condition in [frame_condition_time, frame_condition_weather]:
                    if frame_condition == "undefined":
                        continue
                    
                    if os.path.exists(f"{save_path}/{frame_condition}/{split}/{data_type_name}/{str(sequenze_index[frame_condition]).zfill(6)}") and skip_if_exists:
                        data_types_bdd_package_pbar.write("Sequence already exists, skipping this sample")
                        continue

                    seq_id = sequenze_index[frame_condition]                     
                    if seq_id != current_seq_id[frame_condition]:
                        new_dir = f"{save_path}/{frame_condition}/{split}/{data_type_name}/{str(sequenze_index[frame_condition]).zfill(6)}"
                        Path(new_dir).mkdir(parents=True, exist_ok=True)
                        data_types_bdd_package_pbar.write(f"New sequence found in: {frames_dir}")
                        data_types_bdd_package_pbar.write(f"Moving sequence {str(sequenze_index).zfill(6)} from {frames_dir} to {new_dir} ...")
         
                    current_seq_id[frame_condition] = seq_id

                    new_file_name = f"{str(sequenze_index[frame_condition]).zfill(6)}_{str(1).zfill(9)}_{frame_condition}_{scene}_{f}" 
  
                    if data_type == "videos":
                        if prev is None:
                            if os.path.exists(f"{frames_dir}/{f}"):
                                videogen = skvideo.io.vreader(f"{frames_dir}/{f}")
                                n = f.replace(".mov", ".jpg")  
                                index=1                 
                                for frame in videogen:
                                    skvideo.io.vwrite(f"{new_dir}/{str(sequenze_index[frame_condition]).zfill(6)}_{str(index).zfill(9)}_{frame_condition}_{scene}{n}", frame)
                                    index+=1
                                sequenze_index[frame_condition] += 1
                                prev = frame_condition
                            else:
                                with contextlib.suppress(FileNotFoundError): 
                                    shutil.rmtree(f"{new_dir}")
                        else:
                            index = 1 
                            for file in sorted(glob.glob(f"{save_path}/{prev}/{split}/{data_type_name}/{str(sequenze_index[prev]-1).zfill(6)}/*.jpg")):
                                shutil.copy(file, f"{new_dir}/{str(sequenze_index[frame_condition]).zfill(6)}_{str(index).zfill(9)}_{frame_condition}_{scene}.jpg")
                                index+=1
                            sequenze_index[frame_condition] += 1
                    else:
                        shutil.move(f"{frames_dir}/{f}", f"{new_dir}/{new_file_name}")
                        sequenze_index[frame_condition] += 1

                    prev = frame_condition

                if os.path.exists(f"{frames_dir}/{f}"):
                    os.remove(f"{frames_dir}/{f}") 

            # Some sanity checks
            if undefined_time_conditions:
                if undefined_time_conditions == nr_no_cond_t:
                    data_types_bdd_package_pbar.write(f"There were {undefined_time_conditions} files with undefined daytime in {frames_dir}. {nr_no_cond_t} files with undefined daytime are expected.")
                else:
                    logger.error(f"There were {undefined_time_conditions} files with undefined daytime in {frames_dir}. {nr_no_cond_t} files with undefined daytime are expected. Something went wrong!", ":extractBDD")

            if undefined_weather_conditions:
                if undefined_weather_conditions == nr_no_cond_w:
                    data_types_bdd_package_pbar.write(f"There were {undefined_weather_conditions} files with undefined weather in {frames_dir}. {nr_no_cond_w} files with undefined weather are expected.")
                else:
                    logger.error(f"There were {undefined_weather_conditions} files with undefined weather in {frames_dir}. {nr_no_cond_w} files with undefined weather are expected. Something went wrong!", ":extractBDD")

            remaining_files = os.listdir(f"{frames_dir}")
            if not remaining_files:
                data_types_bdd_package_pbar.write(f"Moved all sequences from {frames_dir}, deleting empty folder...")
                shutil.rmtree(frames_dir)
            else:
                if len(remaining_files) is nr_no_l:
                    data_types_bdd_package_pbar.write(f"There are {len(remaining_files)} remaining files with no label in {frames_dir}. {nr_no_l} files with no label are expected.")
                else:
                    logger.error(f"There are {len(remaining_files)} remaining files with no label in {frames_dir}. {nr_no_l} files with no label are expected. Something went wrong!", ":extractBDD")

    shutil.rmtree(f"{save_path}/bdd100k")

def extractBDD(save_path, split_into_tw_subsets, skip_existing):
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
    
    archive_save_path = f"{save_path}/zips"
    Path(archive_save_path).mkdir(parents=True, exist_ok=True)
    
    splits, data_types, data_type_names, bdd_package_list, bdd_package_list_md5, data_num, nr_no_label, nr_no_condition_files_time, nr_no_condition_files_weather = getDatasetParameters(split_into_tw_subsets)

    logger.info("Checking zips...", ":extractBDD")
    checkZipDownload(archive_save_path, bdd_package_list, bdd_package_list_md5)
    logger.info("Checking zips complete.", ":extractBDD")

    save_path = f"{save_path}/sequences"
    if not skip_if_exists:
        cleanDirs(save_path)

    if not skip_if_exists or not os.path.exists(f"{save_path}/bdd100k/videos"):
        logger.info("Extracting data...", ":extractBDD")
        file_paths = [f"{archive_save_path}/{package}" for package in bdd_package_list]
        extractData(save_path, file_paths)
        logger.info("Extraction complete.", ":extractBDD")

    logger.info("Reformating data to fit the convention...", ":extractBDD")
    reformateDataset(save_path, data_types, data_type_names, splits, nr_no_condition_files_time, nr_no_condition_files_weather, nr_no_label, data_num, split_into_tw_subsets, skip_if_exists, logger)
    logger.info("Reformating data complete.", ":extractBDD")
        
        
if __name__ == "__main__":
    args = parseArguments()
     # Save paths relative from project dir
    save_path = args.save_path
    save_path = os.path.dirname(os.path.abspath("feamgan")) + save_path

    split_into_tw_subsets = args.split_into_tw_subsets
    skip_if_exists = args.skip_if_exists

    extractBDD(save_path, split_into_tw_subsets, skip_if_exists)