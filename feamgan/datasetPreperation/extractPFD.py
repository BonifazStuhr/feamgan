import argparse
import os
import glob
import shutil
import contextlib

from tqdm import tqdm

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler
from feamgan.datasetPreperation.utils.datasetUtils import checkData, extractData

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="the relative save path of the data (default: '/data/PFD')",
                        nargs='?', default="/data/PFD", const="/data/PFD")
    parser.add_argument("--epe_baseline_path", type=str, help="the relative path to the epe baseline images (default: '/data/Baselines/epe')",
                        nargs='?', default="/data/Baselines/from_epe/epe/frames/0", const="/data/Baselines/from_epe/epe/frames/0")
    args = parser.parse_args()
    return args

def getDatasetParameters(epe_baseline_path):
        
    file_ids = ["01_images.zip", "02_images.zip", "03_images.zip", "04_images.zip", "05_images.zip", "06_images.zip", "07_images.zip", "08_images.zip","09_images.zip","10_images.zip",
                "01_labels.zip", "02_labels.zip", "03_labels.zip", "04_labels.zip", "05_labels.zip", "06_labels.zip", "07_labels.zip", "08_labels.zip","09_labels.zip","10_labels.zip"]

    file_types = [".png", ".png"] 
    data_types = ["images", "labels"] 
    data_type_names = ["frames", "segmentations"] 
    epe_ids = [f.split(".")[0] + ".png" for f in os.listdir(epe_baseline_path)]
    split_ids = [epe_ids]
    splits = ["train"]
    split_entries = [19252]
    split_seqs = split_entries

    return splits, split_ids, file_types, data_types, data_type_names, split_seqs, split_entries, file_ids

def reformateDataset(dataset_path, splits, split_ids, data_types, data_type_names, file_types):
    for split, ids in tqdm(zip(splits, split_ids), desc='Reformating subsets'):
        if not os.path.exists(f"{dataset_path}/{split}"):
            os.mkdir(f"{dataset_path}/{split}")

        for data_type, data_type_name, file_type in tqdm(zip(data_types, data_type_names, file_types), total=len(data_types), desc='Reformating datatypes'):                
            files = glob.glob(f"{dataset_path}/{data_type}/*{file_type}")
            if not os.path.exists(f"{dataset_path}/{split}/{data_type_name}"):
                os.mkdir(f"{dataset_path}/{split}/{data_type_name}")

            for file_path in tqdm(files, total=len(files), desc='Reformating files'):
                file_name = os.path.basename(file_path)
                seq_id = file_name.split(".")[0] # To be compatible with the rest of the pipeline
                # if path already exists delete it.
                s_patch = f"{dataset_path}/{split}/{data_type_name}/{seq_id}"
                with contextlib.suppress(FileNotFoundError): 
                    os.remove(f"{s_patch}/{file_name}")
                if file_name in ids:
                    if not os.path.exists(s_patch):
                        os.mkdir(s_patch)
                    shutil.move(file_path, f"{s_patch}/{file_name}")
            shutil.rmtree(f"{dataset_path}/{data_type}")  
        shutil.copytree(f"{dataset_path}/{split}/", f"{dataset_path}/val/")  

def extractPFD(save_path, epe_baseline_path):   
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
    archive_save_path = save_path + "/zips"

    splits, split_ids, file_types, data_types, data_type_names, split_seqs, split_entries, file_ids = getDatasetParameters(epe_baseline_path)
    save_path = f"{save_path}/sequences"

    logger.info("Extracting data...", ":extractPFD")
    relevant_zip_files = [f"{archive_save_path}/{f}" for f in file_ids]
    extractData(save_path, relevant_zip_files)
    logger.info("Extraction complete.", ":extractPFD")

    logger.info("Reformating data to fit the convention...", ":extractPFD")                     
    reformateDataset(save_path, splits, split_ids, data_types, data_type_names, file_types)
    logger.info("Reformating data complete.", ":extractPFD")
    
    logger.info("Checking dataset...", ":extractPFD")
    for data_type_name, file_type in tqdm(zip(data_type_names, file_types), total=len(data_type_names), desc="Checking datatypes"):    
        for split, seqs, num_etries in tqdm(zip(splits, split_seqs, split_entries), total=len(splits), desc="Checking subsets"):
            checkData(data_path=f"{save_path}/{split}/{data_type_name}", file_type=file_type,
                        expected_sequences=seqs, expected_entries=num_etries)
    logger.info("Checking dataset complete.", ":extractPFD")


if __name__ == "__main__":
    args = parseArguments()
    save_path = args.save_path
    save_path = os.path.dirname(os.path.abspath("feamgan")) + save_path
    epe_baseline_path = args.epe_baseline_path
    epe_baseline_path = os.path.dirname(os.path.abspath("feamgan")) + epe_baseline_path
    extractPFD(save_path, epe_baseline_path)