# This script downloads the parts of the VIPER Dataset: https://playing-for-benchmarks.org/download/.
# It downloads all frames from each sequence of the training and evaluation data with segmetation maps.
# The data fill be but in the following foldet structure: 
#       save_path/train/frames/seq_nums/frames_of_seq
#       save_path/train/segmetations/seq_nums/segmetations_of_frames_of_seq
#       save_path/eval/frames/seq_nums/frames_of_seq
#       save_path/eval/segmetations/seq_nums/segmetations_of_frames_of_seq
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
    parser.add_argument("--save_path", type=str, help="the relative save path of the data (default: '/data/VIPER')",
                        nargs='?', default="/data/VIPER", const="/data/VIPER")
    parser.add_argument("--training_data_only", type=bool, help="if True, only the training data will be downloaded (default: False)",
                        nargs='?', default=False, const=False)
    parser.add_argument("--evaluation_data_only", type=bool, help="if True, only the evaluation data will be downloaded (default: False)",
                        nargs='?', default=False, const=False)
    parser.add_argument("--remove_unwanded_sequences", type=bool, help="if True, unwanted sequences for viper to cityscapes transfer (like night) will be removes (default: False)",
                        nargs='?', default=True, const=True)
    args = parser.parse_args()
    return args

def getDatasetParameters(training_data_only, evaluation_data_only, unwanted_sequences):
    file_types = [".png", ".png"] 
    data_types = ["img", "cls"] 
    data_type_names = ["frames", "segmentations"] 
    split_seqs = [77, 47] 
    splits = ["train", "val"]

    split_entries = [134097, 49815]
    if training_data_only: 
        splits = ["train"]
    elif evaluation_data_only: 
        splits = ["val"]

    split_seqs_unwanded_removed = [56, 37]
    split_entries_unwanded_removed = [103086, 39538] 
    for i, split in enumerate(splits):
        if unwanted_sequences[split]:
            split_seqs[i] = split_seqs_unwanded_removed[i]
            split_entries[i] = split_entries_unwanded_removed[i]

    return splits, file_types, data_types, data_type_names, split_seqs, split_entries

def reformateDataset(dataset_path, splits, data_types, data_type_names, unwanted_sequences):
    for split in tqdm(splits, desc='Reformating subsets'):
        for data_type, data_type_name in tqdm(zip(data_types, data_type_names), total=len(data_types), desc='Reformating datatypes'):
            # if path already exists delete it.
            with contextlib.suppress(FileNotFoundError): 
                shutil.rmtree(f"{dataset_path}/{split}/{data_type_name}")
            shutil.move(f"{dataset_path}/{split}/{data_type}", f"{dataset_path}/{split}/{data_type_name}")

            if unwanted_sequences[split]:
                frame_dirs = sorted(glob.glob(f"{dataset_path}/{split}/{data_type_name}/*/"))
                for d in tqdm(frame_dirs, desc='Reformating sequences'):
                    seq_num = d.split("/")[-2]
                    if seq_num in unwanted_sequences[split]:
                        shutil.rmtree(d)
  
def extractVIPER(save_path, training_data_only, evaluation_data_only, unwanted_sequences={"train":False, "val":False}):   
    
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
  
    splits, file_types, data_types, data_type_names, split_seqs, split_entries = getDatasetParameters(training_data_only, evaluation_data_only, unwanted_sequences)
    
    archive_save_path = f"{save_path}/zips"
    save_path = f"{save_path}/sequences"
   
    logger.info("Extracting data...", ":extractVIPER")
    relevant_zip_files = glob.glob(f"{archive_save_path}/*.zip") 
    extractData(save_path, relevant_zip_files)
    logger.info("Extraction complete.", ":extractVIPER")

    logger.info("Reformating data to fit the convention...", ":extractVIPER")                     
    reformateDataset(save_path, splits, data_types, data_type_names, unwanted_sequences)
    logger.info("Reformating data complete.", ":extractVIPER")

    logger.info("Checking dataset...", ":extractVIPER")
    for data_type_name, file_type in tqdm(zip(data_type_names, file_types), total=len(data_type_names), desc="Checking datatypes"):    
        for split, seqs, num_etries in tqdm(zip(splits, split_seqs, split_entries), total=len(splits), desc="Checking subsets"):
            checkData(data_path=f"{save_path}/{split}/{data_type_name}", file_type=file_type,
                        expected_sequences=seqs, expected_entries=num_etries)
    logger.info("Checking dataset complete.", ":extractVIPER")


if __name__ == "__main__":
    args = parseArguments()
    training_data_only = args.training_data_only
    evaluation_data_only = args.evaluation_data_only

    # Save paths relative from project dir
    save_path = args.save_path
    if args.remove_unwanded_sequences:
        unwanted_sequences = {"train":["008","009","010","011","012","013","052","053","054","055","056","057","058","070","071","072","073","074","075","076","077"], "val":["005","006","007","033","034","035","036","045","046","047"]} # night sequences
    else:
        unwanted_sequences = {"train":False, "val":False}
    save_path = os.path.dirname(os.path.abspath("feamgan")) + save_path
    extractVIPER(save_path, training_data_only, evaluation_data_only, unwanted_sequences)