import argparse
import glob
import os
import zipfile
import subprocess
import shutil
import contextlib

from tqdm import tqdm
from pathlib import Path

from feamgan.datasetPreperation.utils.datasetUtils import extractData, checkZipDownload, checkData
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler
from feamgan.LoggerNames import LoggerNames

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="the relative save path of the data (default: '/data/Cityscapes')",
                        nargs='?', default="/data/Cityscapes", const="/data/Cityscapes")
    parser.add_argument("--download_only", type=bool, help="if True, the data will only be downloaded (default: False)",
                        nargs='?', default=False, const=False)
    parser.add_argument("--extract_only", type=bool, help="if True, the data will only be extracted, if the data exists (default: False)",
                        nargs='?', default=True, const=True)
    args = parser.parse_args()
    return args

def getDatasetParameters():
    splits = ["train", "val", "test"]
    file_types = [".png"] 
    data_types = ["leftImg8bit"] 
    data_type_names = ["frames"]  
    data_types = ["leftImg8bit_sequence"] 
    cityscapes_package_list = ["leftImg8bit_sequence_trainvaltest.zip"]  
    split_seqs = [1885, 235, 1051] 
    split_entries = [89250, 15000, 45750] 

    return splits, file_types, data_types, data_type_names, cityscapes_package_list, split_seqs, split_entries

def downloadDataset(archive_save_path, cityscapes_package_list):
    cityscapes_package_list_pbar = tqdm(cityscapes_package_list, desc="Downloading packages")
    for package in cityscapes_package_list_pbar:
        file_names = [os.path.basename(x) for x in glob.glob(f"{archive_save_path}/*.zip")]
        download_file = True
        if package in file_names:
            try:
                x = zipfile.ZipFile(f"{archive_save_path}/{package}")
                x.close()
                download_file = False
                cityscapes_package_list_pbar.write(f"File {package} already exists. Skipping the download of this package.")
            except:
                cityscapes_package_list_pbar.write(f"File {package} is corrupt removing the old file and downloading the file again...")
                os.remove(f"{archive_save_path}/{package}")

        if download_file:  
            cityscapes_package_list_pbar.write(f"Downloading the package {package} (no download progress is shown here during the download)...")      
            subprocess.check_call(["csDownload", "-d", archive_save_path, package], shell=False)     

def cleanDirs(save_path):
    with contextlib.suppress(FileNotFoundError): 
        shutil.rmtree(f"{save_path}/train")
        shutil.rmtree(f"{save_path}/val")
        shutil.rmtree(f"{save_path}/test")

def reformateDataset(save_path, data_types, data_type_names, splits, logger):
    data_types_bdd_package_pbar = tqdm(zip(data_types, data_type_names), total=len(data_types), desc='Reformating datatypes')
    for data_type, data_type_name in data_types_bdd_package_pbar:
        
        # Moving files on up into the folder of the corresponding split
        dir_names = [os.path.basename(x) for x in sorted(glob.glob(f"{save_path}/{data_type}/*"))]
        for d in dir_names:
            data_types_bdd_package_pbar.write(f"Moving {save_path}/{data_type}/{d} to {save_path}/{d}/{data_type_name} ...")
            shutil.move(f"{save_path}/{data_type}/{d}", f"{save_path}/{d}/{data_type_name}")

        if not os.listdir(f"{save_path}/{data_type}"):
            data_types_bdd_package_pbar.write(f"Moved all files from {save_path}/{data_type}, deleting empty folder...")
            shutil.rmtree(f"{save_path}/{data_type}")
        else:    
            logger.error(f"There are unresolved files in {save_path}/{data_type}, something went wrong!", ":downloadAndExtractCityscapes")

        # For each split
        for split in tqdm(splits, desc='Reformating subsets'):
            dir_names = [os.path.basename(x) for x in sorted(glob.glob(f"{save_path}/{split}/{data_type_name}/*"))]
            sequenze_index = 1
            for d in tqdm(dir_names, desc='Reformating dirs'):
                dir_with_sequences = f"{save_path}/{split}/{data_type_name}/{d}"
                files_in_dir = [os.path.basename(x) for x in sorted(glob.glob(f"{dir_with_sequences}/*"))]
                current_dir_seq_num = None
                new_dir = f"{save_path}/{split}/{data_type_name}/{str(sequenze_index).zfill(6)}"
                os.mkdir(new_dir)
                for f in files_in_dir:
                    file_name_parts = f.split("_")

                    new_file_name = None
                    if len(file_name_parts) == 5:
                        new_file_name = f"{file_name_parts[3]}_{file_name_parts[0]}_{file_name_parts[1]}_{file_name_parts[4]}"
                        dir_seq_num = file_name_parts[2]
                    elif len(file_name_parts) == 4:
                        new_file_name = f"{file_name_parts[2]}_{file_name_parts[0]}_{file_name_parts[3]}"
                        dir_seq_num = file_name_parts[1]
                    if current_dir_seq_num is not None: 
                        if dir_seq_num != current_dir_seq_num:
                            sequenze_index += 1
                            current_dir_seq_num = dir_seq_num
                            new_dir = f"{save_path}/{split}/{data_type_name}/{str(sequenze_index).zfill(6)}"
                            os.mkdir(new_dir)
                            data_types_bdd_package_pbar.write(f"New sequence found in current dir: {dir_with_sequences}")
                            data_types_bdd_package_pbar.write(f"Moving sequence {str(sequenze_index).zfill(6)} from {dir_with_sequences} to {new_dir} ...")
                    else:
                        current_dir_seq_num = dir_seq_num

                    new_file_name = f"{str(sequenze_index).zfill(6)}_{new_file_name}" 

                    shutil.move(f"{dir_with_sequences}/{f}", f"{new_dir}/{new_file_name}")
                if not os.listdir(dir_with_sequences):
                    data_types_bdd_package_pbar.write(f"Moved all sequences from {dir_with_sequences}, deleting empty folder...")
                    shutil.rmtree(dir_with_sequences)
                else:    
                    logger.error(f"There are unconverted files in {dir_with_sequences}, something went wrong!", ":downloadAndExtractCityscapes")
                sequenze_index += 1  

def downloadAndExtractCityscapes(save_path, download_only=False, extract_only=False):
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
   
    archive_save_path = f"{save_path}/zips"
    Path(archive_save_path).mkdir(parents=True, exist_ok=True)

    splits, file_types, data_types, data_type_names, cityscapes_package_list, split_seqs, split_entries = getDatasetParameters()

    if not extract_only:
        logger.info("Downloading data (this can take a very long time)...", ":downloadAndExtractCityscapes")
        logger.info(f"Saving files to: {archive_save_path}", ":downloadAndExtractCityscapes")
        downloadDataset(archive_save_path, cityscapes_package_list)
        logger.info("Downloading data done.", ":downloadAndExtractCityscapes")  

        logger.info("Checking download...", ":downloadAndExtractCityscapes")
        checkZipDownload(archive_save_path, cityscapes_package_list)
        logger.info("Checking download complete.", ":downloadAndExtractCityscapes")

    if not download_only:    
        save_path = f"{save_path}/sequences"

        logger.info("Extracting data...", ":downloadAndExtractCityscapes")
        file_paths = [f"{archive_save_path}/{package}" for package in cityscapes_package_list]
        extractData(save_path, file_paths)
        logger.info("Extraction complete.", ":downloadAndExtractCityscapes")

        logger.info("Reformating data to fit the convention...", ":downloadAndExtractCityscapes")  
        cleanDirs(save_path)
        reformateDataset(save_path, data_types, data_type_names, splits, logger)
        logger.info("Reformating data complete.", ":downloadAndExtractCityscapes")
        
        logger.info("Checking dataset...", ":downloadAndExtractCityscapes")
        for data_type_name, file_type in tqdm(zip(data_type_names, file_types), total=len(data_type_names), desc="Checking datatypes"):  
            for split, seqs, num_etries in tqdm(zip(splits, split_seqs, split_entries), total=len(splits), desc="Checking subsets"):
                checkData(data_path=f"{save_path}/{split}/{data_type_name}", file_type=file_type,
                            expected_sequences=seqs, expected_entries=num_etries)
        logger.info("Checking dataset complete.", ":downloadAndExtractCityscapes")

if __name__ == "__main__":
    args = parseArguments()
    download_only = args.download_only
    extract_only = args.extract_only
    save_path = args.save_path
    save_path = os.path.dirname(os.path.abspath("feamgan")) + save_path
    downloadAndExtractCityscapes(save_path, download_only, extract_only)