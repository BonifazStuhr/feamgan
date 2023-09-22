import glob
import zipfile
import tarfile
import hashlib

from tqdm import tqdm
from pathlib import Path

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler

def extractData(extract_path, file_paths):
    Path(extract_path).mkdir(parents=True, exist_ok=True)
    file_paths_pbar = tqdm(file_paths, desc="Extracting files")
    for f in file_paths_pbar:
        file_paths_pbar.write(f"Extracting file {f} to {extract_path}...\n")
        if f.endswith(".zip"):
            with zipfile.ZipFile(f, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif f.endswith(".tar.gz"):
            tar = tarfile.open(f, "r:gz")
            tar.extractall(extract_path)
            tar.close()

def checkZipDownload(download_path, expected_file_names, md5_list=None, a_type=".zip"):
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
    
    if md5_list:
        for fn, md5 in tqdm(zip(expected_file_names, md5_list), total=len(expected_file_names), desc="Checking md5 sums"):
            f = Path(f"{download_path}/{fn}")
            check_sum = hashlib.md5(open(f,'rb').read()).hexdigest()
            if check_sum != md5:
                logger.error(f"Md5 check sum does not match. File {f} is corrupted!", ":checkZipDownload")

    for fn in tqdm(expected_file_names, total=len(expected_file_names), desc="Checking zip files"):
        if not fn.endswith(a_type):
            fn = f"{fn}{a_type}"
        f = Path(f"{download_path}/{fn}")
        if not f.is_file():
            logger.error(f"Error: {a_type} file {f} does not exist!", ":checkZipDownload")
            continue
        try:
            if a_type == ".zip":
                x = zipfile.ZipFile(f)
            elif a_type == ".tar.gz":
                x = tarfile.open(f, "r:gz")
                x.close()
        except:
            logger.error(f"File {f} is corrupted!", ":checkZipDownload")
        
def checkAndPrintFoundExpected(found, expected, name):
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
    if found != expected:
        logger.error(f"Number of {name} not as expected!", ":checkAndPrintFoundExpected")
        logger.error(f"Found: {found}", ":checkAndPrintFoundExpected")
        logger.error(f"Expected: {expected}", ":checkAndPrintFoundExpected")
        return False
    tqdm.write(f"Found {found} of {expected} {name}")
    return True

def checkData(data_path, file_type, expected_sequences, expected_entries):
    num_seq = len(glob.glob(f"{data_path}/*"))
    num_entries = len(glob.glob(f"{data_path}/*/*{file_type}"))
    checkAndPrintFoundExpected(num_seq, expected_sequences, f"sequences in {data_path}")
    checkAndPrintFoundExpected(num_entries, expected_entries, f"entries from type {file_type} in {data_path}")