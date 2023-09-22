import argparse
import os
import glob
import shutil
import random

from tqdm import tqdm

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="the relative save path of the data (default: '/data/BDDnightSubset')",
                        nargs='?', default="/data/PFD", const="/data/PFD")
    parser.add_argument("--dataset_subset", type=str, help="the subset to create the subset for (default: 'val')",
                        nargs='?', default="val", const="val")
    parser.add_argument("--subset_size", type=int, help="the size of the subset to create (default: 2000)",
                        nargs='?', default=2000, const=2000)
    args = parser.parse_args()
    return args

def createDatasetSubsetSubset(dataset_path, subset, subset_size):   
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
    logger.info("Converting segmentations...", ":createDatasetSubsetSubset")

    seg_exists = os.path.exists(f"{dataset_path}/sequences/{subset}/segmentations")
    frames = glob.glob(f"{dataset_path}/sequences/{subset}/frames/*/*")
    msegs = glob.glob(f"{dataset_path}/sequences/{subset}/segmentations_mseg/*/*.png")

    if seg_exists:
        segs = glob.glob(f"{dataset_path}/sequences/{subset}/segmentations/*/*.png")
    
    assert len(frames) == len(msegs)
    if seg_exists:
        assert len(frames) == len(segs)

    random_indices = random.sample(range(0,len(frames)), subset_size)

    for i in tqdm(random_indices, desc='Copying subset...'):
        frame_dest = frames[i].replace(f"/{subset}/", f"/{subset}{subset_size}/")
        os.makedirs(os.path.dirname(frame_dest), exist_ok=True)
        shutil.copy(frames[i], frame_dest)

        mseg_dest = msegs[i].replace(f"/{subset}/", f"/{subset}{subset_size}/")
        os.makedirs(os.path.dirname(mseg_dest), exist_ok=True)
        shutil.copy(msegs[i], mseg_dest)
        if seg_exists: 
            seg_dest = segs[i].replace(f"/{subset}/", f"/{subset}{subset_size}/")
            os.makedirs(os.path.dirname(seg_dest), exist_ok=True)
            shutil.copy(segs[i], seg_dest)
   
    logger.info("Converting segmentations complete.", ":createDatasetSubsetSubset")

if __name__ == "__main__":
    args = parseArguments()
    dataset_path = args.dataset_path
    dataset_subset = args.dataset_subset
    subset_size = args.subset_size
    dataset_path = os.path.dirname(os.path.abspath("feamgan")) + dataset_path
    createDatasetSubsetSubset(dataset_path, dataset_subset, subset_size)