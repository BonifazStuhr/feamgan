import argparse
import os
import glob

import numpy as np

from PIL import Image
from tqdm import tqdm

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="the relative save path of the data (default: '/data/PFD')",
                        nargs='?', default="/data/PFD", const="/data/PFD")
    parser.add_argument("--subset", type=str, help="the subset to convert (default: 'train')",
                        nargs='?', default="train", const="train")
    args = parser.parse_args()
    return args

def convertSegmentations(save_path, subset):   
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
    logger.info("Converting segmentations...", ":convertSegmentations")
    path = f"{save_path}/sequences/{subset}/segmentations"
    for seg_path in tqdm(sorted(glob.glob(f"{path}/*/*.png")), desc='Reformating segmentations'):
        with Image.open(seg_path) as im:
            im = np.asarray(im)  
        im = Image.fromarray(im, "L")
        im.save(seg_path)  
    logger.info("Converting segmentations complete.", ":convertSegmentations")

if __name__ == "__main__":
    args = parseArguments()
    save_path = args.save_path
    subset = args.subset
    save_path = os.path.dirname(os.path.abspath("feamgan")) + save_path
    convertSegmentations(save_path, subset)