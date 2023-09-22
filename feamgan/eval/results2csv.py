import os
import glob
import json

from tqdm import tqdm

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler

def results2csv(results_path):   
    logger = SLoggerHandler().getLogger(LoggerNames.INPUT_C)
    logger.info("Creating csv...", ":results2csv")
    result_paths = sorted(glob.glob(f"{results_path}/*.txt"))
   
    with open('csv_results.csv','wb') as file:
        header = "name;IS;FID;KID;sKVD;all;sky;ground;road;terrain;vegetation;building-parent;roadside-obj.;person-parent;vehicle;rest"
        file.write(header.encode('utf-8'))
        file.write("\n".encode('utf-8'))

        for result_path in tqdm(result_paths, desc='creating csv ...'):
            print(result_path)
            with open(result_path) as f:
                s = f.read()
                s = s.replace("{1:","{'1':").replace("'", "\"")
                print(s)
                d = json.loads(s)
            for key in d:
                if "cKVD" in result_path:
                    d[key] = str(d[key]["1"]).replace(".", ",")
                else:
                    d[key] = str(d[key]).replace(".", ",")
            name = result_path.split("/")[-1].split(".")[0]
            if "cKVD" in result_path:
                line = f'{name}; ; ; ; ; {d["all"]};{d["sky"]};{d["ground"]};{d["road"]};{d["terrain"]};{d["vegetation"]};{d["building-parent"]};{d["roadside-obj."]};{d["person-parent"]};{d["vehicle"]};{d["rest"]}'
            elif "sKVD" in result_path:
                line = f'{name}; ; ; ;  {d["1"]};;;;;;;;;;;'
            elif "quickEval" in result_path:
                line = f'{name};{d["inception_score_mean"]} ;{d["frechet_inception_distance"]} ;{d["kernel_inception_distance_mean"]} ;;;;;;;;;;;;'
            file.write(line.encode('utf-8'))
            file.write("\n".encode('utf-8'))
    logger.info("Creating csv complete.", ":results2csv")

if __name__ == "__main__":
    results_path = os.path.dirname(os.path.abspath("feamgan")) + "/results"
    results2csv(results_path)