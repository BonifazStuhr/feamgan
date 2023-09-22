import argparse
import os

from torch_fidelity import calculate_metrics

from feamgan.utils import distUtils

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="the name of the model to evaluate (default: 'model')",
                        nargs='?', default="FeaMGAN_PFD_to_CS_Crop352_Full_r1", const="FeaMGAN_PFD_to_CS_Crop352_Full_r1")
    parser.add_argument("--dataset_A_name", type=str, help="the name of dataset_A_name, the dataset of source images (default: pfd)",
                        nargs='?', default="pfd", const="pfd")
    parser.add_argument("--dataset_B_name", type=str, help="the name of dataset_B_name, the dataset of target images (default: Cityscapes)",
                        nargs='?', default="Cityscapes", const="Cityscapes")
    parser.add_argument("--subset", type=str, help="the subset to evalaute (default: eval)",
                        nargs='?', default="eval", const="eval")
    parser.add_argument("--repetition", type=int, help="the training repetition to evalaute (default: 0)",
                        nargs='?', default=0, const=0)
    parser.add_argument("--model_path", type=str, help="if model_path is specified, the metrics will be calculated for the model found in model_path (default: 0)",
                        nargs='?', default="", const="")
    args = parser.parse_args()
    return args

def quickEval(model_name, dataset_A_name, dataset_B_name, subset, repetition, model_path):
    logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)
   
    os.environ['TORCH_HOME'] = 'models'
    project_path = os.path.dirname(os.path.abspath("feamgan"))

    if model_path:    
        logger.val(f"Evaluating {model_path}...", ":quickEval")    
        if "from_epe" in model_path:
            input_dir_1 = f"{project_path}/data/Baselines/{model_path}/fake_frames_B"
        else:
            input_dir_1 = f"{project_path}/data/Baselines/{model_path}/frames"
        txt_name = model_path.replace("/", "_")
    else:
        logger.val(f"Evaluating {model_name} repetition {repetition} on {subset} set of {dataset_A_name}...", ":quickEval")
        path = f"{project_path}/experimentResults/{model_name}/{dataset_A_name}/repeatTrainingStep_{repetition}/inference/{subset}"
        input_dir_1 = f"{path}/fake_frames_B"
        txt_name = f"{model_name}_{repetition}_{subset}_{dataset_A_name}"

    input_dir_2 = f"{project_path}/data/{dataset_B_name}/sequences/val/frames"

    metrics_dict = calculate_metrics(input1=input_dir_1, input2=input_dir_2, cuda=True, isc=True, fid=True, kid=True, verbose=False, samples_find_deep=True)
    logger.val(str(metrics_dict), ":quickEval")
    logger.val(f"Completed Evaluation.", ":quickEval")

    if not os.path.exists('results'):
        os.makedirs('results')

    with open(f'results/quickEval_results_{txt_name}.txt', 'w') as f:
        f.write(str(metrics_dict))

if __name__ == "__main__":
    args = parseArguments()
    model_name = args.model_name
    dataset_A_name = args.dataset_A_name
    dataset_B_name = args.dataset_B_name
    subset = args.subset
    repetition = args.repetition
    model_path = args.model_path
    distUtils.setRandomSeed(42, by_rank=True)
    quickEval(model_name, dataset_A_name, dataset_B_name, subset, repetition, model_path)
