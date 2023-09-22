import argparse
import os
import torch

from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import nvidia.dali.types as types
from nvidia.dali.types import DALIImageType

from feamgan.Input_Component.DataPipelines.daliPipelines import dali_sequence_pipeline
from feamgan.Input_Component.inputUtils import createPipelines
from feamgan.utils import distUtils

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler
from feamgan.Experiment_Component.Metrics.SkvdMetric import SkvdMetric
from feamgan.Experiment_Component.Metrics.CskidMetric import CskidMetric
from feamgan.Experiment_Component.Metrics.CkvdMetric import CkvdMetric

#CSKID: 5000 samples
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="the name of the model to evaluate (default: 'model')",
                        nargs='?', default="FeaMGAN_PFD_to_CS_Crop352_Full_r1", const="FeaMGAN_PFD_to_CS_Crop352_Full_r1")
    parser.add_argument("--dataset_A_name", type=str, help="the name of dataset_A_name, the dataset of the spurce images (default: pfd)",
                        nargs='?', default="pfd", const="pfd")
    parser.add_argument("--dataset_B_name", type=str, help="the name of dataset_B_name, the dataset of the target images (default: Cityscapes)",
                        nargs='?', default="Cityscapes", const="Cityscapes")
    parser.add_argument("--subset", type=str, help="the subset to evalaute (default: eval)",
                        nargs='?', default="eval", const="eval")
    parser.add_argument("--repetition", type=int, help="the training repetition to evalaute (default: 0)",
                        nargs='?', default=0, const=0)
    parser.add_argument("--model_path", type=str, help="if model_path is specified, the metrics will be calculated for the model found in model_path (default: 0)",
                        nargs='?', default="", const="")
    parser.add_argument("--segmentations_path", type=str, help="segmentations_path of mseg_segmentations (default: 0)",
                        nargs='?', default="", const="")         
    parser.add_argument("--batch_size", type=int, help="the batch size to use for metric calculation (default: 1)",
                        nargs='?', default=1, const=1)
    parser.add_argument("--num_threads", type=int, help="the nummer of threads to use for metric calculation  (default: 255)",
                        nargs='?', default=255, const=255)
    parser.add_argument("--data_load_height", type=int, help="before metric calculation the images height is rescaled to data_load_height, the aspect ration is kept the same  (default: 526)",
                        nargs='?', default=526, const=526)
    parser.add_argument("--crop_size_h", type=int, help="before metric calculation the images are center cropped to the given cropsize (default: 526)",
                        nargs='?', default=526, const=526)
    parser.add_argument("--crop_size_w", type=int, help="before metric calculation the images are center cropped to the given cropsize (default: 957)",
                        nargs='?', default=957, const=957)
    parser.add_argument("--metric_parts", type=list, help="calculating the metric over the entire dataset is memory consuming, therefore the metric is calculated in parts. (default: ['vgg16_f_ll'])",
                        nargs='?', default=["vgg16_f_ll"], const=["vgg16_f_ll"])
    parser.add_argument("--eval_steps", type=int, help="the evaluation will performed for the specified number of steps, if eval_steps is given. If eval_steps is None the evaluation will be performed for the entire dataset. (default: None)",
                        nargs='?', default=0, const=0)
    parser.add_argument("--every_x_steps", type=int, help="the evaluation will performed every x steps, if every_x_steps is given. If every_x_steps is None the evaluation will be performed for the entire dataset. (default: None)",
                        nargs='?', default=1, const=1)
    parser.add_argument("--shuffel", type=bool, help="if True, the datasets are shuffeld for metric calculation, this is recommended if eval_steps is given. (default: 0)",
                        nargs='?', default=0, const=0)                 
    parser.add_argument("--metric", type=str, help="metric: sKVD or CSKID (default: sKVD)",
                        nargs='?', default="sKVD", const="sKVD")      
    args = parser.parse_args()
    return args


def calculate_metric(input_frames1, input_segmentations1, input_frames2, input_segmentations2, batch_size, num_threads, dataset_name, data_load_height, crop_size, metric_parts, eval_steps, every_x_steps, shuffel, txt_name, metric, logger):         
    results = {}
    for metric_part in metric_parts:
        logger.val(f"Starting to calculate part {metric_part} ...", ":metricEval")
        dataset_dirs = [input_frames1, input_segmentations1, input_frames2, input_segmentations2]  
        data_types = ["frames", "segmentations_mseg", "frames", "segmentations_mseg"]

        steps = [1,1,1,1]
        strides = [1,1,1,1]
        mode = "val"
        paired_to_prev = []
        augmentation_configs = []
        for data_type in data_types:
            aug_conf = { 
                "preprocessMode": "scale_height_crop",
                "dataLoadSize": data_load_height,
                "cropSize": crop_size,  
                "resize": True,
                "flip": 0
                }
            if data_type != "frames":
                aug_conf["method"] = types.INTERP_NN
                aug_conf["normalize"] = False
                aug_conf["resize"] = True 
                aug_conf["daliImageType"] = DALIImageType.GRAY 
            else:
                aug_conf["method"] = types.INTERP_CUBIC
                aug_conf["normalize"] = True
                aug_conf["resize"] = True
                aug_conf["daliImageType"] = DALIImageType.RGB 
        
            augmentation_configs.append(aug_conf)
            paired_to_prev.append(True)

        pipe_mode = "train" if shuffel else "val" # mode "train" shuffels the datasets
        pipes = createPipelines(dataset_dirs, augmentation_configs, pipe_mode, dali_sequence_pipeline, batch_size, 1, 0, 0, 1, num_threads, True, 42, steps, strides, paired_to_prev)
        for pipe in pipes: pipe.build()
        dataset = {mode: [DALIGenericIterator(pipe, [data_type], reader_name=f"{mode}_reader", last_batch_policy=LastBatchPolicy.PARTIAL) for pipe, data_type in zip(pipes, data_types)]} 

        eval_iter_lens = [it._size // batch_size for it in dataset[mode]] 
        iter_len = eval_iter_lens[0]

        if metric == "sKVD":
            eval_metric = SkvdMetric(is_video=False, model_dir="",  dataset_name=dataset_name, dis_model_name=metric_part)
            logger.val(f"Calculating {metric} ...", ":metricEval")
        elif metric == "CSKID":
            eval_metric = CskidMetric(is_video=False, model_dir="",  dataset_name=dataset_name, dis_model_name=metric_part)
            logger.val(f"Calculating {metric} ...", ":metricEval")
        elif metric == "cKVD":
            eval_metric = CkvdMetric(is_video=False, model_dir="",  dataset_name=dataset_name, dis_model_name=metric_part)
            logger.val(f"Calculating {metric} ...", ":metricEval")
        else:
            raise ValueError(f'Metric {metric} not defined!')

        step = 0
        print(eval_iter_lens)
        def getNextIteratorData(dataset, mode):
            data = []
            for it in dataset[mode]:
                try:
                    d=it.next()
                except StopIteration: # Some iterator reset before the main interator [0], therefore we want to continue evaluation 
                    it.reset()
                    print(f"Iterator reseted before main interator as expected, continue {mode} in this epoch", "calculate_skvd:getNextIteratorData")
                    d=it.next()
                data.append(d[0])
            return data

        performed_evaluations = 0
        while iter_len > step: 

            data = getNextIteratorData(dataset,mode)
            if not step % every_x_steps:
                for k in range(len(data)):
                    data[k] = distUtils.dictionaryToCuda(data[k])
                with torch.no_grad():
                    eval_metric.forwardBatch([data[0]["frames"].squeeze(0), data[1]["segmentations_mseg"].squeeze(0)], [data[2]["frames"].squeeze(0), data[3]["segmentations_mseg"].squeeze(0)], mode)
                performed_evaluations += 1
                
            step+=1
            if eval_steps:
                if eval_steps <= step:
                    break
          
            torch.cuda.synchronize()

        logger.val(f"Performed {performed_evaluations} evaluation steps, reducing batches ...:", ":metricEval") 

        reduced_metric_values = eval_metric.reduceBatches(mode)
        logger.val(f"Results for part {metric_part}:", ":metricEval")
        logger.val(str(reduced_metric_values), ":metricEval")
        results[metric_part] = reduced_metric_values 
    
        logger.val("Results:", ":metricEval")
        logger.val(str(results[metric_part]), ":metricEval")
        logger.val(f"Completed Evaluation.", ":metricEval")

        if not os.path.exists('results'):
            os.makedirs('results')

        with open(f'results/{metric}_results_{txt_name}_{metric_part}.txt', 'w') as f:
            f.write(str(results[metric_part]))

    return results

  

def metricEval(model_name, dataset_A_name, dataset_B_name, subset, repetition, model_path, segmentations_path, batch_size, num_threads, data_load_height, crop_size, metric_parts, eval_steps, every_x_steps, shuffel, metric):
    logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)
   
    os.environ['TORCH_HOME'] = 'models'
    project_path = os.path.dirname(os.path.abspath("feamgan"))

    if model_path:    
        logger.val(f"Evaluating {model_path} ...", ":metricEval")
        if "from_epe" in model_path:
            input_dir_1 = f"{project_path}/data/Baselines/{model_path}/frames"
            input_segmentations_dir_1 = f"{project_path}/data/Baselines/{model_path}/segmentations_mseg"     
        else:
            input_dir_1 = f"{project_path}/data/Baselines/{model_path}/fake_frames_B"
            input_segmentations_dir_1 = f"{project_path}/data/Baselines/{segmentations_path}/real_segmentations_mseg_A"
        txt_name = model_path.replace("/", "_")
    else:
        logger.val(f"Evaluating {model_name} repetition {repetition} on {subset} set of {dataset_A_name}...", ":metricEval")
        path = f"{project_path}/experimentResults/{model_name}/{dataset_A_name}/repeatTrainingStep_{repetition}/inference/{subset}"
        input_dir_1 = f"{path}/fake_frames_B"
        input_segmentations_dir_1 = f"{path}/real_segmentations_mseg_A"
        txt_name = f"{model_name}_{repetition}_{subset}_{dataset_A_name}"

    input_dir_2 = f"{project_path}/data/{dataset_B_name}/sequences/val/frames"
    input_segmentations_dir_2 = f"{project_path}/data/{dataset_B_name}/sequences/val/segmentations_mseg"
    
    dataset_name = f"{dataset_A_name}_{dataset_A_name}"
    calculate_metric(input_dir_1, input_segmentations_dir_1, input_dir_2, input_segmentations_dir_2, batch_size, num_threads, dataset_name, data_load_height, crop_size, metric_parts, eval_steps, every_x_steps, shuffel, txt_name, metric, logger)

if __name__ == "__main__":
    args = parseArguments()
    model_name = args.model_name
    dataset_A_name = args.dataset_A_name
    dataset_B_name = args.dataset_B_name
    subset = args.subset
    repetition = args.repetition
    model_path = args.model_path
    segmentations_path = args.segmentations_path
    batch_size = args.batch_size
    num_threads = args.num_threads
    data_load_height = args.data_load_height
    crop_size = [args.crop_size_h, args.crop_size_w]
    print(crop_size)
    metric_parts = args.metric_parts
    eval_steps = args.eval_steps
    every_x_steps = args.every_x_steps
    shuffel = args.shuffel
    metric = args.metric
    distUtils.setRandomSeed(42, by_rank=True)
    metricEval(model_name, dataset_A_name, dataset_B_name, subset, repetition, model_path, segmentations_path, batch_size, num_threads, data_load_height, crop_size, metric_parts, eval_steps, every_x_steps, shuffel, metric)
