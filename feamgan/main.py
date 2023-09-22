#Copyright (c) <2023> <Bonifaz Stuhr>

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

"""
Entry File, which contains the main method.
"""
import os
import sys
import time
import multiprocessing
import math
import argparse
import torch

from pathlib import Path

from feamgan.Controller_Component.Controller import Controller
from feamgan.ConfigInput_Component.ConfigProvider import ConfigProvider

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_schedule_path", type=str, help="the relative path to the configuration of the experiments. This experiments will be executed with the defined schedule. (default: feamgan/experimentSchedule.json)",
                        nargs='?', default="feamgan/experimentSchedule.json", const="feamgan/experimentSchedule.json")
    parser.add_argument("--controller_config_path", type=str, help="the relative path to the configuration of the controller (includes hardware specification) (default: feamgan/controllerConfig.json)",
                        nargs='?', default="feamgan/controllerConfig.json", const="feamgan/controllerConfig.json")
    parser.add_argument("--local_rank", type=int, help="the rank of the current proccess (on a single machine with 4 gpus 4 processes with rank 0-3 will be started) (default: 0)",
                        nargs='?', default=0)
    args = parser.parse_args()
    return args

def get_cpu_quota_within_docker():
    cpu_cores = None

    cfs_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    cfs_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")

    if cfs_period.exists() and cfs_quota.exists():
        # we are in a linux container with cpu quotas!
        with cfs_period.open('rb') as p, cfs_quota.open('rb') as q:
            p, q = int(p.read()), int(q.read())

            # get the cores allocated by dividing the quota
            # in microseconds by the period in microseconds
            cpu_cores = math.ceil(q / p) if q > 0 and p > 0 else None

    return cpu_cores

def main(controller_config_path, experiment_schedule_path, local_rank):
    """
    Main method which initialises and starts the execution via the controller.
    The type of the execution specified in the controllerConfig.

    This function prints information about soft- and hardware as well.

    :param controller_config_path: (String) The relative path to the controller configuration file.
    :param experiment_schedule_path: (String) The relative path to the experiment schedule file.
    :param local_rank: (Integer) The the local rank of the current proccess for distributed training.
    """
    ###### Print Information ######
    cpu_cores = get_cpu_quota_within_docker() or multiprocessing.cpu_count()
    if local_rank == 0:
        print('__Python VERSION:', sys.version)
        print('__pyTorch VERSION:', torch.__version__)
        print('__CUDA Available:', torch.cuda.is_available())
        print('__CUDA Initialized:', torch.cuda.is_initialized())
        print('__CUDA VERSION:', torch.version.cuda)
        print('__CUDNN VERSION:', torch.backends.cudnn.version())

        print('__Number CUDA Devices:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"___CUDA Device {i} Name: ", torch.cuda.get_device_name(i))
        print('__Current CUDA Device: ', torch.cuda.current_device())

        print("__Num CPU-Cores Available: ", cpu_cores)

    ###### Initialisation ######    
    print("Main: Starting initialisation ...")
    start_initialisation_time = time.time()
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
 
    config_provider = ConfigProvider()
    controller_config = config_provider.get_config(controller_config_path)
    controller_config["localRank"] = local_rank
    controller_config["distributed"] = distributed
    controller_config["hardware"] = {}
    controller_config["hardware"]["numCPUCores"] = cpu_cores
    controller_config["hardware"]["numGPUs"] = torch.cuda.device_count()

    experiment_schedule_path = config_provider.get_config(experiment_schedule_path)
    controller = Controller(controller_config, experiment_schedule_path)
    initialisation_ok = controller.init()
    end_initialisation_time = time.time()
    print("#########FINISHED INITIALISATION##########")
    print("Initialisation successful: ", initialisation_ok)
    print("Time for initialisation: ", end_initialisation_time-start_initialisation_time, "s")
    print("##########################################")

    ###### Execution ######
    print("Main: Starting execution ...")
    start_execution_time = time.time()
    execution_ok = controller.execute()
    end_execution_time = time.time()
    print("############FINISHED EXECUTION############")
    print("Execution successful: ", execution_ok)
    print("Time for execution: ", end_execution_time-start_execution_time, "s")
    print("##########################################")  

if __name__ == "__main__":
    args = parseArguments()
    controller_config_path = args.controller_config_path
    experiment_schedule_path = args.experiment_schedule_path
    local_rank = args.local_rank
    main(controller_config_path, experiment_schedule_path, local_rank)
