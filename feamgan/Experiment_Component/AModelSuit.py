import os
import time
import wandb
import torch
import glob

import numpy as np

from PIL import Image
from apex import amp
from pathlib import Path
from abc import ABCMeta

from feamgan.utils import distUtils
from feamgan.utils.AverageMeter import AverageMeter
from feamgan.utils.visualizationUtils import formatFramesToUnit, convertTime
from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler
from feamgan.Experiment_Component.Models.utils import modelUtils


class AModelSuit(metaclass=ABCMeta):
    """
    A AModelSuit handles the train/eval/test/inference of a model. Therefore it brings the input, the model and
    the trainer, together in one place. In each AModelSuit functions for the training and validation must be defined.

    The AModelSuit provides basic functionality like model saving and defines interface methods for ModelSuits.

    :Attributes:
        _model:                         ("Model") The model to handle with the ModelSuit.
        _dataset:                       (Dictionary) The dataset to train/eval/test the model.
        _trainer:                       (ITrainer) The trainer to train the model.
        _batch_size:                    (Integer) The batch size for the model.
        _num_gpus:                      (Integer) The number of used GPUs.
        _seed:                          (Integer) The random seed to use.
        _local_rank:                    (Integer) The local_rank (the number of the GPU of the process).
        _verbose:                       (Boolean) If true, logs are more detailed. "True" by default. 
        _model_name                     (String) The name of the model, the suit holds.
        _logger:                        (Logger) The logger for the ModelSuit.
        _model_dir:                     (String) The directory of the model (e.g. to save it).
        _state:                         (Dictionary) State containing all the running variables for training.
        _state_model:                   (Dictionary) State containing all the running variables for the model.
        _save_checkpoint_steps:         (Integer) Every save_checkpoint_steps steps the ModelSuit saves model
                                            (training) checkpoints. 500 by default. (Optional) set to -1 if not needed.
        _save_checkpoint_epochs:        (Integer) Every save_checkpoint_epochs epochs the ModelSuit saves model
                                            (training) checkpoints. 1 by default. (Optional) set to -1 if not needed. 
                                            List of epochs supported (e.g. [1,5] saves only a checkpoint in the first and fifth epoch)
        _log_steps:                     (Integer) Every log_steps steps the ModelSuit writes logs. 100 by default. (Optional) set to -1 if not needed.
        _log_epochs:                    (Integer) Every log_epoch epochs the ModelSuit writes logs. 1 by default. (Optional) set to -1 if not needed.
        _save_summary_steps:            (Integer) Every save_summary_steps steps the ModelSuit saves summaries. 250 by default. 
                                        (Optional) set to -1 if not needed.
        _save_summary_epochs:           (Integer) Every save_summary_epoch epochs the ModelSuit saves summaries. 1 by default. 
                                        (Optional) set to -1 if not needed.
        _save_checkpoint_steps:         (Integer) Every _save_checkpoint_steps steps the ModelSuit saves checkpoints. 500 by default. 
                                        (Optional) set to -1 if not needed.
        _save_checkpoint_epochs:        (Integer) Every _save_checkpoint_epochs epochs the ModelSuit saves checkpoints. 1 by default. 
                                        (Optional) set to -1 if not needed.
    """

    def __init__(self, model, trainer, dataset, batch_size, num_gpus, model_config, local_rank, seed, model_dir="/model", save_checkpoint_steps=500, save_checkpoint_epochs=1,
                 log_steps=100, log_epochs=1, save_summary_steps=250, save_summary_epochs=1, load_checkpoint="latest", verbose=True, use_wandb=True, dict_keys=None):
        """
        Constructor, initialize member variables.
        :param model: ("Model") The model to handle with the ModelSuit
        :param trainer: (ITrainer) The trainer to train the model.
        :param dataset: (Dictionary) The dataset to train/eval/test the model.
        :param batch_size: (Integer) The batch size for the model.
        :param num_gpus: (Integer) The number of used GPUs.
        :param model_config: (Dictionary) The configuration of the model, containing layers specifications, learning rates, etc.
        :param local_rank: (Integer) The local_rank (the number of the GPU of the process).
        :param seed: (Integer) The random seed to use.
        :param model_dir: (String) The directory of the model (e.g. to save it). "/model" by default.
        :param save_checkpoint_steps: (Integer) Every save_checkpoint_steps steps the ModelSuit saves model
                                        (training) checkpoints. 500 by default. (Optional) set to -1 if not needed.
        :param save_checkpoint_epochs: (Integer) Every save_checkpoint_epochs epochs the ModelSuit saves model
                                        (training) checkpoints. 1 by default. (Optional) set to -1 if not needed. 
                                        List of epochs supported (e.g. [1,5] saves only a checkpoint in the first and fifth epoch)
        :param log_steps: (Integer) Every log_steps steps the ModelSuit writes logs. 100 by default. (Optional) set to -1 if not needed.
        :param log_epochs: (Integer) Every log_epoch epochs the ModelSuit writes logs. 1 by default. (Optional) set to -1 if not needed.
        :param save_summary_steps: (Integer) Every save_summary_steps steps the ModelSuit saves Wandb summaries. 250 by default. 
                                    (Optional) set to -1 if not needed.
        :param save_summary_steps: (Integer) Every save_summary_epoch epochs the ModelSuit saves Wandb summaries. 1 by default. 
                                    (Optional) set to -1 if not needed.
        :param load_checkpoint: (Integer) Loads the given model checkpoint. "latest" by default. 
        :param verbose: (Boolean) If true, logs are more detailed. "True" by default. 
        """
        # Set model, optimizer, dataset, trainer, batch_size, seed.
        self._model = model
        self._dataset = dataset
        self._trainer = trainer
        self._batch_size = batch_size
        self._num_gpus = num_gpus 
        self._seed = seed
        self._local_rank = local_rank
        self._verbose = verbose
        self._model_name = model_config["modelName"]
        self._use_wandb = use_wandb

        self._dict_keys = dict_keys

        # Setting up the Loggers
        self._logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)

        # Dir to save and reload model. 
        self._model_dir = f"experimentResults{model_dir}" if "pretrainedModels/" not in model_dir else model_dir
        Path(self._model_dir).mkdir(parents=True, exist_ok=True)

        # Iterator lengths
        if "train" in self._dataset:
            train_iter_lens = [it._size // self._batch_size for it in self._dataset["train"]]
            main_train_iter_len = train_iter_lens[0]
        if "eval" in self._dataset:
            eval_iter_lens = [it._size // self._batch_size for it in self._dataset["eval"]] 
            main_eval_iter_len = eval_iter_lens[0]

        if self._local_rank == 0:
            self._logger.train(f"Training Iterator lengths are: {train_iter_lens}", "AModelSuit:__init__")
            self._logger.val(f"Validation Iterator lengths are: {eval_iter_lens}", "AModelSuit:__init__")

        # State containing all running variables of training.
        self._state={
            "current_step": 0,
            "current_epoch": 0, 
            "best_current_step": 0, 
            "best_current_epoch": 0,
            "training_done": False,
            "training_ended_in_epoch": False,
            "best_model_values": {"eval":{"loss":9999999}}, 
            "best_trainer_values": {"train":None, "eval":None}, 
            "current_model_values": {"train":None, "eval":None},  
            "current_trainer_values": {"train":None, "eval":None}, 
            "model_state_dict": "Not shown",
            "optimizer_state_dict" : "Not shown",
            "avg_load_batch_time": {"train":None, "eval":None},
            "avg_model_batch_time": {"train":None, "eval":None},
            "avg_batch_time": {"train":None, "eval":None},
            "current_time": {"train":None, "eval":None},
            "avg_epoch_time": {"train":None, "eval":None},
            "avg_log_steps_time": {"train":None, "eval":None},
            "overall_time": None,
            "inter_len":{"train":main_train_iter_len, "eval":main_eval_iter_len}
        }
        self._state_model={"model_state_dict": self._model.state_dict()}

        # Logging times
        self._train_model_batch_time_meter = AverageMeter()
        self._eval_model_batch_time_meter = AverageMeter()
        self._train_load_batch_time_meter = AverageMeter()
        self._eval_load_batch_time_meter = AverageMeter()
        self._train_batch_time_meter = AverageMeter()
        self._eval_batch_time_meter = AverageMeter()
        self._train_epoch_time_meter = AverageMeter()
        self._eval_epoch_time_meter = AverageMeter()
        self._train_log_steps_time_meter = AverageMeter()
        self._eval_log_steps_time_meter = AverageMeter()

        # Log every log_interval_steps and/or _epochs
        self._log_steps = log_steps
        self._log_epochs = log_epochs

        # Save summary every save_summary_steps and/or _epochs
        self._save_summary_steps = save_summary_steps
        self._save_summary_epochs = save_summary_epochs

        # Save checkpoints every save_checkpoints_steps and/or _epochs
        self._save_checkpoint_steps = save_checkpoint_steps
        self._save_checkpoint_epochs = save_checkpoint_epochs

        # Init the model
        self._delay_allreduce = model_config["pipeline"]["delayAllreduce"]
        self._amp_opt_level = model_config["pipeline"]["ampOptLevel"]
        self._initModel()

        # Load specified checkpoint if needed, else continue training if model exists
        self.first_checkpoint = not self._loadCheckpoint(load_checkpoint)

        # To save summary.
        if (self._local_rank == 0) and self._use_wandb:
            #self._summary_txt_writer = TxtSummaryWriter(self._model_dir)
            wandb.watch(self._model, log="all")

    def _initModel(self):
        self._model.cuda()
        if self._model.is_train:
            self._model.initWeights()
        self._model.setModelDir(self._model_dir)
        # Different GPU copies of the same model will receive noises
        # initialized with different random seeds (if applicable) thanks to the
        # set_random_seed command (GPU #K has random seed = args.seed + K).
        distUtils.setRandomSeed(self._seed, by_rank=True)
        if self._model.is_train:
            self._model.initOptimization()
            self._model, self._model.optimizers = amp.initialize(self._model, self._model.optimizers, 
                opt_level=self._amp_opt_level, num_losses=len(self._model.optimizers)) 

        if self._num_gpus > 1:
            self._model = torch.nn.parallel.DistributedDataParallel(self._model,
                                                  device_ids=[self._local_rank],
                                                  output_device=self._local_rank,
                                                  find_unused_parameters=True)
        else:
            self._model = modelUtils.WrappedModel(self._model)

        if (self._local_rank == 0) and self._verbose:
            self.logModelGraph() 
 
    def logModelGraph(self):
        self._model.module.printModel()

    def updateStateAndSummary(self, prev_data, gen_pred, metrics_values, mode):
        if self._state["current_model_values"][mode]:
            if metrics_values:
                self._state["current_model_values"][mode].update(metrics_values)
        else: 
            self._state["current_model_values"][mode] = metrics_values
        if self._local_rank == 0:
            self._saveSummary(prev_data, gen_pred, mode)    

    def _getNextIteratorData(self, mode):
        data = []
        for it in self._dataset[mode]:
            try:
                d=it.next()
            except StopIteration: # Some iterator reset before the main interator [0], therefore we want to continue evaluation 
                it.reset()
                if self._local_rank == 0:
                    self._logger.info(f"Iterator reseted before main interator as expected, continue {mode} in this epoch", "AModelSuit:doValidation")
                d=it.next()
            data.append(d[0])

        # Compatiblity with old pipeline
        if len(data[0])>1:
            data = [{k.replace("_B", ""):data[0][k]} for k in self._dict_keys]
        return data

    def resetIterators(self, mode, dry_run_unfinished_iterators=True):
        main_iter_len = self._state["inter_len"][mode]
        for i, it in enumerate(self._dataset[mode]):
            # since DALI does not support iterator reset when the iterator has not loopt through the entrie dataset we need to dry run the unfinished iterators!
            if (it._size // self._batch_size != main_iter_len):
                if dry_run_unfinished_iterators:  # For validation we reset the iterators at the end of validation to keep the val set the same
                    if self._local_rank == 0:
                        self._logger.info(f"Main iterator length is {main_iter_len}. Dry run iterator {i} with incompatible length {it._size// self._batch_size}", "AModelSuit:resetIterators")
                    for d in it: continue
                    it.reset()
                    if self._local_rank == 0:
                        self._logger.info(f"Reseted iterator {i} with incompatible length {it._size// self._batch_size}", "AModelSuit:resetIterators")
                else: # For training we reset the main iterator at the endo of a epoch and the incompatible iterator when it is finished
                    if self._local_rank == 0:
                        self._logger.info(f"Iterator {i} with length {it._size// self._batch_size} not finished, ignoring reset", "AModelSuit:resetIterators") 
            else:
                it.reset()
                if self._local_rank == 0:
                    self._logger.info(f"Reseted iterator {i} with compatible length {it._size// self._batch_size}", "AModelSuit:resetIterators")

    def _train_epoch(self, train_steps, eval_steps, end_train_log_time):
        
        # Start timers TODO Make times compatible with multi gpu training       
        eval_time_subtract_meter = AverageMeter()

        ce = self._state["current_epoch"]
        self._trainer.startOfEpoch(ce, self._model)
        train_epoch_step = 0
        data = None
        gen_pred = None
        inter_len = self._state["inter_len"]["train"]
        while inter_len > train_epoch_step: 
            end_train_batch_time = time.time()

            end_train_load_batch_time = time.time()
            data = self._getNextIteratorData("train")
            self._train_load_batch_time_meter.update(time.time() - end_train_load_batch_time) 

            # Train 
            cs = self._state["current_step"]   
            data = self._trainer.startOfStep(data, cs, self._model)

            end_train_model_batch_time = time.time()
            for _ in range(self._trainer.getGeneratorTrainingSteps()):
                gen_losses, gen_pred = self._trainer.generator(data, self._model)
            for _ in range(self._trainer.getDiscriminatorTrainingSteps()):
                dis_losses, dis_pred = self._trainer.discriminator(data, gen_pred, self._model)
            self._train_model_batch_time_meter.update(time.time() - end_train_model_batch_time) 

            self._state["current_model_values"]["train"] = {"generator_losses":gen_losses, "discriminator_losses": dis_losses} 
        
            train_epoch_step+=1
            cs +=1
            self._state["current_step"] = cs
            data = self._trainer.endOfStep(data, ce, cs, self._model)
            eval_time_subtract = 0

            # If evaluation of steps is wished and if eval_steps steps past, do validation.
            if (eval_steps > 0) and (cs % eval_steps == 0):
                self.doValidation("eval")
                self._trainer.train()
                self._model.train()
                eval_time_subtract = self._eval_epoch_time_meter.val
                eval_time_subtract_meter.update(eval_time_subtract)

            # If checkpoint should be saved, save checkpoint every save_checkpoint_steps iterations 
            if (self._local_rank == 0) and (self._save_checkpoint_steps > 0) and (cs % self._save_checkpoint_steps == 0):
                self._saveCheckpoint()

            # If a summary should be saved and save_summary_steps steps past, save the summary.
            if (self._save_summary_steps > 0) and (cs % self._save_summary_steps == 0):
                metrics_values = self._model.module.updateMetrics(data, gen_pred, "train")
                metrics_values = self._model.module.reduceMetrics([metrics_values], "train", self._num_gpus)
                torch.cuda.synchronize()
                self.updateStateAndSummary(data, gen_pred, metrics_values, "train")
    
            # If log_steps should be saved and log_steps steps past, print the logs.
            if (self._log_steps > 0) and (cs % self._log_steps == 0):
                self._train_log_steps_time_meter.update(time.time() - end_train_log_time)
                end_train_log_time = time.time()
                self._updateTimes()
                if self._local_rank == 0:
                    self._logger.train(f"Step {cs} of {train_steps}. Epoch step {train_epoch_step} of {inter_len}. Name: {self._model_name}", "AModelSuit:_train")

            # Capture the batch time
            self._train_batch_time_meter.update(time.time() - end_train_batch_time-eval_time_subtract) 
            self._updateTimes()

            # Check if we at the end of training.
            if train_steps >= 0:
                if cs >= train_steps:
                    self._logger.train(f"Finished training at step {cs} in epoch {ce} for {self._model_name}","AModelSuit:_train") 
                    self._state["training_done"] = True
                    break

        cs = self._state["current_step"]
        self._trainer.endOfEpoch(data, ce+1, cs, self._model)
        self.resetIterators("train", dry_run_unfinished_iterators=False)

        return eval_time_subtract_meter.sum, data, gen_pred

    def doTraining(self, train_steps, eval_steps, train_epochs=-1, eval_epochs=-1):
        """
        Trains the model with the trainer and the input of the ModelSuit.
        :param train_steps: (Integer) The steps to train the model. (Optional) set to -1 if not needed.
        :param eval_steps: (Integer) Every eval_steps steps the Model will be evaluated. (Optional) set to -1 if not needed.
        :param train_epochs: (Integer) The epochs to train the model. (Optional) set to -1 if not needed. -1 by default.
        :param eval_epochs: (Integer) Every eval_epochs epochs the Model will be evaluated. (Optional) set to -1 if not needed. -1 by default.
        """
        if self._local_rank == 0:
            self._logger.train(f"\nStarted training for {train_steps} steps or {train_epochs} epochs for {self._model_name}.\nEvaluation every {eval_steps} steps and/or {eval_epochs} epochs.", "AModelSuit:doTraining")

        # Check if the model is already trained for the given steps or epochs
        if train_steps >= 0:
            if self._state["current_step"] >= train_steps:
                return
        elif train_epochs >= 0:
            if self._state["current_epoch"] >= train_epochs:
                return
        self._state["training_done"] = False

        # Start timers         
        end_epoch_train_time = time.time()
        end_train_log_time = time.time()     
 
        # Log, save ... the untrained model
        consider = {"validation":False, "checkpoint": False, "summary":True, "log":True}
        first_run = True
        if (self._state["current_epoch"] > 0) or (self._state["current_step"] > 0):
            consider = {"validation":True, "checkpoint": True, "summary":True, "log":True}
            first_run = False
       
        self._model.module.initTraining() 
        self._trainer.train()
        self._model.train()
        while not self._state["training_done"]: 
            ce = self._state["current_epoch"]
            eval_time_subtract = 0

            # Train for one epoch
            if first_run:
                data = None 
                pred = None
            else:
                passed_eval_time, data, pred = self._train_epoch(train_steps, eval_steps, end_train_log_time)
                eval_time_subtract += passed_eval_time
                ce += 1
                self._state["current_epoch"] = ce

            # Check if at the end of training.
            if train_epochs >= 0:
                if ce >= train_epochs:
                    self._state["training_done"] = True

            # Evaluate on evaluation set
            if (eval_epochs > 0 and ce % eval_epochs == 0 and consider["validation"]) or (self._state["training_done"] and eval_epochs > 0):
                self.doValidation("eval")
                self._trainer.train() 
                self._model.train()
                eval_time_subtract = self._eval_epoch_time_meter.val
                            
            # If checkpoint should be saved, save checkpoint every save_checkpoint_epochs iterations 
            if self._local_rank == 0:
                if (self._save_checkpoint_epochs > 0 and ce % self._save_checkpoint_epochs == 0 and consider["checkpoint"]) or self._state["training_done"]:
                    self._saveCheckpoint()

            # If a summary should be saved and save_summary_epochs epochs past, save the summary.
            if (self._save_summary_epochs > 0 and ce % self._save_summary_epochs == 0 and consider["summary"]) or self._state["training_done"]:
                metrics_values = self._model.module.updateMetrics(data, pred, "train")
                metrics_values = self._model.module.reduceMetrics([metrics_values], "train", self._num_gpus)
                torch.cuda.synchronize()
                self.updateStateAndSummary(data, pred, metrics_values, "train")
  
            # If log_epochs should be saved and log_epochs epochs past, print the logs.
            if (self._log_epochs > 0 and ce % self._log_epochs == 0 and consider["log"]) or self._state["training_done"]:
                if ce > 0:
                    self._train_log_steps_time_meter.update(time.time() - end_train_log_time)
                    end_train_log_time = time.time()
                    self._updateTimes()
                if self._local_rank == 0:
                    #self._logger.train(f"Epoch {ce} of {train_epochs}:\nName: {self._model_name}\nStats:\n{pprint.pformat(self._state)}", "AModelSuit:doTraining")
                    self._logger.train(f"Epoch {ce} of {train_epochs}. Name: {self._model_name}", "AModelSuit:doTraining")

            if first_run:
                consider = {"validation":True, "checkpoint": True, "summary":True, "log":True}
                first_run = False
            
            # Capture the epoch time
            self._train_epoch_time_meter.update(time.time() - end_epoch_train_time - eval_time_subtract) 
            end_epoch_train_time = time.time()
            self._updateTimes()
   
        if self._local_rank == 0:
            self._logger.train(f"Finished training for {train_epochs} epochs or {train_steps} steps for {self._model_name}. Evaluation was every {eval_steps} steps and/or {eval_epochs} epochs.", "AModelSuit:doTraining")
           
        if (eval_epochs > 0) or (eval_steps > 0):
            # Run extended, more expensive validation at the end.
            self._logger.train(f"Starting extended evaluation...", "AModelSuit:doTraining")
            self.doValidation("eval", extended_validation=True)

    def doValidation(self, mode, extended_validation=False):
        """
        Validates the model on the subdataset subset defined by the mode.
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        """
        if self._local_rank == 0:
            self._logger.val(f"Started validation for {mode} dataset.", "AModelSuit:doValidation")
        
        self._trainer.val() 
        self._model.train() # We set the model to train to use the statistics of the current input
    
        # Start timers   
        end_eval_epoch_time = time.time()
        end_eval_log_time = time.time()

        self._trainer.startOfEpoch(None, self._model)

        val_step = 0 
        data = None
        gen_pred = None
        overall_metrics_values = []
        inter_len = self._state["inter_len"][mode]
        fake_B_batch = []
        while inter_len > val_step: 
            end_eval_batch_time = time.time()
            end_eval_load_batch_time = time.time()
            data = self._getNextIteratorData(mode)
            self._eval_load_batch_time_meter.update(time.time() - end_eval_load_batch_time) 

            with torch.no_grad():
                data = self._trainer.startOfStep(data, None, self._model)
                end_eval_model_batch_time = time.time()  
                gen_pred = self._model.module.val(data)
                if len(fake_B_batch)<15:
                    fake_B_batch.append(gen_pred["fake_B_batch"])

                self._eval_model_batch_time_meter.update(time.time() - end_eval_model_batch_time)
                val_step += 1
                data = self._trainer.endOfStep(data, None, None, self._model)

            torch.cuda.synchronize()
            overall_metrics_values.append(self._model.module.updateMetrics(data, gen_pred, mode, extended_validation))

            # If log_steps should be saved and log_steps steps past, print the logs.
            if (self._log_steps >= 0) and (val_step % self._log_steps == 0):
                self._state["current_trainer_values"][mode] = None 
                self._eval_log_steps_time_meter.update(time.time() - end_eval_log_time)
                end_eval_log_time = time.time()
                self._updateTimes()
                if self._local_rank == 0:
                    self._logger.val(f"Validation Step {val_step} of {inter_len} for mode {mode}. Name: {self._model_name}", "AModelSuit:_val")

            # Capture the batch time
            self._eval_batch_time_meter.update(time.time() - end_eval_batch_time)
            self._updateTimes() 
        
        self._trainer.endOfEpoch(data, None, None, self._model)
        self.resetIterators(mode)

        # Calculate avarage values of the evaluation
        mean_eval_model_values = self._model.module.reduceMetrics(overall_metrics_values, mode, self._num_gpus, extended_validation)

        # Set final evaluation values of this evaluation cycle
        self._state["current_model_values"][mode] = mean_eval_model_values
        self._state["current_trainer_values"][mode] = None

        # Capture the evaluation time
        self._eval_epoch_time_meter.update(time.time() - end_eval_epoch_time)
        self._updateTimes()

        # Save summary at the end of the validation 
        if self._local_rank == 0:
            gen_pred["fake_B_batch"] = fake_B_batch
            self._saveSummary(data, gen_pred, mode)
            self._logger.val(f"Finished validation for {mode} dataset. Name: {self._model_name}", "AModelSuit:doValidation")

    def doInference(self, mode):
        """
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        """
        if self._local_rank == 0:
            self._logger.val(f"Started inference for {mode} dataset.", "AModelSuit:doInference")
        
        self._trainer.val() 
        self._model.train() # We set the model to train to use the statistics of the current input2

        # Len of the dataset iterator
        eval_iter_lens = [it._size // self._batch_size for it in self._dataset[mode]]
        self._logger.val(f"Validation Iterator lenghts are: {eval_iter_lens}", "AModelSuit:doInference")
        inter_len = eval_iter_lens[0]

        # Start timers   
        end_infer_epoch_time = time.time()
        infer_gen_batch_time_meter = AverageMeter()

        self._trainer.startOfEpoch(None, self._model)

        infer_step = 0    
        data = None
        while inter_len > infer_step: # Some iterator reset before the main interator [0], therefore we want to continue evaluation 

            data = self._getNextIteratorData(mode)

            with torch.no_grad():
                data = self._trainer.startOfStep(data, None, self._model)
                end_infer_gen_batch_time = time.time()
                gen_pred = self._model.module.inference(data)
                infer_gen_batch_time_meter.update(time.time() - end_infer_gen_batch_time)
                infer_step += 1
                data = self._trainer.endOfStep(data, None, None, self._model)

            image_dictionary = self._model.module.getImages(data, gen_pred, mode)
            self.saveImages(image_dictionary, infer_step, self._local_rank, mode)

            # If log_steps should be saved and log_steps steps past, print the logs.
            if (self._log_steps >= 0) and (infer_step % self._log_steps == 0):
                if self._local_rank == 0: 
                    self._logger.val(f"Inference Step {infer_step} of {inter_len} for mode {mode}.\nName: {self._model_name}\nAvg batch time: {convertTime(infer_gen_batch_time_meter.avg)}.", "AModelSuit:_infer") 

        self._trainer.endOfEpoch(data, None, None, self._model)

        # Capture the evaluation time
        infer_time = time.time() - end_infer_epoch_time

        if self._local_rank == 0:
            self._logger.val(f"Finished inferece for {mode} dataset.\nAvg batch time: {convertTime(infer_gen_batch_time_meter.avg)}.\nInference time: {convertTime(infer_time)}", "AModelSuit:doInference")

    def _saveCheckpoint(self): 
        ce = self._state["current_epoch"]
        cs = self._state["current_step"]

        latest_checkpoint_name = f"epoch_{ce}_step_{cs}_checkpoint.pt"
        latest_checkpoint_path = os.path.join(self._model_dir, "checkpoints", latest_checkpoint_name)
        self._state["model_state_dict"] = self._model.state_dict()
        self._state["optimizer_state_dict"] = self._model.module.getOptimizerStateDict()
        self._state["amp_state_dict"] = amp.state_dict()

        cp = {
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._model.module.getOptimizerStateDict(),
            "amp_state_dict": amp.state_dict(),
            "current_epoch": ce,
            "current_step": cs,
            "best_current_step": self._state["best_current_step"],
            "best_current_epoch": self._state["best_current_epoch"] 

        }
        torch.save(cp, latest_checkpoint_path)
        
        fn = os.path.join(self._model_dir, "checkpoints", 'latest_checkpoint.txt')
        with open(fn, 'wt') as f:
            f.write('latest_checkpoint: %s' % latest_checkpoint_name)
        self._state["model_state_dict"] = "Not shown"
        self._state["optimizer_state_dict"] = "Not shown"
        self._state["amp_state_dict"] = "Not shown"
        self._logger.train(f"\nSaved checkpoint for epoch {ce} step {cs} at {fn}. Name: {self._model_name}","AModelSuit:_saveCheckpoint")
    
    def _loadCheckpoint(self, load_checkpoint):
        restore_checkpoint = None
        if load_checkpoint is not None:
            if load_checkpoint == "latest":
                cp_path = os.path.join(self._model_dir, "checkpoints", 'latest_checkpoint.txt')
                if os.path.exists(cp_path):
                    with open(cp_path, 'r') as f:
                        line = f.read().splitlines()                         
                    restore_checkpoint = line[0].split(' ')[-1]
            else:
                ce = load_checkpoint["current_epoch"]
                cs = load_checkpoint["current_step"]
                restore_checkpoint = f"epoch_{ce}_step_{cs}_checkpoint.pt"
        
        checkpoint_found = False
        if restore_checkpoint:   
            checkpoint_path = os.path.join(self._model_dir, "checkpoints", restore_checkpoint) 
            self._loaded_checkpoint = torch.load(checkpoint_path, map_location = lambda storage, loc: storage.cuda(self._local_rank))
            self._model.load_state_dict(self._loaded_checkpoint['model_state_dict'], strict=False)
            if self._model.module.is_train:
                self._model.module.loadOptimizerStateDict(self._loaded_checkpoint['optimizer_state_dict'])
                amp.load_state_dict(self._loaded_checkpoint['amp_state_dict'])

            self._state = self._loaded_checkpoint
            self._state["model_state_dict"] = "Not shown"
            self._state['optimizer_state_dict'] = "Not shown"
            self._state['amp_state_dict'] = "Not shown"
            if self._local_rank == 0:
                self._logger.info(f"Restored model from {checkpoint_path}", "AModelSuit:loadCheckpoint")
            checkpoint_found = True
        else:
            self._loaded_checkpoint = None
            if self._local_rank == 0:
                self._logger.info("No checkpoint found. Initializing model from scratch", "AModelSuit:loadCheckpoint")
            Path(os.path.join(self._model_dir, "checkpoints")).mkdir(parents=True, exist_ok=True)
        return checkpoint_found

    def getModel(self):
        """
        Returns the model.
        :return: model: ("Model") The model to handle with the ModelSuit
        """
        return self._model

    def _saveSummary(self, data, out_values, mode):
        cs = self._state["current_step"]
        ce = self._state["current_epoch"]
       
        model_summary = self._model.module.getSummary(data, self._state["current_model_values"][mode], out_values, mode)
        trainer_summary = self._trainer.getSummary()
        self._state["current_model_values"][mode] = model_summary
        self._state["current_trainer_values"][mode] = trainer_summary
        
        wandb_log_dict = distUtils.dictionaryRemoveNone(self._state)
        if self._use_wandb:
            wandb.log(wandb_log_dict, step=cs)

        save_msg = f"Saved {mode} summary for epoch {ce} step {cs}."
        if mode == "train":
            self._logger.train(save_msg,"AModelSuit:_saveSummary")
        else:     
            self._logger.val(save_msg,"AModelSuit:_saveSummary")

    def _updateTimes(self):
        # Update time in seconds
        self._state["avg_load_batch_time"]["train"] = self._train_load_batch_time_meter.avg
        self._state["avg_load_batch_time"]["eval"] = self._eval_load_batch_time_meter.avg
        self._state["avg_model_batch_time"]["train"] = self._train_model_batch_time_meter.avg
        self._state["avg_model_batch_time"]["eval"] = self._eval_model_batch_time_meter.avg
        self._state["avg_batch_time"]["train"] = self._train_batch_time_meter.avg
        self._state["avg_batch_time"]["eval"] = self._eval_batch_time_meter.avg
        self._state["avg_epoch_time"]["train"] = self._train_epoch_time_meter.avg
        self._state["avg_epoch_time"]["eval"] = self._eval_epoch_time_meter.avg
        self._state["avg_log_steps_time"]["train"] = self._train_log_steps_time_meter.avg
        self._state["avg_log_steps_time"]["eval"] = self._eval_log_steps_time_meter.avg
        self._state["current_time"]["train"] = self._train_epoch_time_meter.sum
        self._state["current_time"]["eval"] = self._eval_epoch_time_meter.sum
        overall = 0
        if self._state["current_time"]["train"]: 
            overall = overall + self._state["current_time"]["train"]
        if self._state["current_time"]["eval"]: 
            overall = overall + self._state["current_time"]["eval"]
        self._state["overall_time"] = overall

    def saveImages(self, image_dictionary, infer_step, rank, mode):
        for name, images in image_dictionary.items():
            save_dir = f"{self._model_dir}/inference/{mode}/{name}"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for s in range(len(images)):
                seq_save_dir = f"{save_dir}/{str(infer_step).zfill(8)}_{str(rank).zfill(3)}_{str(s).zfill(3)}"
                Path(seq_save_dir).mkdir(parents=True, exist_ok=True)
                for i in range(len(images[s])):
                    # /step_gpu_sequence_frame
                    save_path = f"{seq_save_dir}/{str(infer_step).zfill(8)}_{str(rank).zfill(3)}_{str(s).zfill(3)}_{str(i).zfill(6)}.jpg"      
                    if images[s][i].shape[0] is 3: 
                        image = formatFramesToUnit(images[s][i]).cpu().numpy()
                    else:
                        image = images[s][i].byte().cpu().numpy()
                    image = np.transpose(image, (1,2,0))
                    image = np.squeeze(image)
                    im = Image.fromarray(image)
                    im.save(save_path)


    def createVideoFromInference(self, mode, dataset_config, num_frames):
        """
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        """
        if self._local_rank == 0:
            self._logger.val(f"Started creating video from inference for {mode} subset and {self._model_name}.", ":createVideoFromInference")

            data_types_A = dataset_config["datasetATypes"]
            dataset_dirs = [os.path.join(self._model_dir, "inference", mode, f"real_{real_A}_A") for real_A in data_types_A]  
            dataset_dirs.extend([os.path.join(self._model_dir, "inference", mode, f"real_frames_B"), os.path.join(self._model_dir, "inference", mode, f"fake_frames_B")])

            for dataset_dir in dataset_dirs:
                frame_list = []
                name = dataset_dir.split("/")[-1]
                jpgs = sorted(glob.glob(f'{dataset_dir}/*/*.jpg'))
                jpgs.sort(key=lambda x: ''.join(x.split(f"/{name}/")[-1].split("_")[1:]))
                for f, filename in enumerate(jpgs):
                    im = Image.open(filename)
                    im = np.asarray(im)
                    if len(im.shape) == 2:
                        im = np.expand_dims(im, axis=2)
                    frame_list.append(im)
                    if f >= num_frames:
                        break   

            frame_list = np.transpose(np.asarray(frame_list), [0,3,1,2])  
            if self._use_wandb:    
                summary = {f"{name}_video_1fps":wandb.Video(frame_list, fps=1, format="mp4"),
                            f"{name}_video_10fps":wandb.Video(frame_list, fps=10, format="mp4")}
                wandb.log(summary, step=self._state["current_step"])

            self._logger.val(f"Finished creating video from inference for {mode} subset and {self._model_name}.", ":createVideoFromInference")

    
    
