import importlib
import traceback

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler
from feamgan.ConfigInput_Component.ConfigProvider import ConfigProvider

class ExperimentScheduler:
    """
    The ExperimentSchedule schedules the experiments defined in the given schedule.
    If a experiment went wrong, the execution goes on for the next experiment and an execution information is printed
    after each experiment.

    :Attributes:
        __schedule:                (Dictionary) The schedule containing the experiment order.
        __successful_experiments:  (Array) Contains the names of the successful experiments.
        __canceled_experiments:    (Array) Contains the names of the canceled experiments.
        __logger:                  (Logger) The logger for the experiments.
        __finished_experiments:    (Integer) Counts the finished experiments (successful or not).
        __controller_config :      (Dictionary) The config of the controller.
    """

    def __init__(self, schedule, controller_config):
        """
        Constructor, initialize member variables.
        :param schedule: (Dictionary) The schedule containing the experiment order.
        :param controller_config: (Dictionary) The config of the controller.
        """
        self.__schedule = schedule
        self.__successful_experiments = []
        self.__canceled_experiments = []
        self.__logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)
        self.__finished_experiments = 0
        self.__controller_config = controller_config

    def execute(self):
        """
        Executes the experiments defined in the given schedule.
        If a experiment went wrong, the execution goes on for the next experiment and an execution information is printed
        after each experiment.
        """
        if self.__schedule["mode"] == "sequential":
            self.__finished_experiments = 0

            for experiment_name in self.__schedule["experimentsToRun"]:
                self.__finished_experiments = self.__finished_experiments + 1
                try:
                    # Dynamically import the experiment class by name.
                    experiment_module = importlib.import_module("feamgan.Experiment_Component.Experiments." + experiment_name)

                    # Dynamically load the provider class by name.
                    # Combined with the above import its like: from Experiment_Component.Experiments.CsnMnistExperiment import CsnMnistExperiment
                    experiment = getattr(experiment_module, experiment_name)

                    for config_name in self.__schedule["experimentConfigsToRun"][experiment_name]:
                        try:
                            config = ConfigProvider().get_config(
                                "feamgan/Experiment_Component/ExperimentConfigs/"+config_name)
                            experiment(config, self.__controller_config).execute()
                            self.__successful_experiments.append(experiment_name)
                            self.__logExecutionInfo()
                        except:
                            self.__logger.error(
                            "Cancled experiment " + experiment_name + " An error accrued:" + str(traceback.format_exc()),
                            "ExperimentScheduler:execute")
                            self.__canceled_experiments.append(experiment_name)
                            self.__logExecutionInfo()

                except:
                    self.__logger.error("Cancled experiment " + experiment_name + " An error accrued:" + str(traceback.format_exc()), "ExperimentScheduler:execute")
                    self.__canceled_experiments.append(experiment_name)
                    self.__logExecutionInfo()


    def __logExecutionInfo(self):
        """
        Logs the execution information (for example the successful or canceled experiments) after each experiment.
        """
        info_text = "\n************ExperimentScheduler************\n" + "Experiment " + str(
            self.__finished_experiments) + " of " + str(
            len(self.__schedule["experimentsToRun"])) + " finished.\nSuccessful experiments: " + str(
            self.__successful_experiments) + "\nCancled experiments: " + str(
            self.__canceled_experiments) + "\n*******************************************"
        self.__logger.info(info_text, "ExperimentScheduler:__logExecutionInfo")
