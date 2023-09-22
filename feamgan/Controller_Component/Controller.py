import traceback
import subprocess

from feamgan.LoggerNames import LoggerNames
from feamgan.Logger_Component.SLoggerHandler import SLoggerHandler
from feamgan.Experiment_Component.ExperimentScheduler import ExperimentScheduler

class Controller:
    """
    The controller is the central point of the framework. It takes care of the execution of various programs like
    the execution of experiments via a scheduler.

    :Attributes:
        __controller_config:           (Dictionary) The config of the controller.
        __experiment_schedule:         (Dictionary) The schedule containing the experiment order.
        __logger:                      (Logger) The logger for the controller.
        __experiment_scheduler:        (ExperimentScheduler) The scheduler to handle multiple experiments.t.
    """

    def __init__(self, controller_config, experiment_schedule):
        """
        Constructor, initialize member variables.
        :param controller_config_path: (String) The relative path to the controller configuration file.
        :param experiment_schedule_path: (String) The relative path to the experiment schedule file.
        """
        print("Controller: Starting __init__() ...")
        self.__controller_config = controller_config
        self.__experiment_schedule = experiment_schedule
        self.__logger = None
        self.__experiment_scheduler = None
        print("Controller: Finished __init__()")

    def init(self):
        """
        Init method, initialize member variables and other program parts.
        :return: successful: (Boolean) Was the execution successful?
        """
        print("Controller: Starting init() ...")
        self.__logger = SLoggerHandler().getLogger(LoggerNames.CONTROLLER_C)
        self.__logger.info("Loading config ...", "Controller:init")
        successful = True
        try:
            if (self.__controller_config["localRank"] == 0) and self.__controller_config["modelLogging"]["usewandb"]:
                api_key = self.__controller_config["modelLogging"]["wandb"]["apiKey"]
                subprocess.check_call([f"wandb login --relogin {api_key}"], shell=True) 
            self.__logger.info("Finished init()", "Controller:init")
        except:
            successful = False
            self.__logger.error("Canceled init(). An error accrued!", "Controller:init")
            print(traceback.format_exc())

        return successful

    def execute(self):
        """
        Executes the execution specified in the controllers config.
        :return: successful: (Boolean) Was the execution successful?
        """
        self.__logger.info("Starting execute() ...", "Controller:execute")
        successful = True
        if self.__controller_config["executeExperiments"]:
            try:
                self.__logger.info("Starting executeExperiments() ...", "Controller:execute")
                # load schedule
                self.__experiment_scheduler = ExperimentScheduler(self.__experiment_schedule, self.__controller_config)
                self.__experiment_scheduler.execute()
                self.__logger.info("Finished executeExperiments()", "Controller:execute")
            except:
                successful = False
                self.__logger.error("Canceled executeExperiments(). An error occurred!", "Controller:execute")
                print(traceback.format_exc())

        return successful
