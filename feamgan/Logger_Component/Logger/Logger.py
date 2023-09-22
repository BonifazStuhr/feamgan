import datetime
import os
import glob
import numpy as np

from termcolor import colored

class LogLevel:
    """
    A "enum" class for the loglevel flags.
    """
    DEBUG = 1
    INFO = 2
    WARNING = 4
    ERROR = 8
    TRAIN = 16
    VAL = 32
    INFER = 64
    ALL = 127

    Color = {1: "blue", 2: "white", 4: "magenta", 8: "red", 16: "yellow", 32: "green", 64: "cyan"}
    String = {1: "DEBUG", 2: "INFO", 4: "WARNING", 8: "ERROR", 16: "TRAIN", 32: "VAL", 64: "INFER"}


class Logger:
    """
    Class for logging messages.
    Messages that correspond to the LogLevel are written to a logfile and output to the console.

    :Attributes:
        __name:          (String) The name of the logger and the corresponding logfile.
        __log_level:     (LogLevel) Specifies which messages are logged.
        __print_level:   (LogLevel) Specifies which messages are printed on the console.
        __file_name:     (String) Name of the logfile.
        __folder:        (String) Relative path to the logfile.
        __mode:          (Char) The write mode.
        __index_file:    (String) The connection to the logfile into a html file..
        __counter:       (Integer) For the naming of the pictures, so that no two pictures are written at the same time.
        __log_html:      (Boolean) If true, logs will be written in a html file.
    """

    def __init__(self, name, folder="", append=False, log_level=LogLevel.ALL, print_level=LogLevel.ALL, log_html=False):
        """
        Constructor for a Logger.
        :param name: (String) The name of the logger. Name of the subdirectory containing the logfiles.
        :param folder: (String) Relative path to the log folder.
        :param append: (Boolean) Determines whether log messages are attached to an existing logfile or whether the old
                        logfile is overwritten. Applies only to the first call of the logger. False by default.
        :param log_level: (LogLevel) Specifies which messages are logged into a html file.
        :param print_level: (LogLevel) Specifies which messages are printed on the console.
        :param log_html: (Boolean) If true, logs will be written in a html file.
        """
        self.__name = name
        self.__log_level = log_level
        self.__print_level = print_level
        self.__log_html = log_html
        self.__file_name = "index.html"
        self.__folder = f"{name}/"
        self.__mode = ""
        self.__counter = 0

        if folder != "":
            self.__folder = f"{folder}/{name}/"

            if not os.path.exists(folder):
                os.mkdir(folder)

        if not os.path.exists(self.__folder):
            os.mkdir(self.__folder)

        if append:
            self.__mode = "a"
        else:
            self.__mode = "w"

        self.__index_file = open(f"{self.__folder}/{self.__file_name}", self.__mode)

    def __del__(self):
        """
        Destructor.
        Closes the link to the logfile.
        """
        self.__index_file.close()

    def getName(self):
        """
        Getter for the logger name.
        :return: (String) The name of the logger.
        """
        return self.__name

    def setLogLevel(self, level):
        """
        Setter for the loglevel.
        Determines which levels are written to the logfile.
        :param level : (LogLevel) The loglevel.
        :example:
            logger.setLogLevel(LogLevel.DEBUG)
            logger.setLogLevel(LogLevel.DEBUG | LogLevel.INFO)
            logger.setLogLevel(LogLevel.DEBUG | LogLevel.WARNING | LogLevel.ERROR)
            logger.setLogLevel(LogLevel.ALL)
        """
        self.__log_level = level

    def setLogHtml(self, log_html):
        """
        Setter for the log_html.
        Determines if logs ar written to a html file.
        :param log_html : (Boolean) If true, logs will be written in a html file.
        :example:
            logger.setLogHtml(True)
        """
        self.__log_html = log_html


    def setPrintLevel(self, level):
        """
        Setter for the print level.
        Determines which levels are output on the console.
        :param level : (LogLevel) The print level.
        :example:
            logger.setPrintLevel(LogLevel.DEBUG)
            logger.setPrintLevel(LogLevel.DEBUG | LogLevel.INFO)
            logger.setPrintLevel(LogLevel.DEBUG | LogLevel.WARNING | LogLevel.ERROR)
            logger.setPrintLevel(LogLevel.ALL)
        """
        self.__print_level = level

    def saveImages(self, value):
        """
        Determines whether images are saved (True) or not (False).
        :param value: (Boolean) Value for the __save_image flag.
        """
        self.__save_images = value

    def debug(self, message, sender=None):
        """
        Logs a debug message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        """
        if ((LogLevel.DEBUG & self.__log_level) > 0) and self.__log_html:
            self.__log(LogLevel.DEBUG, message, sender)

        if (LogLevel.DEBUG & self.__log_level) > 0:
            self.__print(LogLevel.DEBUG, message, sender)

    def info(self, message, sender=None):
        """
        Logs a info message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. [None] by default.
        """
        if ((LogLevel.INFO & self.__log_level) > 0) and self.__log_html:
            self.__log(LogLevel.INFO, message, sender)

        if (LogLevel.INFO & self.__log_level) > 0:
            self.__print(LogLevel.INFO, message, sender)

    def warning(self, message, sender=None):
        """
         Logs a warning message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. [None] by default.
        """
        if ((LogLevel.WARNING & self.__log_level) > 0) and self.__log_html:
            self.__log(LogLevel.WARNING, message, sender)

        if (LogLevel.WARNING & self.__log_level) > 0:
            self.__print(LogLevel.WARNING, message, sender)

    def error(self, message, sender=None):
        """
        Logs a error message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. [None] by default.
        """
        if ((LogLevel.ERROR & self.__log_level)) and self.__log_html > 0:
            self.__log(LogLevel.ERROR, message, sender)

        if (LogLevel.ERROR & self.__log_level) > 0:
            self.__print(LogLevel.ERROR, message, sender)

    def train(self, message, sender=None):
        """
        Logs a train message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        """
        if ((LogLevel.TRAIN & self.__log_level)) and self.__log_html > 0:
            self.__log(LogLevel.TRAIN, message, sender)

        if (LogLevel.TRAIN & self.__log_level) > 0:
            self.__print(LogLevel.TRAIN, message, sender)

    def val(self, message, sender=None):
        """
        Logs a val message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        """
        if ((LogLevel.VAL & self.__log_level)) and self.__log_html > 0:
            self.__log(LogLevel.VAL, message, sender)

        if (LogLevel.VAL & self.__log_level) > 0:
            self.__print(LogLevel.VAL, message, sender)

    def infer(self, message, sender=None):
        """
        Logs a infer message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        """
        if ((LogLevel.INFER & self.__log_level)) and self.__log_html > 0:
            self.__log(LogLevel.INFER, message, sender)

        if (LogLevel.INFER & self.__log_level) > 0:
            self.__print(LogLevel.INFER, message, sender)

    def tfDatasetInfo(self, subset_type, dataset, sender=None):
        """
        Prints information of the given tensorflow dataset.
        :param subset_type: (String) The subset_type of dataset. E.g. "train".
        :param dataset: (tf.data.Dataset) The tensorflow dataset.
        :param sender: (String) The sender of the message. None by default.
        """
        message = "\n-----------------------------------Tensorflow Dataset Information for Subset: " + subset_type \
                  + "-----------------------------------\n"
        message += "Data:\n"
        message += str(dataset)

        if ((LogLevel.INFO & self.__log_level) > 0) and self.__log_html:
            self.__log(LogLevel.INFO, message, sender)

        if (LogLevel.INFO & self.__log_level) > 0:
            self.__print(LogLevel.INFO, message, sender)


    def npDatasetInfo(self, dataset, sender=None):
        """
        Prints information of the given numpy dataset.
        :param dataset: (Dictionary) The numpy dataset.
        :param sender: (String) The sender of the message. None by default.
        """
        message = "\n-----------------------------------Numpy Dataset Information-----------------------------------\n"
        message += "Data:\n"
        for key, value in dataset.items():
            message += f"{key} | Shape: {value.shape} | Type: {value.dtype} | Mean: {np.mean(value)} | Max: {np.max(value)}"
            if "y_" in key:
                message += f" | Label Count: {np.sum(value, axis=0)}\n"
            else:
                message += "\n"

        if ((LogLevel.INFO & self.__log_level) > 0) and self.__log_html:
            self.__log(LogLevel.INFO, message, sender)

        if (LogLevel.INFO & self.__log_level) > 0:
            self.__print(LogLevel.INFO, message, sender)

    def __log(self, level, message, sender):
        """
        Write a message in the logfile.
        :param level: (LogLevel) The loglevel of the message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        """
        if sender is None:
            sender = "UNDEFINED"

        output = "<font style='color: " + LogLevel.Color[
            level] + "; font-family: courier new; font-weight: bold'>" + datetime.datetime.now().strftime(
            "%Y-%m-%d_%H:%M:%S") + ": " + LogLevel.String[
                     level] + " from " + sender + ": '" + message + "'" + "</font><br>"

        self.__index_file.write(output + "<br>")

    def __print(self, level, message, sender):
        """
        Write a message to the console.
        :param level: (LogLevel) The loglevel of the message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        """
        if sender is None:
            sender = "UNDEFINED"

        print(colored(self.__name + ": " + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ": " + LogLevel.String[
            level] + " from " + sender + ": '" + message + "'", LogLevel.Color[level]))