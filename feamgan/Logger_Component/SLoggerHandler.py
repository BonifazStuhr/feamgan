import os

from feamgan.utils.TSingleton import TSingleton
from feamgan.Logger_Component.Logger.Logger import Logger

class SLoggerHandler(metaclass=TSingleton):
    """
    Singleton class to handle all loggers.

    :Attributes:
        __logger:    ([Logger]) A list of registrated loggers.
        __folder:    (String) Folder in which the html logfiles will be written.
        __index:     (String) Path to html index file.
        __file:      (String) link to file.
    """

    def __init__(self):
        """
        Constuctor of the logger.
        Determines the location of the logfiles and creates an index file listing all registered loggers.
        """
        self.__logger = []
        self.__folder = "logs"
        self.__index = f"{self.__folder}/index.html"
        self.__file = None

        if self.__folder != "":
            if not os.path.exists(self.__folder):
                os.mkdir(self.__folder)

        self.__file = open(self.__index, "w")
        self.__file.write("<h1 style='font-family: courier new; font-wight: bold;'>List of registered Loggers</h1>")

    def getLogger(self, name, append=False):
        """
        Getter for a logger.
        Returns the logger with the given name; creates a new logger if no logger with the given name exists.
        :param name: (String) The name of the logger.
        :param append: (Boolean) Determines whether log messages are attached to an existing logfile or whether the old
                       logfile is overwritten. Applies only to the first call of the logger. [False] by default.
        :returns log: (Logger) The requested logger.
        """
        for log in self.__logger:
            if (log.getName() == name):
                break
        else:
            log = Logger(name, self.__folder, append)
            self.__logger.append(log)
            self.__file.write(
                "<a style='font-family: courier new; font-wight: bold;' href='./" + name + "/index.html'>" + name + "</a><br>")
        return log


