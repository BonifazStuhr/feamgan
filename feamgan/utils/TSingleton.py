
class TSingleton(type):
    """
    TSingleton is a "template" used to realize a singleton class, witch prevents multi initialization of the class.
    Classes that should be a singleton must specify this class as metaclass.

    :Attributes:
        _instances : (Class) Protected member, containing the one and only instance of the class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        If the singleton class is called, this function returns the one and only instance of the class.
        The instance is created in the first call.
        :param args : Args for the class.
        :param kwargs : Kwargs for the class.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(TSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

