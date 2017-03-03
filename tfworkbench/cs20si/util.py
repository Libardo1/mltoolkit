import os
import time


def newlogname():
    log_basedir = './graphs'
    run_label = time.strftime('%d-%m-%Y_%H-%M-%S')  # e.g. 12-11-2016_18-20-45
    return os.path.join(log_basedir, run_label)


class Example(object):
    """Abstract class for create examples. self.log_path stores
    the directory for tensorboar visualization """
    def __init__(self):
        self.log_path = newlogname()
        self.graphsession()

    def graphsession(self):
        """method to implement all the content of the example
        """
        raise NotImplementedError("Each Model must re-implement this method.")
