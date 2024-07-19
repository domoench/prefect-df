from prefect import get_run_logger
from prefect.exceptions import MissingContextError


class Logger:
    """Rudimentary logger that logs to prefect if in a prefect context,
    and to print statements if not (e.g. in a notebook)"""
    def __init__(self):
        try:
            self.logger = get_run_logger()
        except MissingContextError:
            print('Not logging in Prefect context')
            self.logger = None

    def info(self, s):
        if self.logger:
            self.logger.info(s)
        else:
            print(s)


# Singleton logger instance
logger = Logger()


def get_logger():
    return logger
