import logging
import sys
from IPython import get_ipython


def in_jupyter():
    try:
        if 'IPKernelApp' in get_ipython().config:
            return True
    except AttributeError:
        pass
    return False


# Singleton logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configure handlers
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter)
if not in_jupyter():
    # In the prefect context they add their own handler to stderr.
    # Remove that so I can see log lines from my core module
    for handler in logger.handlers:
        logger.removeHandler(handler)
logger.addHandler(stream_handler)


def get_logger():
    return logger
