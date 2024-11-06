"""
Classes and functions for handling logging throughout the program.
Generally initialized from main.py, imported via 'import logging as log', and invoked with log.<level>(<string>).
Levels include: debug, info, warning, error, critical.
"""

import os
import sys
import logging
import threading
from traceback import format_exc
from datetime import datetime

from config import ROOT_DIR
logs_dir = os.path.join(ROOT_DIR, 'logs')
os.makedirs(logs_dir, exist_ok=True)

context_data = threading.local()  # can be injected into logs via ContextFilter


class ContextFilter(logging.Filter):
    """
    This filter injects contextual info from `threading.local`.
    """

    def __init__(self, attributes: tuple[str] = ()):
        super().__init__()
        self.attributes = attributes

    def filter(self, record):
        for a in self.attributes:
            setattr(record, a, getattr(context_data, a, ''))
        setattr(record, 'traceback', format_exc())
        return True


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Code that runs when an exception is uncaught.
    """

    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    else:
        logging.critical(f"Uncaught exception: {exc_type}: {exc_value}: {exc_traceback}",
                         exc_info=(exc_type, exc_value, exc_traceback))


def config_logger(descriptor="", time=True, thread=True, level=True, overwrite=False):
    """
    Initializes the loggers. logging.[debug | info | warning | error | critical] will print to console and save to log.
    """

    descriptor = f"_{descriptor}" if descriptor else ""
    time_suffix = "" if overwrite else datetime.now().strftime("_%Y-%m-%d_%H%M%S")
    log_file_name = os.path.join(logs_dir, f"logs{descriptor}{time_suffix}.log")

    # Root logger
    root_logger = logging.getLogger()
    root_logger.level = logging.DEBUG  # default: logging.WARNING
    logs_formatter = logging.Formatter(f"{'%(asctime).16s '   if time   else ''}"
                                       f"{'%(threadName).11s ' if thread else ''}"
                                       f"{'%(levelname).8s ' if level  else ''}"
                                       f"%(message)s")

    # Console logger
    cons_handler = logging.StreamHandler()
    # cons_handler.setFormatter(logs_formatter)
    cons_handler.addFilter(ContextFilter())
    root_logger.addHandler(cons_handler)

    # File logger
    file_handler = logging.FileHandler(log_file_name, 'w+' if overwrite else 'a')
    file_handler.setFormatter(logs_formatter)
    file_handler.addFilter(ContextFilter())
    root_logger.addHandler(file_handler)

    # other settings
    # sys.excepthook = handle_exception                             # Function runs when an exception is uncaught
    # sys.stdout = Logger(log_file_name)                            # Redirect standard out to the log file
    # logging.getLogger('matplotlib.font_manager').disabled = True  # Can disable to avoid potential clutter
