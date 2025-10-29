"""General utilities"""

import logging
import os
from functools import wraps
from pathlib import Path
from time import time

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
FORMAT = "%(levelname)s:[%(filename)s:%(lineno)s %(funcName)20s() ] %(message)s"

# Create handlers
c_handler = logging.StreamHandler()

# Determine log file path with process-safe default
log_file_path = os.getenv("LOG_FILE")
if log_file_path is None:
    log_file_path = f"file-{os.getpid()}.log"

# Ensure the log directory exists
log_path = Path(log_file_path)
log_path.parent.mkdir(parents=True, exist_ok=True)

f_handler = logging.FileHandler(str(log_path))
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.WARNING)

# Create formatters and add it to handlers
c_format = logging.Formatter(FORMAT)
f_format = logging.Formatter("%(asctime)s|" + FORMAT)
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.debug(f"Timeit: {f.__name__}({args}, {kw}), took: {te - ts:2.4f} sec")
        return result

    return wrap
