"""General utilities"""

from functools import wraps
from time import time

from loguru import logger


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.debug(f"Timeit: {f.__name__}({args}, {kw}), took: {te - ts:2.4f} sec")
        return result

    return wrap
