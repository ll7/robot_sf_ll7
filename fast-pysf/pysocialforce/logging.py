"""General utilities"""

from functools import wraps
from time import time

from loguru import logger


def timeit(f):
    """Decorator logging how long a function call took."""

    @wraps(f)
    def wrap(*args, **kw):
        """Measure duration of the wrapped function and emit a debug log."""
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.debug(f"Timeit: {f.__name__}({args}, {kw}), took: {te - ts:2.4f} sec")
        return result

    return wrap
