"""General utilities"""

from functools import wraps
from time import time

from loguru import logger


def timeit(f):
    """Wrap a function and log its execution time.

    Args:
        f: Function to wrap.

    Returns:
        Callable: Wrapped function that logs elapsed time.
    """

    @wraps(f)
    def wrap(*args, **kw):
        """Measure duration of the wrapped function and emit a debug log.

        Returns:
            Any: Result of the wrapped function.
        """
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.debug(f"Timeit: {f.__name__}({args}, {kw}), took: {te - ts:2.4f} sec")
        return result

    return wrap
