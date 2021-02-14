import functools
import time
from .logger import log


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        log.info(f"Elapsed time: {elapsed_time:0.4f} seconds for {func.__name__}")
        return value

    return wrapper_timer
