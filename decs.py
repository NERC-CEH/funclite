import functools as _functools
import time as _time


def timer(number: int = 1) -> tuple:
    """Times a method, printing out the time in seconds

    Args:
        number (int) : The number of repeated executions of the function being wrapped

    Returns:
        tuple[any, int]: The wrapped function return values, and the elapsed time in milliseconds

    Examples:
        >>> @timer;def f(a, b): return a, b  # noqa
        >>> print(f(1, 2))  # noqa
        Elapsed time of f for 1 runs: 1 seconds
        ((1, 2), 1000000)

    Notes:
        Credit to https://bielsnohr.github.io/2021/01/21/timing-decorators-python.html
    """
    def actual_wrapper(func):
        @_functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            tic = _time.perf_counter()
            for i in range(number - 1):
                func(*args, **kwargs)
            else:
                value = func(*args, **kwargs)
            toc = _time.perf_counter()
            elapsed_time = toc - tic
            print(f"Elapsed time of {func.__name__} for {number} runs: "
                  f" {elapsed_time:0.6f} seconds")
            return value, elapsed_time
        return wrapper_timer
    return actual_wrapper


