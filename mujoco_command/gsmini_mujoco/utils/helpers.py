import argparse
import time
from contextlib import contextmanager


def Bool(x):
    """
    Convert a value to a boolean-like string.

    This function interprets the input value and returns a boolean value based on common
    representations of `True`. It recognizes the strings "true", "1", and "yes" as
    `True` and everything else as `False`.

    Parameters
    ----------
    x : any
        The value to convert to a boolean-like string. This can be of any type, as it will be
        converted to a string for comparison.

    Returns
    ----------
    bool
        `True` if the value is one of ["true", "1", "yes"], otherwise `False`.
    """
    return str(x).lower() in ["true", "1", "yes"]


def Callable(callable_name: str):
    """
    Retrieve a callable object from the global namespace by its name.

    This function searches for a callable object in the global namespace using its name.
    If the callable is found, it is returned; otherwise, an exception is raised.

    Parameters
    ----------
    callable_name : str
        The name of the callable object to retrieve from the global namespace.

    Returns
    ----------
    callable
        The callable object if it is found in the global namespace.

    Raises
    ------
    argparse.ArgumentTypeError
        If the callable name does not correspond to a callable object in the global namespace.
    """
    # Get the function object from the global namespace
    callable = globals().get(callable_name)
    if callable is None or not callable(callable):
        raise argparse.ArgumentTypeError(f"Function '{callable_name}' not found")
    return callable


@contextmanager
def timer(context: str = None):
    """
    Context manager to measure and print the elapsed time of a code block.

    This context manager allows for the measurement of execution time within a code block.
    It prints the elapsed time upon exiting the context, optionally with a provided
    description.

    Parameters
    ----------
    context : str, optional
        A description to include in the printout alongside the elapsed time. If not provided,
        only the elapsed time will be printed.

    Yields
    ----------
    None
        The context manager does not yield any value; it only measures and prints elapsed time.

    Prints
    ------
    Elapsed time
        The time taken to execute the code block, optionally with the provided context description.
    """
    t = time.perf_counter()

    yield

    if context is not None:
        print(f"{context}: Elapsed time:", time.perf_counter() - t)
    else:
        print("Elapsed time:", time.perf_counter() - t)
