"""
matmul: Uses numpy matrix multiplication function (shorthand: @) to multiply two array-like inputs.
"""

from typing import Union

import numpy as np

import logging as log


def matmul(mat_a: Union[list, tuple, np.array], mat_b: Union[list, tuple, np.array]) -> Union[int, float, np.ndarray]:
    """
    Uses numpy matrix multiplication function (shorthand: @) to multiply two array-like inputs.

    :raises TypeError: If either input is not an array-like, including list, tuple, or np.ndarray.
    """

    if type(mat_a) not in [list, tuple, np.ndarray] or \
       type(mat_b) not in [list, tuple, np.ndarray]:
        raise TypeError  # numpy cannot convert input to an array
    result = np.array(mat_a) @ np.array(mat_b)
    log.info(f'Result: {result}')

    return result


def uncovered():
    """
    This function will not execute, demonstrating incomplete coverage...
    """

    assert matmul([1, 2], [1, 1]) == 10  # Line not tested...
