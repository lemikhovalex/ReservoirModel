import numpy as np


def pad_with(vector, pad_width, iaxis, kwargs):
    """
    hard function from https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    used to extract sub matrix with padding
    Args:
        vector: 2d array, matrix
        pad_width: size of pad
        iaxis: dull, need for integration to np
        kwargs: must include "pad_value"

    Returns: nothing, inplace operator

    """
    pad_value = kwargs.get('pad_value', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def get_sub_matrix(x: np.ndarray, k_size: int, center: tuple, pad_value: float) -> np.ndarray:
    """
    reci
    Args:
        x: 2d matrix, fro which the sub matrix will be extracted
        k_size: size of square sub matrix
        center: center of matrix, where from the sub matrix will be taken
        pad_value: taking sub matrix of fixed size must be defined for close to frontier positions.
                   values that are not in matrix, but required by indexing - padding
    Returns: 2d sub matrix
    """
    x_padded_ = np.pad(x, k_size // 2, pad_with, pad_value=pad_value)
    out = x_padded_[center[0]: center[0] + 2 * (k_size // 2) + 1,
                    center[1]: center[1] + 2 * (k_size // 2) + 1
                    ]
    return out
