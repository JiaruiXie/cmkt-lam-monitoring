"""
Dataset preparation utilities.

This module combines:
- normal samples
- abnormal samples
- labels
- train/validation/test preparation
"""

import numpy as np


# ------------------------------------------------------
# Combine normal and abnormal datasets
# ------------------------------------------------------
def prepare_dataset(
    normal_input,
    normal_label,
    abnormal_input,
    abnormal_label
):
    """
    Merge normal and abnormal datasets.

    Parameters
    ----------
    normal_input : ndarray

    normal_label : ndarray

    abnormal_input : ndarray

    abnormal_label : ndarray

    Returns
    -------
    tuple
        Combined inputs and labels.
    """

    data_input = np.concatenate(
        (normal_input, abnormal_input),
        axis=0
    )

    data_label = np.concatenate(
        (normal_label, abnormal_label),
        axis=0
    )

    return data_input, data_label




# ------------------------------------------------------
# Compile dataset into numpy arrays
# ------------------------------------------------------
def compile_data(dataset):
    """
    Convert dataset object into numpy arrays.

    Parameters
    ----------
    dataset : list

    Returns
    -------
    tuple
        Input array and label array.
    """

    data_input = np.zeros(
        (
            len(dataset),
            dataset[0][1].shape[0],
            dataset[0][1].shape[1]
        )
    )

    data_label = np.zeros((len(dataset), 1))

    for i in range(len(dataset)):
        data_input[i] = dataset[i][1]
        data_label[i] = dataset[i][0]

    return data_input, data_label