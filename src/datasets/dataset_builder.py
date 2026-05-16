"""
Dataset construction utilities for CMKT framework.

This module:
- matches samples with annotations
- assigns labels
- separates classes
- compiles numpy arrays
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------
# Compile dataset into numpy arrays
# ------------------------------------------------------
def compile_data(dataset):

    """
    Convert dataset records into numpy arrays.

    Parameters
    ----------
    dataset : list

    Returns
    -------
    data_input : ndarray

    data_label : ndarray
    """

    data_input = np.zeros(
        (
            len(dataset),
            dataset[0][1].shape[0],
            dataset[0][1].shape[1]
        )
    )

    data_label = np.zeros(
        (
            len(dataset),
            1
        )
    )

    for i in range(len(dataset)):

        data_input[i] = dataset[i][1]

        data_label[i] = dataset[i][-1]

    return data_input, data_label


# ------------------------------------------------------
# Build labeled datasets
# ------------------------------------------------------
def build_datasets(
    sample_record,
    annotation_dataframe
):

    """
    Build normal and abnormal datasets.

    Parameters
    ----------
    sample_record : list

    annotation_dataframe : DataFrame

    Returns
    -------
    tuple
    """

    normal_record = []

    abnormal_record = []

    laser_off_record = []

    for record in sample_record:

        ind = annotation_dataframe.index[
            annotation_dataframe[
                'image_file_name'
            ] == record[0]
        ]

        class_name = annotation_dataframe[
            'class_name'
        ][ind[0]]

        # ------------------------------------------
        # Defect-free
        # ------------------------------------------
        if class_name == 'Defect-free':

            normal_record.append(
                record + [[0]]
            )

        # ------------------------------------------
        # Defective
        # ------------------------------------------
        elif class_name == 'Defective':

            abnormal_record.append(
                record + [[1]]
            )

        # ------------------------------------------
        # Laser-off
        # ------------------------------------------
        elif class_name == 'Laser-off':

            laser_off_record.append(
                record + [[2]]
            )

    return (
        normal_record,
        abnormal_record,
        laser_off_record
    )