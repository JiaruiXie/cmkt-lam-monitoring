"""
Label processing utilities for CMKT framework.

This module:
- cleans metadata
- removes invalid samples
- removes Laser-off samples
- converts class labels into numeric labels
"""

import numpy as np


# ------------------------------------------------------
# Prepare binary classification labels
# ------------------------------------------------------
def prepare_binary_labels(metadata):

    """
    Prepare binary labels for defect classification.

    Parameters
    ----------
    metadata : DataFrame

    Returns
    -------
    metadata : DataFrame
        Filtered metadata.

    labels : ndarray
        Numeric labels.
    """

    # ----------------------------------------------
    # Clean class names
    # ----------------------------------------------

    metadata["class_name"] = (
        metadata["class_name"]
        .astype(str)
        .str.strip()
    )

    # ----------------------------------------------
    # Keep only valid classes
    # ----------------------------------------------

    metadata = metadata[
        metadata["class_name"].isin([
            "Defect-free",
            "Defective"
        ])
    ]

    # ----------------------------------------------
    # Reset index
    # ----------------------------------------------

    metadata = metadata.reset_index(
        drop=True
    )

    # ----------------------------------------------
    # Label mapping
    # ----------------------------------------------

    label_mapping = {
        "Defect-free": 0,
        "Defective": 1
    }

    # ----------------------------------------------
    # Convert labels
    # ----------------------------------------------

    labels = metadata["class_name"].map(
        label_mapping
    )

    labels = labels.values

    return metadata, labels