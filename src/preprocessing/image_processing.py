"""
Image preprocessing module for CMKT framework.

This module preprocesses melt pool images into
fixed-size 80x80 grayscale representations.
"""

import numpy as np
from PIL import Image


# ------------------------------------------------------
# Load image
# ------------------------------------------------------
def load_image(image_path):
    """
    Load image from disk.

    Parameters
    ----------
    image_path : str

    Returns
    -------
    PIL.Image
    """

    return Image.open(image_path)


# ------------------------------------------------------
# Convert image to grayscale
# ------------------------------------------------------
def convert_to_grayscale(img):
    """
    Convert RGB image to grayscale.

    Parameters
    ----------
    img : PIL.Image

    Returns
    -------
    PIL.Image
    """

    return img.convert('L')


# ------------------------------------------------------
# Crop image
# ------------------------------------------------------
def crop_image(
    img,
    crop_width=480
):
    """
    Crop image width.

    Parameters
    ----------
    img : PIL.Image

    crop_width : int

    Returns
    -------
    PIL.Image
    """

    return img.crop(
        (
            0,
            0,
            crop_width,
            img.size[1]
        )
    )


# ------------------------------------------------------
# Resize image
# ------------------------------------------------------
def resize_image(
    img,
    output_size=(80, 80)
):
    """
    Resize image to fixed dimensions.

    Parameters
    ----------
    img : PIL.Image

    output_size : tuple

    Returns
    -------
    PIL.Image
    """

    return img.resize(output_size)


# ------------------------------------------------------
# Convert image to numpy array
# ------------------------------------------------------
def image_to_array(img):
    """
    Convert PIL image to numpy array.

    Parameters
    ----------
    img : PIL.Image

    Returns
    -------
    ndarray
    """

    return np.array(img)


# ------------------------------------------------------
# Full preprocessing pipeline
# ------------------------------------------------------
def preprocess_image(
    image_path,
    output_size=(80, 80)
):
    """
    Complete melt pool image preprocessing pipeline.

    Steps:
    1. Load image
    2. Convert to grayscale
    3. Crop image
    4. Resize image
    5. Convert to numpy array

    Parameters
    ----------
    image_path : str

    output_size : tuple

    Returns
    -------
    ndarray
        Processed image array.
    """

    # Load image
    img = load_image(image_path)

    # Convert to grayscale
    img = convert_to_grayscale(img)

    # Crop image
    img = crop_image(img)

    # Resize image
    img = resize_image(
        img,
        output_size
    )

    # Convert to numpy array
    img_array = image_to_array(img)

    return img_array