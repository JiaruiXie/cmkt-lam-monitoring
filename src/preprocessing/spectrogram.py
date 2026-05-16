"""
Spectrogram preprocessing module for CMKT framework.

This module converts raw audio signals into
fixed-size 80x80 spectrogram representations.
"""

import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import pylab

from scipy.interpolate import interp2d


# ------------------------------------------------------
# Load WAV file
# ------------------------------------------------------
def get_wav_info(wav_file):
    """
    Read WAV audio file.

    Parameters
    ----------
    wav_file : str
        Path to WAV file.

    Returns
    -------
    sound_info : ndarray
        Audio waveform.

    frame_rate : int
        Audio sampling rate.
    """

    wav = wave.open(wav_file, 'r')

    n_frames = wav.getnframes()

    frames = wav.readframes(-1)

    sound_info = pylab.fromstring(frames, 'int16')

    frame_rate = wav.getframerate()

    wav.close()

    return sound_info, frame_rate


# ------------------------------------------------------
# Generate spectrogram
# ------------------------------------------------------
def graph_spectrogram(wav_file):
    """
    Generate spectrogram using matplotlib specgram.

    Parameters
    ----------
    wav_file : str

    Returns
    -------
    s : ndarray
        Spectrogram matrix.

    f : ndarray
        Frequency bins.

    t : ndarray
        Time bins.
    """

    sound_info, frame_rate = get_wav_info(wav_file)

    s, f, t, _ = plt.specgram(
        sound_info,
        Fs=frame_rate,
        NFFT=165,
        noverlap=145
    )

    plt.close()

    return s, f, t


# ------------------------------------------------------
# Resize spectrogram to 80x80
# ------------------------------------------------------
def resize_spectrogram(
    spectrogram,
    output_shape=(80, 80)
):
    """
    Resize spectrogram using interpolation.

    Parameters
    ----------
    spectrogram : ndarray

    output_shape : tuple

    Returns
    -------
    ndarray
        Resized spectrogram.
    """

    x_new = np.linspace(
        0,
        spectrogram.shape[1] - 1,
        output_shape[1]
    )

    y_new = np.linspace(
        0,
        spectrogram.shape[0] - 1,
        output_shape[0]
    )

    interpolation_function = interp2d(
        np.arange(spectrogram.shape[1]),
        np.arange(spectrogram.shape[0]),
        spectrogram,
        kind='linear'
    )

    resized = interpolation_function(
        x_new,
        y_new
    )

    return resized


# ------------------------------------------------------
# Full preprocessing pipeline
# ------------------------------------------------------
def preprocess_audio(
    wav_file,
    output_shape=(80, 80)
):
    """
    Complete audio preprocessing pipeline.

    Steps:
    1. Load WAV
    2. Generate spectrogram
    3. Resize to fixed dimensions

    Parameters
    ----------
    wav_file : str

    output_shape : tuple

    Returns
    -------
    ndarray
        Processed spectrogram.
    """

    s, _, _ = graph_spectrogram(wav_file)

    s_resized = resize_spectrogram(
        s,
        output_shape
    )

    return s_resized