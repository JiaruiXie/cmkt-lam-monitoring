"""
Multimodal dataset module for CMKT framework.

This dataset combines:
- visual melt pool images
- audio spectrograms
- labels

into unified PyTorch samples.
"""

import torch
from torch.utils.data import Dataset

import numpy as np


class MultimodalDataset(Dataset):

    """
    Multimodal dataset for CMKT framework.
    """

    def __init__(
        self,
        visual_data,
        audio_data,
        labels
    ):
        """
        Initialize dataset.

        Parameters
        ----------
        visual_data : ndarray
            Melt pool image dataset.

        audio_data : ndarray
            Spectrogram dataset.

        labels : ndarray
            Ground truth labels.
        """

        self.visual_data = visual_data
        self.audio_data = audio_data
        self.labels = labels

    # ------------------------------------------------------
    # Dataset length
    # ------------------------------------------------------
    def __len__(self):

        return len(self.labels)

    # ------------------------------------------------------
    # Retrieve one sample
    # ------------------------------------------------------
    def __getitem__(self, idx):

        # ----------------------------------------------
        # Visual modality
        # ----------------------------------------------
        visual_sample = self.visual_data[idx]

        visual_sample = torch.tensor(
            visual_sample,
            dtype=torch.float32
        )

        # Add channel dimension
        visual_sample = visual_sample.unsqueeze(0)

        # ----------------------------------------------
        # Audio modality
        # ----------------------------------------------
        audio_sample = self.audio_data[idx]

        audio_sample = torch.tensor(
            audio_sample,
            dtype=torch.float32
        )

        # Add channel dimension
        audio_sample = audio_sample.unsqueeze(0)

        # ----------------------------------------------
        # Label
        # ----------------------------------------------
        label = self.labels[idx]

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return {
            "visual": visual_sample,
            "audio": audio_sample,
            "label": label
        }