"""
Semantic alignment architecture for CMKT framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


# ------------------------------------------------------
# Shared convolutional encoder
# ------------------------------------------------------
class SharedConvNet(nn.Module):

    def __init__(
        self,
        num_conv_layers,
        filters1,
        filters2,
        kernel_size
    ):

        super().__init__()

        self.conv_layers = nn.ModuleList()

        self.pool_layers = nn.ModuleList()

        # ------------------------------------------
        # Initial convolution
        # ------------------------------------------

        self.conv_layers.append(
            nn.Conv2d(
                1,
                filters1,
                kernel_size=kernel_size,
                padding='same'
            )
        )

        self.pool_layers.append(
            nn.MaxPool2d(2, 2)
        )

        # ------------------------------------------
        # Intermediate convolutions
        # ------------------------------------------

        for _ in range(num_conv_layers):

            self.conv_layers.append(
                nn.Conv2d(
                    filters1,
                    filters1,
                    kernel_size=kernel_size,
                    padding='same'
                )
            )

            self.pool_layers.append(
                nn.MaxPool2d(1, 1)
            )

        # ------------------------------------------
        # Final convolutions
        # ------------------------------------------

        self.conv_layers.append(
            nn.Conv2d(
                filters1,
                filters2,
                kernel_size=kernel_size,
                padding='same'
            )
        )

        self.pool_layers.append(
            nn.MaxPool2d(2, 2)
        )

        self.conv_layers.append(
            nn.Conv2d(
                filters2,
                filters2 * 2,
                kernel_size=kernel_size,
                padding='same'
            )
        )

        self.pool_layers.append(
            nn.MaxPool2d(2, 2)
        )

        self.flatten = nn.Flatten()

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    def forward(self, x):

        for conv, pool in zip(
            self.conv_layers,
            self.pool_layers
        ):

            x = pool(
                F.relu(conv(x))
            )

        x = self.flatten(x)

        return x


# ------------------------------------------------------
# Task classifier
# ------------------------------------------------------
class ClassifierNet(nn.Module):

    def __init__(
        self,
        dense_units1,
        output_dims: List[int],
        drops,
        filters2
    ):

        super().__init__()

        self.dense_layers = nn.ModuleList()

        self.drop_layers = nn.ModuleList()

        # ------------------------------------------
        # Flatten dimension
        # ------------------------------------------

        flat_dim = filters2 * 2 * 10 * 10

        # ------------------------------------------
        # First dense layer
        # ------------------------------------------

        self.dense_layers.append(
            nn.Linear(
                flat_dim,
                dense_units1
            )
        )

        self.drop_layers.append(
            nn.Dropout(drops)
        )

        input_dim = dense_units1

        # ------------------------------------------
        # Additional dense layers
        # ------------------------------------------

        for output_dim in output_dims:

            self.dense_layers.append(
                nn.Linear(
                    input_dim,
                    output_dim
                )
            )

            self.drop_layers.append(
                nn.Dropout(drops)
            )

            input_dim = output_dim

        # ------------------------------------------
        # Final output layer
        # ------------------------------------------

        self.fc3 = nn.Linear(
            input_dim,
            1
        )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    def forward(self, x):

        for dense, drop in zip(
            self.dense_layers,
            self.drop_layers
        ):

            x = F.relu(
                drop(
                    dense(x)
                )
            )

        x = self.fc3(x)

        return torch.sigmoid(x)


# ------------------------------------------------------
# Pairwise label distance
# ------------------------------------------------------
def pairwise_y(X, Y):

    batch_size_x = X.shape[0]

    batch_size_y = Y.shape[0]

    dim = int(
        torch.prod(
            torch.tensor(X.shape[1:])
        ).item()
    )

    X = X.view(batch_size_x, dim)

    Y = Y.view(batch_size_y, dim)

    X = X.unsqueeze(-1).expand(
        batch_size_x,
        dim,
        batch_size_y
    )

    Y = Y.unsqueeze(-1).expand(
        batch_size_y,
        dim,
        batch_size_x
    ).transpose(0, 2)

    return torch.sum(
        torch.abs(X - Y),
        1
    ) / 2.0


# ------------------------------------------------------
# Pairwise feature distance
# ------------------------------------------------------
def pairwise_X(X, Y):

    X2 = torch.tile(
        torch.sum(
            X**2,
            dim=1,
            keepdim=True
        ),
        [1, Y.shape[0]]
    )

    Y2 = torch.tile(
        torch.sum(
            Y**2,
            dim=1,
            keepdim=True
        ),
        [1, X.shape[0]]
    )

    XY = torch.matmul(
        X,
        Y.t()
    )

    return X2 + Y2.t() - 2 * XY