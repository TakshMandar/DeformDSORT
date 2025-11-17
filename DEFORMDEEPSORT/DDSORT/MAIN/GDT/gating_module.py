import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import os
import cv2  # OpenCV for image loading and cropping
import numpy as np
import random
import torch.optim as optim

class GatingModule(nn.Module):
    """
    This module implements the gating branch.
    It takes the standard feature map (X) as input and computes the
    gating weights (sigma) which are used to fuse the standard and
    deformable features.

    This follows the paper:
    1. "produced by two consecutive fully connected layers"
    2. "followed by a sigmoid activation"
    3. "outputs a 3 x 3 gating values"
    """
    def __init__(self, in_channels, in_spatial_size):
        super(GatingModule, self).__init__()

        self.in_spatial_size = in_spatial_size # This should be 3 (for 3x3)

        # 1. Calculate the flattened size for the FC layers
        flattened_size = in_channels * (in_spatial_size ** 2) # e.g., 256 * 3 * 3 = 2304

        # 2. Define the hidden size for the first FC layer
        # This is not specified in the paper, so we choose a standard size.
        hidden_size = 512

        # 3. Define the output size
        # "outputs a 3 x 3 gating values" -> 3 * 3 = 9
        output_size = in_spatial_size ** 2

        # 4. Define the layers as a sequential block
        # This block contains the Flatten layer, "two consecutive fully
        # connected layers", and the "sigmoid activation".
        self.gate_layers = nn.Sequential(
            nn.Flatten(),

            # 1st FC Layer
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),

            # 2nd FC Layer
            nn.Linear(hidden_size, output_size),

            # Final Sigmoid Activation
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the gating module.

        Args:
            x (torch.Tensor): The standard feature map X.
                              Shape: [N, C, 3, 3]

        Returns:
            torch.Tensor: The gating weights (sigma).
                          Shape: [N, 1, 3, 3]
        """
        # 1. Pass the input map through the sequential layers
        # Input: [N, 256, 3, 3]
        # Output: [N, 9] (after flatten -> fc1 -> relu -> fc2 -> sigmoid)
        sigma_flat = self.gate_layers(x)

        # 2. Reshape the output to be a 3x3 spatial map
        # We reshape from [N, 9] to [N, 1, 3, 3]
        # The '1' in the channel dimension is crucial. It allows this
        # weight map to be broadcast and multiplied across all 256 channels
        # of the feature maps in the final fusion step.
        sigma = sigma_flat.view(-1, 1, self.in_spatial_size, self.in_spatial_size)

        return sigma