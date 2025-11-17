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

class DeformableConvModule(nn.Module):
    """
    This module implements the deformable convolution branch.
    It takes the standard feature map (X) as input, calculates the
    deformation offsets, and then uses a bilinear sampler to generate
    the new deformable feature map (X').

    This follows the paper:
    1. "consists of a convolutional layer and a fully connected layer"
    2. "generates the deformation offsets of 3x3x2"
    3. "feature maps are reconstructed according to the offsets by a bilinear sampler"
    """
    def __init__(self, in_channels, in_spatial_size):
        super(DeformableConvModule, self).__init__()

        self.in_channels = in_channels
        self.in_spatial_size = in_spatial_size # This should be 3 (for 3x3)

        # 1. "a convolutional layer..."
        # We'll use a standard 3x3 conv to process the features
        self.offset_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.offset_relu = nn.ReLU()

        # 2. "...and a fully connected layer"
        # This layer will generate the offsets.
        flattened_size = in_channels * (in_spatial_size ** 2)

        # The output size is 3 * 3 * 2 = 18
        # This corresponds to an (x, y) offset for each of the 3x3 grid points
        output_offset_size = (in_spatial_size ** 2) * 2

        self.offset_fc = nn.Linear(flattened_size, output_offset_size)

        # 3. Create a base (identity) grid for the sampler.
        # This grid contains normalized coordinates from -1 to 1.
        # We'll register it as a 'buffer' so it gets saved with the model
        # and moved to the GPU automatically, but is not a trainable parameter.
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, in_spatial_size),
            torch.linspace(-1, 1, in_spatial_size),
            indexing='ij' # Use 'ij' indexing for (H, W) order
        )
        # Stack them to get (H, W, 2) where the last dim is (x, y)
        self.base_grid = torch.stack((grid_x, grid_y), 2)
        self.register_buffer('identity_grid', self.base_grid.unsqueeze(0)) # Add batch dim

    def forward(self, x):
        """
        Forward pass of the deformable module.

        Args:
            x (torch.Tensor): The standard feature map X.
                              Shape: [N, C, 3, 3]

        Returns:
            torch.Tensor: The new deformable feature map X'.
                          Shape: [N, C, 3, 3]
        """
        N = x.shape[0] # Get batch size

        # --- Calculate Offsets ---
        # 1. Pass through conv layer
        offset_features = self.offset_relu(self.offset_conv(x)) # Shape: [N, C, 3, 3]

        # 2. Flatten for FC layer
        offset_features_flat = offset_features.view(N, -1)

        # 3. Pass through FC layer to get offset values
        # Shape: [N, 18]
        offsets = self.offset_fc(offset_features_flat)

        # 4. Reshape offsets to [N, 3, 3, 2] to match the grid shape
        # (N, H_out, W_out, 2)
        offsets = offsets.view(N, self.in_spatial_size, self.in_spatial_size, 2)

        # --- Create Deformable Features ---

        # 5. Create the deformed grid
        # We add the predicted offsets to our base (identity) grid.
        # The identity_grid is [1, 3, 3, 2] and will be broadcasted
        # to match the batch size N of the offsets.
        deformed_grid = self.identity_grid + offsets

        # 6. Reconstruct feature map with the "bilinear sampler"
        # F.grid_sample is PyTorch's implementation of a bilinear sampler.
        # It takes the original feature map 'x' and samples from it
        # at the locations specified by 'deformed_grid'.
        deformable_feature_map = F.grid_sample(
            x,
            deformed_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )

        return deformable_feature_map