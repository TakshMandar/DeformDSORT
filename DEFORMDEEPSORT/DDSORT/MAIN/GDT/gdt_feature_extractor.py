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

class GDTFeatureExtractor(nn.Module):
    """
    This is the complete GDT Feature Extractor module.
    It combines the Backbone, Deformable, and Gating modules.

    It takes a batch of raw image patches as input and outputs the
    final 1D feature descriptor vector for the tracker.
    """
    def __init__(self, backbone_out_channels=256, spatial_size=3):
        super(GDTFeatureExtractor, self).__init__()

        # 1. Initialize the Backbone Feature Extractor
        self.feature_extractor = FeatureExtractor()

        # 2. Initialize the Deformable Convolution Module
        self.deform_module = DeformableConvModule(
            in_channels=backbone_out_channels,
            in_spatial_size=spatial_size
        )

        # 3. Initialize the Gating Module
        self.gate_module = GatingModule(
            in_channels=backbone_out_channels,
            in_spatial_size=spatial_size
        )

        # 4. A final flatten layer to create the 1D descriptor vector
        self.flatten = nn.Flatten()

        # Calculate the final descriptor size
        self.descriptor_size = backbone_out_channels * (spatial_size ** 2)

    def forward(self, x):
        """
        Forward pass of the complete GDT module.

        Args:
            x (torch.Tensor): A batch of input image patches.
                              Shape: [N, 3, H, W] (e.g., [32, 3, 128, 64])

        Returns:
            torch.Tensor: The final 1D feature descriptor.
                          Shape: [N, descriptor_size] (e.g., [32, 2304])
        """

        # 1. Get the standard feature map (X)
        # Input: [N, 3, H, W] -> Output: [N, 256, 3, 3]
        standard_features = self.feature_extractor(x)

        # 2. Get the deformable feature map (X')
        # Input: [N, 256, 3, 3] -> Output: [N, 256, 3, 3]
        deformable_features = self.deform_module(standard_features)

        # 3. Get the sigma gating weights
        # Input: [N, 256, 3, 3] -> Output: [N, 1, 3, 3]
        sigma_weights = self.gate_module(standard_features)

        # 4. Perform the Gated Fusion
        # Y = X' * sigma + X * (1 - sigma)
        fused_features = (deformable_features * sigma_weights) + \
                         (standard_features * (1 - sigma_weights))

        # 5. Flatten the 3D feature map into a 1D vector
        # Input: [N, 256, 3, 3] -> Output: [N, 2304]
        descriptor_vector = self.flatten(fused_features)

        return descriptor_vector