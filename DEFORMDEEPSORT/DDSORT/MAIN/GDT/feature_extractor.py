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

class FeatureExtractor(nn.Module):
  """
    This module extracts the "standard feature map" (X) from an input image patch.
    It is based on the first three convolutional layers (blocks) of a pre-trained
    VGG-M model. We use the readily available VGG-16
    from torchvision as a practical and powerful substitute.
    """
  def __init__(self):
    super(FeatureExtractor,self).__init__()
    vgg16 = models.vgg16(pretrained = True)
    self.backbone = nn.Sequential(*list(vgg16.features.children())[:17]) #TAKING OUT THE LAYERS
    self.pool = nn.AdaptiveAvgPool2d((3, 3)) #MAX POOLING
    for param in self.backbone.parameters(): #FREEZING UPDATION
      param.requires_grad = True

  def forward(self, x):
    """
    Forward pass of the feature extractor.

    Args:
        x (torch.Tensor): A batch of input image patches.
                Shape: [N, 3, H, W] (e.g., [32, 3, 128, 64])

    Returns:
      torch.Tensor: The standard feature map X.
              Shape: [N, 256, 3, 3]
    """
    x = self.backbone(x)
    x = self.pool(x)
    return x