import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2

# Make sure your GDTFeatureExtractor class is importable
from gdt_feature_extractor import GDTFeatureExtractor 

class GDT_Descriptor:
    """
    A wrapper class to load the trained GDT descriptor and make it 
    compatible with the DeepSORT feature extractor.
    """
    def __init__(self, model_path, use_cuda=True):
        print("Loading GDT Descriptor Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        # 1. Initialize the standalone extractor model structure
        self.model = GDTFeatureExtractor().to(self.device)

        # 2. Load your final trained weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            print(f"Error loading weights from {model_path}: {e}")
            print("Please ensure 'gdt_descriptor_final.pth' is in the correct path.")
            raise e

        # 3. Set to evaluation mode
        self.model.eval()
        print(f"GDT Descriptor Model loaded and set to eval mode on {self.device}.")

        # 4. Define the input transforms (must match your training)
        self.transform = T.Compose([
            # Resize to the patch size your model was trained on (H, W)
            T.ToTensor(),
            T.Resize((128, 64)), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _preprocess(self, cv2_patches):
        """ Pre-processes a list of OpenCV (NumPy) image patches. """
        batch_tensors = []
        for patch in cv2_patches:
            # Convert BGR (OpenCV) to RGB
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            # Apply transforms
            tensor = self.transform(patch_rgb).to(self.device)
            batch_tensors.append(tensor)

        # Stack into a single batch
        return torch.stack(batch_tensors)

    def __call__(self, cv2_patches):
        """
        Takes a list of CV2 (NumPy) patches and returns a NumPy
        array of feature descriptors.
        """
        if not cv2_patches:
            return np.array([])

        # Pre-process the patches and create a batch
        batch = self._preprocess(cv2_patches)

        # Run inference
        with torch.no_grad():
            features = self.model(batch)

        # Return features as a NumPy array (on the CPU)
        return features.cpu().numpy()