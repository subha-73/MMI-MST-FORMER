#spatial_encoder.py
import torch
import torch.nn as nn
import torchvision.models as models

class SpatialEncoder(nn.Module):
    """
    Module 2A: Spatial Encoder (Pre-trained CNN)
    Takes an Eye Image (3, 224, 224) and outputs a feature vector.
    """
    def __init__(self, output_dim=512, freeze_layers=True):
        super(SpatialEncoder, self).__init__()
        
        # 1. Load Pre-trained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Remove the last classification layer (fc)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 3. Add a projection layer to map features to desired dimension
        self.projector = nn.Sequential(
            nn.Linear(num_features, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.01),  # Correct activation
            nn.Dropout(0.3)
        )
        
        # 4. Freeze Early Layers (Transfer Learning)
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze the last convolutional block (Layer 4)
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
                
    def forward(self, x):
        # Pass through ResNet backbone
        features = self.backbone(x)
        
        # Pass through custom projector
        embeddings = self.projector(features)
        
        return embeddings

if __name__ == "__main__":
    model = SpatialEncoder()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Spatial Encoder Output Shape: {output.shape}")
"""
#working code
import torch
import torch.nn as nn
import torchvision.models as models

class SpatialEncoder(nn.Module):
    
    def __init__(self, output_dim=512, freeze_layers=True):
        super(SpatialEncoder, self).__init__()
        
        # 1. Load Pre-trained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Remove the last classification layer (fc)
        # ResNet18 outputs 512 features before the fc layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 3. Add a projection layer to map features to desired dimension
        # UPDATED: Changed ReLU to LeakyReLU to prevent dead neurons (all zeros)
        self.projector = nn.Sequential(
            nn.Linear(num_features, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.01),  # <--- CHANGED HERE
            nn.Dropout(0.3)
        )
        
        # 4. Freeze Early Layers (Transfer Learning)
        if freeze_layers:
            # Freeze everything first
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze the last convolutional block (Layer 4)
            # This allows the model to learn "eye-specific" shapes while keeping
            # basic edge detectors frozen.
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
                
    def forward(self, x):
        # Pass through ResNet backbone
        features = self.backbone(x)
        
        # Pass through custom projector
        embeddings = self.projector(features)
        
        return embeddings

if __name__ == "__main__":
    # Quick Test
    model = SpatialEncoder()
    dummy_input = torch.randn(2, 3, 224, 224) # Batch of 2 images
    output = model(dummy_input)
    print(f"Spatial Encoder Output Shape: {output.shape}")
"""