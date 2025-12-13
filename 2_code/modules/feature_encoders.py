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
        
        # 2. Remove the last classification layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 3. Add a projection layer
        self.projector = nn.Sequential(
            nn.Linear(num_features, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 4. Freeze Early Layers
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
                
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projector(features)
        return embeddings

class ClinicalEncoder(nn.Module):
    """
    Module 2B: Clinical Encoder (MLP)
    """
    def __init__(self, input_dim, output_dim=128):
        super(ClinicalEncoder, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)

class FusionLayer(nn.Module):
    """
    Module 2C: Fusion Layer
    """
    def __init__(self, spatial_dim, clinical_dim, fused_dim):
        super(FusionLayer, self).__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + clinical_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, spatial_emb, clinical_emb):
        combined = torch.cat((spatial_emb, clinical_emb), dim=1)
        return self.fusion(combined)