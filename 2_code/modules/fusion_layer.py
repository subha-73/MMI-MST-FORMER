import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    """
    Module 2C: Fusion Layer
    Combines Spatial and Clinical vectors into a single multi-modal embedding.
    """
    def __init__(self, spatial_dim, clinical_dim, fused_dim):
        super(FusionLayer, self).__init__()
        
        # We concatenate inputs, so input dimension is spatial + clinical
        input_dim = spatial_dim + clinical_dim
        
        self.fusion = nn.Sequential(
            # 1. Compress the combined vector
            nn.Linear(input_dim, fused_dim),
            
            # 2. Normalize to keep training stable
            nn.LayerNorm(fused_dim),
            
            # 3. Activation Function
            nn.LeakyReLU(negative_slope=0.01),  # <--- CHANGED: No more dead zeros
            
            # 4. Dropout to prevent overfitting
            nn.Dropout(0.3)
        )
        
    def forward(self, spatial_emb, clinical_emb):
        # Concatenate along the feature dimension (dim=1)
        # spatial_emb:  [Batch, 512]
        # clinical_emb: [Batch, 128]
        # combined:     [Batch, 640]
        combined = torch.cat((spatial_emb, clinical_emb), dim=1)
        
        # Pass through the fusion network -> [Batch, 256]
        return self.fusion(combined)
"""
import torch
import torch.nn as nn

class FusionLayer(nn.Module):
 
    def __init__(self, spatial_dim, clinical_dim, fused_dim):
        super(FusionLayer, self).__init__()
        
        # We concatenate inputs, so input dimension is spatial + clinical
        input_dim = spatial_dim + clinical_dim
        
        self.fusion = nn.Sequential(
            # 1. Compress the combined vector
            nn.Linear(input_dim, fused_dim),
            
            # 2. Normalize to keep training stable
            nn.LayerNorm(fused_dim),
            
            # 3. Activation Function
            nn.ReLU(),
            
            # 4. Dropout to prevent overfitting
            nn.Dropout(0.3)
        )
        
    def forward(self, spatial_emb, clinical_emb):
        # Concatenate along the feature dimension (dim=1)
        # spatial_emb:  [Batch, 512]
        # clinical_emb: [Batch, 128]
        # combined:     [Batch, 640]
        combined = torch.cat((spatial_emb, clinical_emb), dim=1)
        
        # Pass through the fusion network -> [Batch, 256]
        return self.fusion(combined)
    """