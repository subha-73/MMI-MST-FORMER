#fusion_layer.py
import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, spatial_dim=512, clinical_dim=128, fused_dim=256):
        super(FusionLayer, self).__init__()
        
        # Combined depth: 512 + 128 = 640
        self.input_dim = spatial_dim + clinical_dim
        
        # Use Conv2d (1x1) to fuse clinical info into every patch of the 7x7 grid
        self.projector = nn.Sequential(
            nn.Conv2d(self.input_dim, fused_dim, kernel_size=1),
            nn.GroupNorm(8, fused_dim), # Better than BatchNorm for spatial grids
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2)
        )

    def forward(self, spatial_grid, clinical_emb):
        """
        spatial_grid: [Batch, 512, 7, 7]
        clinical_emb: [Batch, 128]
        """
        b, c, h, w = spatial_grid.shape
        
        # 1. Expand clinical vector to match the 7x7 grid [Batch, 128, 7, 7]
        clinical_rep = clinical_emb.view(b, -1, 1, 1).expand(-1, -1, h, w)
        
        # 2. Concatenate along the CHANNEL dimension
        combined = torch.cat((spatial_grid, clinical_rep), dim=1) # [Batch, 640, 7, 7]
        
        # 3. Project to [Batch, 256, 7, 7]
        return self.projector(combined)
'''
import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    """
    Module 3: Multi-Modal Fusion
    Concatenates Spatial (Image) and Clinical vectors, then projects them.
    """
    def __init__(self, spatial_dim=512, clinical_dim=128, fused_dim=256):
        super(FusionLayer, self).__init__()
        
        # 1. Calculate Combined Input Size
        # We concatenate features: 512 + 128 = 640
        self.input_dim = spatial_dim + clinical_dim
        
        # 2. Projection Layer (Compress to 256)
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, spatial_emb, clinical_emb):
        """
        Args:
            spatial_emb: [Batch, Seq_Len, 512]
            clinical_emb: [Batch, Seq_Len, 128]
        """
        # 1. Concatenate along the FEATURE dimension (Last dimension)
        # dim=2 or dim=-1. NOT dim=1 (which is time/sequence)
        combined = torch.cat((spatial_emb, clinical_emb), dim=-1) 
        
        # Shape is now [Batch, Seq_Len, 640]
        
        # 2. Project down to [Batch, Seq_Len, 256]
        fused = self.projector(combined)
        
        return fused
'''
"""
import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    
    #Module 3: Multi-Modal Fusion
    #Concatenates Spatial (Image) and Clinical vectors, then projects them.
    
    def __init__(self, spatial_dim=512, clinical_dim=128, fused_dim=256):
        super(FusionLayer, self).__init__()
        
        # 1. Calculate Combined Input Size
        self.input_dim = spatial_dim + clinical_dim
        
        # 2. Projection Layer (Compress to 256)
        # CRITICAL FIX: Changed ReLU to LeakyReLU for consistency
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.LeakyReLU(negative_slope=0.01), # <--- CORRECTED
            nn.Dropout(0.2)
        )

    def forward(self, spatial_emb, clinical_emb):
        
        #Args:
        #    spatial_emb: [Batch, Seq_Len, 512]
         #   clinical_emb: [Batch, Seq_Len, 128]
        
        # 1. Concatenate along the FEATURE dimension (Last dimension: dim=-1)
        combined = torch.cat((spatial_emb, clinical_emb), dim=-1)
        
        # 2. Project down to [Batch, Seq_Len, 256]
        fused = self.projector(combined)
        
        return fused


#working code 
import torch
import torch.nn as nn

class FusionLayer(nn.Module):
   
    def __init__(self, spatial_dim=512, clinical_dim=128, fused_dim=256):
        super(FusionLayer, self).__init__()
        
        # 1. Calculate Combined Input Size
        # We concatenate features: 512 + 128 = 640
        self.input_dim = spatial_dim + clinical_dim
        
        # 2. Projection Layer (Compress to 256)
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, spatial_emb, clinical_emb):
       
        # 1. Concatenate along the FEATURE dimension (Last dimension)
        # dim=2 or dim=-1. NOT dim=1 (which is time/sequence)
        combined = torch.cat((spatial_emb, clinical_emb), dim=-1) 
        
        # Shape is now [Batch, Seq_Len, 640]
        
        # 2. Project down to [Batch, Seq_Len, 256]
        fused = self.projector(combined)
        
        return fused

import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    
    def __init__(self, spatial_dim=512, clinical_dim=128, fused_dim=256):
        super(FusionLayer, self).__init__()
        
        # 1. Calculate Combined Input Size
        # We concatenate features: 512 + 128 = 640
        self.input_dim = spatial_dim + clinical_dim
        
        # 2. Projection Layer (Compress to 256)
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, spatial_emb, clinical_emb):
        
        # 1. Concatenate along the FEATURE dimension (Last dimension)
        # dim=2 or dim=-1. NOT dim=1 (which is time/sequence)
        combined = torch.cat((spatial_emb, clinical_emb), dim=-1) 
        
        # Shape is now [Batch, Seq_Len, 640]
        
        # 2. Project down to [Batch, Seq_Len, 256]
        fused = self.projector(combined)
        
        return fused
    

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