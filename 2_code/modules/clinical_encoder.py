import torch
import torch.nn as nn

class ClinicalEncoder(nn.Module):
    """
    Module 2B: Clinical Encoder (MLP)
    Takes clinical numbers and outputs a feature vector.
    """
    def __init__(self, input_dim, output_dim=128):
        super(ClinicalEncoder, self).__init__()
        
        self.net = nn.Sequential(
            # Layer 1: Expansion
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            
            # Layer 2: Compression to Embedding
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
    def forward(self, x):
        return self.net(x)
"""
#WORKING CODE 

import torch
import torch.nn as nn

class ClinicalEncoder(nn.Module):
    
    def __init__(self, input_dim, output_dim=128):
        super(ClinicalEncoder, self).__init__()
        
        self.net = nn.Sequential(
            # Layer 1: Expansion
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),  # <--- CHANGED: No more dead zeros
            nn.Dropout(0.3),
            
            # Layer 2: Compression to Embedding
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.01)   # <--- CHANGED: No more dead zeros
        )
        
    def forward(self, x):
        return self.net(x)
#NEXT-----------------------------------------    
import torch
import torch.nn as nn

class ClinicalEncoder(nn.Module):
   
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
        """