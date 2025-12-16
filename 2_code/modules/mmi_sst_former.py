import torch
import torch.nn as nn
import torch.nn.functional as F

# Assume modules are imported from the same directory or accessible via sys.path
from .spatial_encoder import SpatialEncoder
from .clinical_encoder import ClinicalEncoder
from .fusion_layer import FusionLayer


# ==========================================
# 1. CORE COMPONENTS
# ==========================================

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for the Transformer.
    """
    def __init__(self, d_model, max_len=9):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape (1, max_len, d_model)

    def forward(self, x):
        # x is [B, S, D]
        # Add positional embedding to the input sequence
        return x + self.pe[:, :x.size(1), :]


class MMI_SST_Former(nn.Module):
    """
    Multi-Modal Sequence-to-Single-Step Transformer for Forecasting.
    
    The architecture:
    1. Feature Extraction: SpatialEncoder & ClinicalEncoder (Parallel, Shared)
    2. Fusion: FusionLayer (Concatenation + Projection)
    3. Temporal Modeling: Transformer Encoder (Self-Attention)
    4. Prediction: Regression Head
    """
    def __init__(self, 
                 clinical_input_dim, 
                 vf_output_dim,
                 max_seq_len=9,
                 spatial_dim=512, 
                 clinical_dim=128, 
                 fused_dim=256,
                 num_heads=8,
                 num_layers=3,
                 dropout=0.1):
        
        super(MMI_SST_Former, self).__init__()
        
        # Hyperparameters
        self.fused_dim = fused_dim
        self.max_seq_len = max_seq_len
        
        # 1. Feature Encoders (Shared across all sequence steps)
        self.spatial_encoder = SpatialEncoder(output_dim=spatial_dim)
        self.clinical_encoder = ClinicalEncoder(input_dim=clinical_input_dim, output_dim=clinical_dim)
        
        # 2. Multi-Modal Fusion
        self.fusion_layer = FusionLayer(spatial_dim=spatial_dim, clinical_dim=clinical_dim, fused_dim=fused_dim)
        
        # 3. Temporal Modeling: Transformer Encoder
        self.positional_encoding = PositionalEncoding(fused_dim, max_len=max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim, 
            nhead=num_heads, 
            dim_feedforward=fused_dim * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Regression Head (Predicts VF features for V_t+1)
        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, vf_output_dim) # Output: [B, S, VF_DIM]
        )

    def forward(self, image_seq, clinical_seq, seq_mask):
        """
        Args:
            image_seq (Tensor): [B, S, 3, 224, 224] (B=Batch, S=Seq_Len)
            clinical_seq (Tensor): [B, S, F_clin]
            seq_mask (Tensor): [B, S] (Boolean mask for valid steps)
        """
        B, S, C, H, W = image_seq.shape # B=Batch, S=Sequence, (C, H, W) for image
        F_clin = clinical_seq.shape[-1]
        
        # --- 1. Feature Extraction (Flatten Sequence to Batch) ---
        
        # Flatten: [B, S, ...] -> [B*S, ...]
        flat_image = image_seq.view(B * S, C, H, W)
        flat_clinical = clinical_seq.view(B * S, F_clin)
        
        # Run Encoders
        # Spatial: [B*S, 3, 224, 224] -> [B*S, spatial_dim]
        spatial_emb_flat = self.spatial_encoder(flat_image)
        # Clinical: [B*S, F_clin] -> [B*S, clinical_dim]
        clinical_emb_flat = self.clinical_encoder(flat_clinical)
        
        # Restore Sequence Dimension: [B*S, D] -> [B, S, D]
        spatial_emb_seq = spatial_emb_flat.view(B, S, -1)
        clinical_emb_seq = clinical_emb_flat.view(B, S, -1)
        
        # --- 2. Multi-Modal Fusion ---
        # Fused: [B, S, spatial_dim + clinical_dim] -> [B, S, fused_dim]
        fused_seq = self.fusion_layer(spatial_emb_seq, clinical_emb_seq)
        
        # --- 3. Temporal Modeling (Transformer) ---
        
        # Add Positional Encoding
        fused_seq = self.positional_encoding(fused_seq)
        
        # Prepare Transformer Padding Mask
        # PyTorch Transformer expects mask where True means MASKED/IGNORED (False=Keep)
        # We assume seq_mask (1=Real, 0=Padding). So invert it: 0=Real, 1=Padding.
        # This is a key detail for handling padded sequence inputs correctly.
        bool_padding_mask = (seq_mask == 0).bool() # [B, S]
        
        # Transformer Forward Pass
        # output_seq: [B, S, fused_dim]
        output_seq = self.transformer_encoder(
            fused_seq, 
            src_key_padding_mask=bool_padding_mask
        )
        
        # --- 4. Regression Head ---
        # Predict V_t+1 based on the representation at time t
        # prediction_seq: [B, S, VF_DIM]
        prediction_seq = self.regression_head(output_seq)
        
        return prediction_seq


if __name__ == '__main__':
    # Test instantiation and forward pass
    
    # 1. Simulate data dimensions
    CLINICAL_INPUT_DIM = 75 # Example: 14 clinical + 61 VF features
    VF_OUTPUT_DIM = 61      # 61 VF points
    BATCH_SIZE = 2
    
    # 2. Initialize Model
    model = MMI_SST_Former(
        clinical_input_dim=CLINICAL_INPUT_DIM,
        vf_output_dim=VF_OUTPUT_DIM
    )
    
    # 3. Simulate Input Tensors (S=9)
    dummy_image_seq = torch.randn(BATCH_SIZE, 9, 3, 224, 224)
    dummy_clinical_seq = torch.randn(BATCH_SIZE, 9, CLINICAL_INPUT_DIM)
    dummy_seq_mask = torch.ones(BATCH_SIZE, 9)
    # Mask one sample partially
    dummy_seq_mask[0, 5:] = 0 # V6-V9 are padded for patient 0
    
    print("--- MMI-SST-Former Test ---")
    print(f"Input Image Seq Shape: {dummy_image_seq.shape}")
    
    # 4. Forward Pass
    with torch.no_grad():
        output = model(dummy_image_seq, dummy_clinical_seq, dummy_seq_mask)
    
    # 5. Check Output
    # Expected shape: [B, S, VF_OUTPUT_DIM]
    print(f"Output Prediction Seq Shape: {output.shape}")
    assert output.shape == (BATCH_SIZE, 9, VF_OUTPUT_DIM)
    print("Test Successful: Output shape matches expected Sequence-to-Sequence prediction.")

    #  (Diagram illustrating the PE -> Multi-Head Attention -> Feed Forward flow.)