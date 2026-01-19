import torch
import torch.nn as nn

class MXBlock(nn.Module):
    """Spatial-Temporal Transformer Block applied at each scale level."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.encoder(x))

class MSTFormer(nn.Module):
    def __init__(self, input_dim=256, output_dim=61):
        super().__init__()
        
        # Scale 1: 49 Patches
        self.mx_encoder_s1 = MXBlock(input_dim)
        self.trans_1to2 = nn.Conv2d(input_dim, input_dim, kernel_size=2, stride=2)
        
        # Scale 2: 9 Regions
        self.mx_encoder_s2 = MXBlock(input_dim)
        self.trans_2to3 = nn.AdaptiveAvgPool2d(1)
        
        # Scale 3: Global Timeline
        self.mx_encoder_s3 = MXBlock(input_dim)

        # MXDECODER: Masked Time-aware Attention
        self.mx_decoder = nn.TransformerDecoderLayer(d_model=input_dim, nhead=8, batch_first=True)

        # Multi-Task Heads
        self.vf_head = nn.Linear(input_dim, output_dim)
        self.time_head = nn.Linear(input_dim, 1)

    def forward(self, fused_seq, mask=None):
        b, v, c, h, w = fused_seq.shape
        x = fused_seq.view(b*v, c, h, w)
        
        # --- SCALE 1 ---
        x_s1 = x.view(b*v, c, -1).permute(0, 2, 1)
        x_s1 = self.mx_encoder_s1(x_s1)
        
        # --- SCALE 2 ---
        x_img = x_s1.permute(0, 2, 1).view(b*v, c, h, w)
        x_s2_img = self.trans_1to2(x_img) # Zooming out
        x_s2 = x_s2_img.view(b*v, c, -1).permute(0, 2, 1)
        x_s2 = self.mx_encoder_s2(x_s2)
        
        # --- SCALE 3 ---
        x_s3 = self.trans_2to3(x_s2_img).view(b, v, -1)
        x_s3 = self.mx_encoder_s3(x_s3)
        
        # --- DECODER ---
        key_mask = (mask == 0) if mask is not None else None
        target = x_s3[:, -1:, :]
        decoded = self.mx_decoder(target, x_s3, memory_key_padding_mask=key_mask)
        
        return self.vf_head(decoded.squeeze(1)), self.time_head(decoded.squeeze(1))
'''
import torch
import torch.nn as nn
import math

class TimeAwarePositionalEncoding(nn.Module):
    """
    Encodes the ACTUAL time (months) into the embedding.
    Critical for irregular glaucoma visits.
    """
    def __init__(self, d_model, max_len=1000):
        super(TimeAwarePositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Precompute the denominator for sin/cos
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, x, time_steps):
        """
        x: [Batch, Seq, Dim]
        time_steps: [Batch, Seq] (Actual cumulative months e.g. 0, 6, 14)
        """
        # Create positional encoding matrix based on REAL time
        t = time_steps.unsqueeze(-1).float()
        phase = t * self.div_term
        
        pe = torch.zeros_like(x)
        pe[:, :, 0::2] = torch.sin(phase)
        pe[:, :, 1::2] = torch.cos(phase)
        
        return x + pe

class MSTFormer(nn.Module):
    """
    Multi-Scale Spatio-Temporal Transformer (MST-Former)
    
    NOVELTY:
    1. Time-Aware Encoding: Handles irregular visits.
    2. Multi-Scale: Fuses Local (Last Visit) + Global (Avg History).
    3. Multi-Task: Predicts both VF (Vision) and Delta-T (Time).
    """
    def __init__(self, 
                 input_dim=256,      # Matches FusionLayer output
                 num_heads=8,        
                 num_layers=4,       
                 dropout=0.1,
                 output_dim=61):     # 61 Visual Field Points
        super(MSTFormer, self).__init__()
        
        # 1. Time Encoding
        self.pos_encoder = TimeAwarePositionalEncoding(input_dim)
        
        # 2. Transformer Encoder Core
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=input_dim*4, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True # Better stability for medical data
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Multi-Scale Aggregation (Global Pooling)
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        
        # --- HEAD A: Visual Field Predictor ---
        # Predicts the 61 sensitivity points
        self.vf_regressor = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim) 
        )

        # --- HEAD B: Time Interval Predictor (Novelty) ---
        # Predicts the gap (in months) to the next visit
        self.time_regressor = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1) # Outputs 1 scalar (Delta T)
        )

    def forward(self, fused_features, time_steps, mask=None):
        """
        Returns:
            pred_vf: [Batch, 61]
            pred_gap: [Batch, 1]
        """
        # A. Inject Time Awareness
        x = self.pos_encoder(fused_features, time_steps)
        
        # B. Temporal Modeling (Transformer)
        trans_out = self.transformer(x, src_key_padding_mask=mask)
        
        # C. Multi-Scale Extraction
        # Scale 1: Local Context (The most recent visit state)
        local_context = trans_out[:, -1, :]
        
        # Scale 2: Global Context (Long-term history)
        # Permute [Batch, Seq, Dim] -> [Batch, Dim, Seq] for pooling
        global_context = self.global_pool(trans_out.permute(0, 2, 1)).squeeze(-1)
        
        # D. Fusion of Scales
        # Shape: [Batch, 512]
        multi_scale_context = torch.cat((local_context, global_context), dim=1)
        
        # E. Multi-Task Prediction
        pred_vf = self.vf_regressor(multi_scale_context)
        pred_gap = self.time_regressor(multi_scale_context)
        
        return pred_vf, pred_gap

if __name__ == "__main__":
    # Sanity Check
    model = MSTFormer(input_dim=256, output_dim=61)
    dummy_fused = torch.randn(2, 5, 256)
    dummy_time = torch.tensor([[0, 6, 12, 18, 24], [0, 6, 12, 18, 24]])
    
    out_vf, out_time = model(dummy_fused, dummy_time)
    print(f"VF Shape: {out_vf.shape} (Should be [2, 61])")
    print(f"Time Shape: {out_time.shape} (Should be [2, 1])")
'''