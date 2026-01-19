import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================
# 1. SPACE-TIME POSITIONAL ENCODING (Paper Eq. 2)
# ============================================

class SpaceTimePositionalEncoding(nn.Module):
    """
    Encodes both spatial (patch position) and temporal (visit time) information.
    Follows paper equation (2).
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, B, L, N, time_intervals):
        """
        Args:
            B: batch size
            L: sequence length (number of visits)
            N: number of patches per image
            time_intervals: [B, L] tensor with time deltas from first visit
        
        Returns:
            pe: [B, L, N, d_model] positional encoding
        """
        device = time_intervals.device
        pe = torch.zeros(B, L, N, self.d_model, device=device)
        
        # Frequency bands
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float() 
            * (-math.log(10000.0) / self.d_model)
        )
        
        # Temporal encoding: sin/cos(Δt / 10000^(2i/dm))
        time_enc = time_intervals.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        time_enc = time_enc.expand(B, L, N, self.d_model // 2)  # [B, L, N, d_model/2]
        
        # Spatial encoding: sin/cos(n / 10000^(2i/dm))
        position = torch.arange(N, device=device).float()  # [N]
        position = position.unsqueeze(0).unsqueeze(0).expand(B, L, N)  # [B, L, N]
        position = position.unsqueeze(-1)  # [B, L, N, 1]
        position = position.expand(B, L, N, self.d_model // 2)  # [B, L, N, d_model/2]
        
        # Combine temporal and spatial (Paper Eq. 2)
        pe[..., 0::2] = torch.sin(time_enc * div_term) + torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(time_enc * div_term) + torch.cos(position * div_term)
        
        return pe


# ============================================
# 2. TIME-AWARE TEMPORAL ATTENTION (Paper Eq. 4-5)
# ============================================

class TimeAwareTemporalAttention(nn.Module):
    """
    Temporal attention with time-distance matrix scaling.
    Handles irregular sampling via Ω (omega) matrix.
    """
    def __init__(self, d_model, num_heads, alpha=0.5, beta=0.5, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.alpha = alpha
        self.beta = beta
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_intervals, return_attention=False):
        """
        Args:
            x: [B*N, L, d_model] - features per patch across time
            time_intervals: [B*N, L] - time intervals for each sequence
            return_attention: whether to return attention weights
        
        Returns:
            out: [B*N, L, d_model]
            attn (optional): attention weights
        """
        B_N, L, D = x.shape
        
        # Project to Q, K, V and split heads
        Q = self.q_proj(x).view(B_N, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B_N, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B_N, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B_N, num_heads, L, L]
        
        # Compute time-distance matrix Ω (Paper Eq. 4)
        time_diff = torch.abs(
            time_intervals.unsqueeze(2) - time_intervals.unsqueeze(1)
        )  # [B_N, L, L]
        omega = 1.0 / (1.0 + torch.exp(self.alpha * time_diff - self.beta))
        omega = omega.unsqueeze(1)  # [B_N, 1, L, L] for broadcasting
        
        # Apply causal mask (future tokens masked)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply time-aware scaling (Paper Eq. 5)
        scores = scores * omega
        
        # Softmax and attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = (attn_weights @ V).transpose(1, 2).contiguous()
        out = out.view(B_N, L, D)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn_weights.mean(dim=1)  # Average over heads
        return out


# ============================================
# 3. SPATIAL ATTENTION (Paper Eq. 3)
# ============================================

class SpatialAttention(nn.Module):
    """Standard multi-head self-attention for spatial dimension."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: [B*L, N, d_model] - patches within each image
        """
        B_L, N, D = x.shape
        
        Q = self.q_proj(x).view(B_L, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B_L, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B_L, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = (attn_weights @ V).transpose(1, 2).contiguous()
        out = out.view(B_L, N, D)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn_weights.mean(dim=1)
        return out


# ============================================
# 4. SPATIAL-TEMPORAL ATTENTION BLOCK (Paper Fig. 2b)
# ============================================

class SpatialTemporalAttentionBlock(nn.Module):
    """
    Combines spatial and temporal attention sequentially.
    Processes spatial within each image, then temporal across images.
    """
    def __init__(self, d_model, num_heads, alpha=0.5, beta=0.5, dropout=0.1):
        super().__init__()
        self.spatial_attn = SpatialAttention(d_model, num_heads, dropout)
        self.temporal_attn = TimeAwareTemporalAttention(d_model, num_heads, alpha, beta, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, time_intervals, return_attention=False):
        """
        Args:
            x: [B, L, N, d_model]
            time_intervals: [B, L]
        
        Returns:
            out: [B, L, N, d_model]
        """
        B, L, N, D = x.shape
        
        # --- SPATIAL ATTENTION ---
        x_spatial = x.view(B * L, N, D)
        spatial_out = self.spatial_attn(x_spatial, return_attention=return_attention)
        
        if return_attention:
            spatial_out, spatial_attn = spatial_out
        
        x = self.norm1(x + spatial_out.view(B, L, N, D))
        
        # --- TEMPORAL ATTENTION ---
        x_temporal = x.permute(0, 2, 1, 3).contiguous()  # [B, N, L, D]
        x_temporal = x_temporal.view(B * N, L, D)
        
        # Expand time_intervals for each patch
        time_intervals_expanded = time_intervals.unsqueeze(1).expand(B, N, L).contiguous()
        time_intervals_expanded = time_intervals_expanded.view(B * N, L)
        
        temporal_out = self.temporal_attn(
            x_temporal, time_intervals_expanded, return_attention=return_attention
        )
        
        if return_attention:
            temporal_out, temporal_attn = temporal_out
        
        temporal_out = temporal_out.view(B, N, L, D).permute(0, 2, 1, 3)
        x = self.norm2(x + temporal_out)
        
        # --- FEED-FORWARD ---
        x = self.norm3(x + self.ffn(x))
        
        if return_attention:
            return x, spatial_attn, temporal_attn
        return x


# ============================================
# 5. PATCH MERGING (Multi-scale transition)
# ============================================

class PatchMerging(nn.Module):
    """2x2 patch merging for multi-scale hierarchy. Handles odd-sized grids."""
    def __init__(self, d_model):
        super().__init__()
        self.reduction = nn.Linear(4 * d_model, 2 * d_model)
        self.norm = nn.LayerNorm(2 * d_model)

    def forward(self, x):
        """
        Args:
            x: [B, L, N, d_model] where N = H*W (H and W can be odd)
        
        Returns:
            out: [B, L, N', 2*d_model] where N' ≈ N/4
        """
        B, L, N, D = x.shape
        H = int(math.sqrt(N))
        W = H  # Assuming square patches
        
        # Reshape to 2D grid
        x = x.view(B, L, H, W, D)
        
        # Pad if odd dimensions for safe 2x2 merging
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1, 0, 0, 0, 0, 0, 0))  # Pad height
            H += 1
        if W % 2 != 0:
            x = F.pad(x, (0, 0, 0, 0, 1, 0, 0, 0, 0, 0))  # Pad width
            W += 1
        
        # Reshape after padding
        x = x.view(B, L, H, W, D)
        
        # 2x2 merging
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        
        # Concatenate and project
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, L, H/2, W/2, 4D]
        x = x.view(B, L, -1, 4 * D)  # [B, L, N/4, 4D]
        x = self.reduction(x)
        x = self.norm(x)
        
        return x


# ============================================
# 6. CLINICAL FEATURE INTEGRATION
# ============================================

class ClinicalFusion(nn.Module):
    """Fuses clinical features at each scale."""
    def __init__(self, d_model, num_clinical_features):
        super().__init__()
        self.clinical_proj = nn.Sequential(
            nn.Linear(num_clinical_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, features, clinical_data):
        """
        Args:
            features: [B, L, N, d_model]
            clinical_data: [B, L, num_clinical_features]
        
        Returns:
            fused: [B, L, N, d_model]
        """
        B, L, N, D = features.shape
        
        # Project clinical features
        clinical_proj = self.clinical_proj(clinical_data)  # [B, L, d_model]
        
        # Expand to match patch dimension
        clinical_expanded = clinical_proj.unsqueeze(2).expand(B, L, N, D)  # [B, L, N, d_model]
        
        # Concatenate and fuse
        combined = torch.cat([features, clinical_expanded], dim=-1)  # [B, L, N, 2D]
        combined = combined.view(-1, 2 * D)
        fused = self.fusion(combined)
        fused = fused.view(B, L, N, D)
        
        return fused


# ============================================
# 7. MAIN MMI-MST-FORMER MODEL
# ============================================

class MMI_MST_Former(nn.Module):
    """
    Multi-Scale Spatio-Temporal Transformer for glaucoma forecasting.
    Follows the architecture in the paper.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_scales=3,
        d_model=256,
        num_heads=8,
        num_clinical_features=67,
        num_vf_points=61,
        dropout=0.1,
        alpha=0.5,
        beta=0.5
    ):
        super().__init__()
        self.num_scales = num_scales
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        
        # Positional encoding
        self.pos_embed = SpaceTimePositionalEncoding(d_model)
        
        # Multi-scale blocks
        self.scales = nn.ModuleList([
            SpatialTemporalAttentionBlock(
                d_model * (2 ** i), num_heads, alpha, beta, dropout
            )
            for i in range(num_scales)
        ])
        
        # Clinical fusion at each scale
        self.clinical_fusions = nn.ModuleList([
            ClinicalFusion(d_model * (2 ** i), num_clinical_features)
            for i in range(num_scales)
        ])
        
        # Patch merging for transitions
        self.transitions = nn.ModuleList([
            PatchMerging(d_model * (2 ** i))
            for i in range(num_scales - 1)
        ])
        
        # Decoder
        total_feat_dim = sum(d_model * (2 ** i) for i in range(num_scales))
        self.decoder = nn.Sequential(
            nn.Linear(total_feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_vf_points)
        )

    def forward(self, images, clinical_data, time_intervals, return_attention=False):
        """
        Args:
            images: [B, L, 3, 224, 224]
            clinical_data: [B, L, 67]
            time_intervals: [B, L] - normalized time differences from first visit
            return_attention: whether to return attention weights
        
        Returns:
            out: [B, 61] VF point predictions
        """
        B, L = images.shape[0], images.shape[1]
        
        # Ensure time_intervals is proper shape and device
        if time_intervals.dim() == 1:
            time_intervals = time_intervals.unsqueeze(0).expand(B, -1)
        time_intervals = time_intervals.float().to(images.device)
        
        # Patch embedding for each timestep
        patches = []
        for t in range(L):
            patch_t = self.patch_embed(images[:, t])  # [B, d_model, H', W']
            patch_t = patch_t.flatten(2).transpose(1, 2)  # [B, N, d_model]
            patches.append(patch_t)
        patches = torch.stack(patches, dim=1)  # [B, L, N, d_model]
        
        # Add positional encoding
        pos_enc = self.pos_embed(B, L, self.num_patches, time_intervals)
        x = patches + pos_enc  # [B, L, N, d_model]
        
        # Multi-scale processing
        scale_features = []
        attention_maps = {"spatial": [], "temporal": []}
        
        for i in range(self.num_scales):
            # Spatial-temporal attention
            res = self.scales[i](x, time_intervals, return_attention=return_attention)
            if return_attention:
                x, spatial_attn, temporal_attn = res
                attention_maps["spatial"].append(spatial_attn)
                attention_maps["temporal"].append(temporal_attn)
            else:
                x = res
            
            # Clinical fusion
            x = self.clinical_fusions[i](x, clinical_data)
            
            # Pool from last timestep for forecasting
            scale_feat = x[:, -1].mean(dim=1)  # [B, d_model*(2^i)]
            scale_features.append(scale_feat)
            
            # Patch merging for next scale
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        # Concatenate features from all scales
        combined_feat = torch.cat(scale_features, dim=-1)
        
        # Decode to VF predictions
        vf_pred = self.decoder(combined_feat)  # [B, 61]
        
        if return_attention:
            return {
                "vf_prediction": vf_pred,
                "spatial_attention": attention_maps["spatial"],
                "temporal_attention": attention_maps["temporal"]
            }
        return vf_pred