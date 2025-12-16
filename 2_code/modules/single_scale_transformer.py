import torch
import torch.nn as nn

class SingleScaleTransformer(nn.Module):
    def __init__(self, input_dim=256, num_heads=4, hidden_dim=512, num_layers=2, output_dim=61):
        super(SingleScaleTransformer, self).__init__()
        
        # We need 9 sequence slots max (since V_t -> V_t+1 shifts)
        self.pos_embedding = nn.Parameter(torch.randn(1, 9, input_dim)) 
        
        # Transformer with Batch First
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # FORECASTING HEAD (Regression)
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim) 
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Dim]
        b, s, f = x.shape
        
        # 1. Add Positional Encoding
        x = x + self.pos_embedding[:, :s, :]
        
        # 2. CREATE CAUSAL MASK (The Blindfold)
        # We enforce Causal attention to ensure model only uses V_t to predict V_t+1
        mask = nn.Transformer.generate_square_subsequent_mask(s).to(x.device)
        
        # 3. Transformer Pass (With Mask)
        # is_causal=True ensures the encoder respects the mask
        x = self.transformer_encoder(x, mask=mask, is_causal=True) 
        
        # 4. Prediction
        output = self.regressor(x) 
        
        return output
"""
#regression
import torch
import torch.nn as nn

class SingleScaleTransformer(nn.Module):
    def __init__(self, input_dim=256, num_heads=4, hidden_dim=512, num_layers=2, output_dim=61):
       
        super(SingleScaleTransformer, self).__init__()
        
        # 1. Feature Projection
        self.embedding = nn.Linear(input_dim, input_dim)
        
        # 2. Positional Encoding (Learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 10, input_dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. FORECASTING HEAD (Regression)
        # We output a vector of size 61 (the predicted VF)
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim) 
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Dim]
        b, s, f = x.shape
        
        # Add Positional Encoding
        x = x + self.pos_embedding[:, :s, :]
        
        # Transformer Pass
        x = self.transformer_encoder(x) # Shape: [Batch, Seq_Len, Input_Dim]
        
        # Prediction
        output = self.regressor(x) # Shape: [Batch, Seq_Len, 61]
        
        return output

import torch
import torch.nn as nn

class SingleScaleTransformer(nn.Module):
    def __init__(self, input_dim=256, num_heads=4, hidden_dim=512, num_layers=2, output_dim=61):
       
        super(SingleScaleTransformer, self).__init__()
        
        # 1. Feature Projection
        self.embedding = nn.Linear(input_dim, input_dim)
        
        # 2. Positional Encoding (Learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 10, input_dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. FORECASTING HEAD (Regression)
        # We output a vector of size 61 (the predicted VF)
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim) 
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Dim]
        b, s, f = x.shape
        
        # Add Positional Encoding
        x = x + self.pos_embedding[:, :s, :]
        
        # Transformer Pass
        x = self.transformer_encoder(x) # Shape: [Batch, Seq_Len, Input_Dim]
        
        # Prediction
        # We predict the VF values for every visit in the sequence
        output = self.regressor(x) # Shape: [Batch, Seq_Len, 61]
        
        return output
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SingleScaleTransformer(nn.Module):
   
    def __init__(self, input_dim=256, num_heads=4, num_layers=2, num_classes=2, dropout=0.3):
        super(SingleScaleTransformer, self).__init__()

        self.input_dim = input_dim

        # 1. Positional Encoding
        self.pos_encoder = PositionalEncoding(input_dim)
        
        # 2. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, 
                                                    dim_feedforward=512, dropout=dropout, 
                                                    activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes) 
        )

    def forward(self, src):
        # A. Add Positional Encoding
        x = self.pos_encoder(src)
        
        # B. Transformer
        memory = self.transformer_encoder(x)
        
        # C. Global Average Pooling
        cls_token = torch.mean(memory, dim=1)
        
        # D. Classification
        logits = self.classifier(cls_token)
        
        return memory, logits
"""