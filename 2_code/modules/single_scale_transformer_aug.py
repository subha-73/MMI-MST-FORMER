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
    """
    Module 3: Single Scale Transformer (Baseline)
    """
    # CHANGED: Default dropout increased to 0.5
    def __init__(self, input_dim=256, num_heads=4, num_layers=2, num_classes=2, dropout=0.5):
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
            nn.Dropout(dropout), # This will now use 0.5
            nn.Linear(64, num_classes) 
        )

    def forward(self, src):
        x = self.pos_encoder(src)
        memory = self.transformer_encoder(x)
        cls_token = torch.mean(memory, dim=1)
        logits = self.classifier(cls_token)
        return memory, logits