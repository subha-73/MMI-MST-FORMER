import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import sys
import os
import numpy as np

# SETUP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer
from modules.single_scale_transformer import SingleScaleTransformer
from modules.dataset_loader import GrapeDataset

DEVICE = torch.device("cpu") # CPU is fine for inspection
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_forecasting_model.pth')

def print_module_block(name, input_tensor, output_tensor):
    print(f"\n{'='*60}")
    print(f"MODULE: {name}")
    print(f"{'='*60}")
    
    # INPUT INFO
    print(f" [INPUT]")
    print(f"  > Shape: {list(input_tensor.shape)}")
    if input_tensor.numel() > 0:
        flat_in = input_tensor.flatten().detach().numpy()
        print(f"  > Sample Values (First 5): {flat_in[:5]}")
    
    print(f"\n [OUTPUT]")
    print(f"  > Shape: {list(output_tensor.shape)}")
    if output_tensor.numel() > 0:
        flat_out = output_tensor.flatten().detach().numpy()
        print(f"  > Sample Values (First 5): {flat_out[:5]}")
    print(f"{'-'*60}")

def inspect_modules():
    print("LOADING DATA FOR INSPECTION...")
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_train_images.pt')
    csv_path = os.path.join(processed_dir, 'grape_clinical_full_processed3.csv')
    
    dataset = GrapeDataset(tensor_path, csv_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # GET ONE PATIENT
    images, clin_data, targets = next(iter(loader))
    
    # INITIALIZE
    clin_dim = dataset.get_clinical_dim()
    out_dim = dataset.get_output_dim()
    
    spatial = SpatialEncoder(output_dim=512)
    clinical = ClinicalEncoder(input_dim=clin_dim, output_dim=128)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256)
    transformer = SingleScaleTransformer(input_dim=256, output_dim=out_dim)
    
    # LOAD WEIGHTS (If trained)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        spatial.load_state_dict(checkpoint['spatial'])
        clinical.load_state_dict(checkpoint['clinical'])
        fusion.load_state_dict(checkpoint['fusion'])
        transformer.load_state_dict(checkpoint['transformer'])
        print(" > Weights Loaded Successfully.")
    else:
        print(" > WARNING: No trained model found. Using random weights.")

    # --- INSPECTION LOOP ---
    
    # 1. SPATIAL ENCODER
    b, v, c, h, w = images.shape
    flat_imgs = images.view(-1, c, h, w)
    spatial_out = spatial(flat_imgs)
    print_module_block("Spatial Encoder (CNN)", images, spatial_out)
    
    # 2. CLINICAL ENCODER
    flat_clin = clin_data.view(-1, clin_dim)
    clin_out = clinical(flat_clin)
    print_module_block("Clinical Encoder (MLP)", clin_data, clin_out)
    
    # 3. FUSION LAYER
    # Prepare inputs for fusion (Reshape back to sequence)
    spatial_seq = spatial_out.view(b, v, -1)
    clin_seq = clin_out.view(b, v, -1)
    fusion_out = fusion(spatial_seq, clin_seq)
    
    # For display, we show what goes INTO fusion (Tuple of tensors)
    # But print_module_block expects one tensor. We'll show the combined shape conceptually.
    print_module_block("Fusion Layer", spatial_seq, fusion_out)
    
    # 4. TRANSFORMER
    trans_out = transformer(fusion_out)
    print_module_block("MST-Former (Forecasting Head)", fusion_out, trans_out)

if __name__ == "__main__":
    inspect_modules()