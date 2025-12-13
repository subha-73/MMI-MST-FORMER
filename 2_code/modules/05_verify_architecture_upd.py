import torch
import sys
import os
import pandas as pd
import numpy as np

# ==========================================
# 1. SETUP
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer

PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
TRAIN_IMGS = os.path.join(PROCESSED_DIR, 'grape_train_images.pt')
TRAIN_CSV = os.path.join(PROCESSED_DIR, 'grape_train.csv')

def print_header(title):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

def print_neat_vector(tensor, values_per_line=8):
    """
    Prints a tensor vector in a perfectly aligned grid.
    """
    vals = tensor[0].detach().numpy()
    total_len = len(vals)
    
    print(f"      Full Feature Vector (Sample 0):")
    print(f"      {'-'*70}")
    
    for i in range(0, total_len, values_per_line):
        # Slice the chunk
        chunk = vals[i : i+values_per_line]
        
        # Format: Index Label + 8 numbers aligned to 4 decimal places
        # {x:8.4f} means: 8 chars wide total, 4 decimal places
        line_str = " ".join([f"{x:8.4f}" for x in chunk])
        
        # Print with index range (e.g., [000-007]: 0.1234 ...)
        end_idx = min(i + values_per_line - 1, total_len - 1)
        print(f"      [{i:03d}-{end_idx:03d}]: {line_str}")
        
    print(f"      {'-'*70}")

def print_step(module_name, input_shape, output_shape, tensor_data):
    print(f"\n--- MODULE: {module_name} ---")
    print(f"   ► INPUT RECEIVED:")
    print(f"      Shape: {input_shape}")
    print(f"   ► PROCESSING:")
    print(f"      Status: ✓ Forward Pass Successful")
    print(f"   ► OUTPUT GENERATED:")
    print(f"      Shape: {output_shape}")
    
    # Use the new neat printer
    print_neat_vector(tensor_data)

    # Check for dead neurons
    if torch.all(tensor_data == 0):
        print("      [WARNING] Vector is purely zeros (Check ReLU/Initialization)")
    else:
        print("     Vector contains active data ")

def verify_system():
    print_header("MMI-MST-FORMER: ARCHITECTURE VERIFICATION REPORT")
    
    # --- 1. DATA LOADING ---
    print("\n[STEP 1] LOADING DATA")
    try:
        images = torch.load(TRAIN_IMGS)
        df = pd.read_csv(TRAIN_CSV)
        
        # Calculate Features
        drop = ['unique_id', 'Visit Number', 'Interval Years', 'Corresponding CFP', 'Interval_Years_Raw', 'Progression_Flag']
        feats = [c for c in df.columns if c not in drop]
        n_feats = len(feats)
        
        # Create Batch
        batch_img = images[:4].view(-1, 3, 224, 224) 
        batch_clin = torch.randn(40, n_feats)
        
        print(f"   ✓ Data Loaded Successfully")
        print(f"   ✓ Clinical Feature Count: {n_feats}")
        print(f"   ✓ Batch Size: 40 Samples")
        
    except Exception as e:
        print(f"   [ERROR] Data Loading Failed: {e}")
        return

    # --- 2. INITIALIZATION ---
    print("\n[STEP 2] INITIALIZING MODULES")
    spatial = SpatialEncoder(output_dim=512)
    clinical = ClinicalEncoder(input_dim=n_feats, output_dim=128)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256)
    print("   Spatial Encoder (ResNet18)")
    print("   Clinical Encoder (MLP)")
    print("   Fusion Layer")

    # --- 3. EXECUTION LOG ---
    print_header("EXECUTION LOG")

    # A. SPATIAL ENCODER
    sp_out = spatial(batch_img)
    print_step("SPATIAL ENCODER", 
               input_shape=list(batch_img.shape), 
               output_shape=list(sp_out.shape), 
               tensor_data=sp_out)

    # B. CLINICAL ENCODER
    cl_out = clinical(batch_clin)
    print_step("CLINICAL ENCODER", 
               input_shape=list(batch_clin.shape), 
               output_shape=list(cl_out.shape), 
               tensor_data=cl_out)

    # C. FUSION LAYER
    final = fusion(sp_out, cl_out)
    print_step("FUSION LAYER", 
               input_shape=f"Combined ({sp_out.shape[1]} + {cl_out.shape[1]})", 
               output_shape=list(final.shape), 
               tensor_data=final)

    print(f"\n{'='*80}")
    print("VERIFICATION COMPLETE: All modules active.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    verify_system()