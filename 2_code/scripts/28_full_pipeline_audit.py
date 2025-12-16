import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import warnings

# Suppress PyTorch/NumPy future warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)

# --- SETUP PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

# Import your core modules
from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer
from modules.single_scale_transformer import SingleScaleTransformer
from modules.dataset_loader import GrapeDataset

# CONFIGURATION
DEVICE = torch.device("cpu") 
# Path to your successfully trained model weights
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_forecasting_model.pth')

def print_audit_block(name, vector, role, is_full_sequence=False):
    """
    Prints a clear, neat summary of a tensor, focusing on the required 
    input/output for each module, with full vectors printed for Z_LF and V_t+1.
    """
    vector_cpu = vector.detach().cpu()
    
    # 1. Standardize the array for printing/indexing
    if vector_cpu.dim() > 0:
        if vector_cpu.dim() > 1:
             # For a sequence input/output, we focus the summary on the first step [0, 0]
             if is_full_sequence:
                 vector_to_print = np.ravel(vector_cpu.view(-1, vector_cpu.shape[-1])[0].numpy())
             else:
                 vector_to_print = np.ravel(vector_cpu.numpy())
        else:
             vector_to_print = np.ravel(vector_cpu.numpy())
    else:
        vector_to_print = np.array([vector_cpu.item()])

    print("\n" + "="*80)
    print(f"### {name} ###")
    print(f"Role: {role}")
    print(f"Shape: {vector_cpu.shape}")
    print(f"Min: {vector_cpu.min().item():.5f}, Max: {vector_cpu.max().item():.5f}")
    
    
    # --- Custom Printing Logic ---
    if '61 VF Points' in role or role == "Low-Frequency Feature Vector (Z_LF)":
        # Force full print for the key 256-D and 61-D outputs for the review
        print("\nNUMERICAL ARRAY (FULL VECTOR DISPLAYED):")
        formatted_output = [f"{x:.5f}" for x in vector_to_print]
        # Printing 10 values per line for better readability of 256-D vector
        print("".join([f"{val}, " if (i + 1) % 10 != 0 else f"{val},\n" for i, val in enumerate(formatted_output)]))
    
    elif 'Image Tensor' in role:
         print(f"Content: First image tensor (3x224x224).")
         
    else:
         # Summarized print for other intermediate vectors (67, 512, 128-D)
         summary = np.round(vector_to_print[:10], 5).tolist()
         print(f"Summary (First 10 values): {summary} ...")
    
    print("="*80)


def run_single_step_audit():
    print("="*80)
    print(" **SINGLE-SCALE TRANSFORMER**")
    #print(" (Demonstrates full data flow for 30% Review)")
    print("="*80)

    # 1. LOAD DATA, INITIALIZE, AND LOAD MODEL
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_test_images1.pt') 
    csv_path = os.path.join(processed_dir, 'grape_test.csv')
    
    try:
        dataset = GrapeDataset(tensor_path, csv_path)
    except FileNotFoundError:
        print(f" [ERROR] Data files not found in {processed_dir}. Please check paths.")
        return
        
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False) 
    images_full, clin_data_full, targets_full = next(iter(test_loader))
    b, s, c, h, w = images_full.shape # b=1, s=9, c=3, h=224, w=224

    # Extract first step (V_t) inputs for single encoder audit
    flat_image_step = images_full[0, 0].unsqueeze(0)
    flat_clin_step = clin_data_full[0, 0].unsqueeze(0)
    target_step = targets_full[0, 0].unsqueeze(0).unsqueeze(0) 

    clin_dim = dataset.get_clinical_dim()
    out_dim = dataset.get_output_dim()
    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clin_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256, output_dim=out_dim).to(DEVICE) 
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        spatial.load_state_dict(checkpoint['spatial'])
        clinical.load_state_dict(checkpoint['clinical'])
        fusion.load_state_dict(checkpoint['fusion'])
        transformer.load_state_dict(checkpoint['transformer'])
    else:
        print(f" [ERROR] Model file not found at: {MODEL_PATH}. Cannot run audit.")
        return

    spatial.eval(); clinical.eval(); fusion.eval(); transformer.eval()

    # --- FORWARD PASS AND AUDIT ---
    with torch.no_grad():
        
        # === MODULE 1: CLINICAL ENCODER ===
        clinical_output = clinical(flat_clin_step)
        print_audit_block("CLINICAL ENCODER", flat_clin_step, 
                          role=f"INPUT (Clinical Data for Visit V_t): {clin_dim}-D Vector")
        print_audit_block("CLINICAL ENCODER", clinical_output, 
                          role="OUTPUT (Clinical Feature Embedding)")

        # === MODULE 2: SPATIAL ENCODER ===
        spatial_output = spatial(flat_image_step)
        print_audit_block("SPATIAL ENCODER", flat_image_step, 
                          role="INPUT (Image Data for Visit V_t): 3x224x224 Image Tensor")
        print_audit_block("SPATIAL ENCODER", spatial_output, 
                          role="OUTPUT (Structural Feature Embedding)")
        
        # --- Prepare Full Sequence for Fusion ---
        flat_images_full = images_full.view(-1, c, h, w)
        flat_clin_full = clin_data_full.view(-1, clin_dim)
        spatial_feats_full = spatial(flat_images_full).view(b, s, -1)
        clin_feats_full = clinical(flat_clin_full).view(b, s, -1)
        
        # === MODULE 3: FUSION LAYER ===
        transformer_input = fusion(spatial_feats_full, clin_feats_full) 
        print_audit_block("FUSION LAYER", spatial_feats_full, 
                          role=f"INPUT 1: Spatial Sequence ({s} steps)", is_full_sequence=True)
        print_audit_block("FUSION LAYER", clin_feats_full, 
                          role=f"INPUT 2: Clinical Sequence ({s} steps)", is_full_sequence=True)
        print_audit_block("FUSION LAYER", transformer_input[0], 
                          role=f"OUTPUT: Fused Sequence (Transformer Input)", is_full_sequence=True)
        
        # === MODULE 4: SINGLE-SCALE TRANSFORMER (LF BRANCH) ===
        predictions = transformer(transformer_input) 
        
        # 4a. EXTRACT Z_LF (The Output for the next stage of the MST)
        # Z_LF is the feature vector corresponding to the LAST step (index s-1 = 8)
        Z_LF = transformer_input[0, s-1].unsqueeze(0) 
        
        print_audit_block("SST ENCODER OUTPUT (Z_LF)", Z_LF, 
                          role="Low-Frequency Feature Vector (Z_LF)", is_full_sequence=False)

        # 4b. Final Prediction Output (V_t+1)
        print_audit_block("SST REGRESSION HEAD OUTPUT", 
                          predictions[0, 0].unsqueeze(0), 
                          role="61 VF Points (Predicted V_t+1)")
        
        # 4c. Ground Truth (for comparison)
        print_audit_block("GROUND TRUTH TARGET", 
                          target_step[0, 0].unsqueeze(0), 
                          role="61 VF Points (Actual Data V_t+1)")


if __name__ == "__main__":
    run_single_step_audit()