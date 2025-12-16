import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import warnings

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
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_forecasting_model.pth')

def print_vector_summary(name, vector, element_type="vector", full_print=False):
    """
    Prints a clear, neat summary of a tensor, showing the first 5 values 
    or the full array for the final prediction output.
    """
    vector_cpu = vector.detach().cpu()
    
    # 1. Standardize the array for printing/indexing
    if vector_cpu.dim() > 0:
        # Flatten to 1D, taking the first element of any batch/sequence for the audit
        if vector_cpu.dim() > 1:
             # If it's a sequence ([9, 256]), flatten the first step for summary
             if 'Sequence' in element_type:
                 vector_to_print = np.ravel(vector_cpu.view(-1, vector_cpu.shape[-1])[0].numpy())
             # Otherwise, just flatten the whole thing (like a single [1, 256])
             else:
                 vector_to_print = np.ravel(vector_cpu.numpy())
        else:
             vector_to_print = np.ravel(vector_cpu.numpy())
    else:
        # Scalar case
        vector_to_print = np.array([vector_cpu.item()])

    print("\n" + "="*80)
    print(f"### {name} ###")
    print(f"Role: {element_type}")
    print(f"Shape: {vector_cpu.shape}")
    print(f"Min: {vector_cpu.min().item():.4f}, Max: {vector_cpu.max().item():.4f}")
    
    # --- Custom Printing Logic ---
    if full_print or 'VF Points' in element_type:
        print("\nNUMERICAL ARRAY (FULL VECTOR DISPLAYED):")
        # Custom format for guaranteed printing of all 61 VF points
        formatted_output = [f"{x:.5f}" for x in vector_to_print]
        print(", ".join(formatted_output))
    
    elif element_type in ["Sequence of 9 Fused Vectors", "Low-Frequency Feature Vector (Z_LF)"]:
         # Summary for 256-D features
         summary = np.round(vector_to_print[:10], 5).tolist() # Printing first 10 for better sample
         print(f"Content: First step of the historical sequence (256-D vector).")
         print(f"Summary (First 10 values): {summary} ...")
    else:
         # Should not be reached based on the request, but for completeness
         summary = np.round(vector_to_print[:5], 5).tolist()
         print(f"Summary (First 5 values): {summary} ...")
    
    print("="*80)


def run_single_step_audit():
    print("="*80)
    print("         TRANSFORMER OUTPUTS: FULL NUMERICAL AUDIT FOR REVIEW")
    print("================================================================================\n")

    # 1. LOAD DATA, INITIALIZE, AND LOAD MODEL (Same setup as before)
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_test_images.pt') 
    csv_path = os.path.join(processed_dir, 'grape_test.csv')
    
    dataset = GrapeDataset(tensor_path, csv_path)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False) 
    
    images_full, clin_data_full, targets_full = next(iter(test_loader))
    images_full, clin_data_full, targets_full = images_full.to(DEVICE), clin_data_full.to(DEVICE), targets_full.to(DEVICE)
    b, s, c, h, w = images_full.shape

    # Extract target for the first forecast step
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
        print(f" [ERROR] Model file not found at: {MODEL_PATH}")
        return

    spatial.eval(); clinical.eval(); fusion.eval(); transformer.eval()

    # --- FORWARD PASS AND AUDIT ---
    with torch.no_grad():
        
        # Calculate Fused Sequence Input
        flat_images_full = images_full.view(-1, c, h, w)
        flat_clin_full = clin_data_full.view(-1, clin_dim)
        spatial_feats_full = spatial(flat_images_full)
        clin_feats_full = clinical(flat_clin_full)
        spatial_seq_full = spatial_feats_full.view(b, s, -1)
        clin_seq_full = clin_feats_full.view(b, s, -1)
        transformer_input = fusion(spatial_seq_full, clin_seq_full) 
        
        # 1. RUN TRANSFORMER TO GET FINAL PREDICTION
        predictions = transformer(transformer_input) 
        
        # 2. EXTRACT LF BRANCH OUTPUT (Z_LF)
        # Z_LF is the feature vector corresponding to the LAST step (index s-1 = 8)
        # This is the last input vector (F_t) processed by the transformer.
        Z_LF = transformer_input[0, s-1].unsqueeze(0) 
        
        # --- PRINTING THE REQUIRED OUTPUTS ---
        
        # A. Low-Frequency Feature Vector (Z_LF) - The Intermediate Output
        print_vector_summary("LF BRANCH OUTPUT (Z_LF)", 
                             Z_LF, 
                             element_type="Low-Frequency Feature Vector (Z_LF)")
        
        # B. Full Fused Sequence Input (For Context)
        print_vector_summary("TRANSFORMER INPUT (Full Fused Sequence)", 
                             transformer_input[0], 
                             element_type="Sequence of 9 Fused Vectors")


        # C. Final Prediction Output (V_t+1)
        print_vector_summary("FINAL OUTPUT (Predicted V_t+1)", 
                             predictions[0, 0].unsqueeze(0), 
                             element_type="61 VF Points (Completed Forecast)", 
                             full_print=True)
        
        # D. Actual Target (V_t+1) - The Ground Truth
        print_vector_summary("TARGET (V_t+1): Actual Forecast Target", 
                             target_step[0, 0].unsqueeze(0), 
                             element_type="61 VF Points (Actual Data)", 
                             full_print=True)


if __name__ == "__main__":
    run_single_step_audit()