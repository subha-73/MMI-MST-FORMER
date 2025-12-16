import torch
import sys
import os
import pandas as pd

# --- SILENCE WARNINGS ---
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code', 'modules')) 

PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
# --- CRITICAL FIX: File names/Max Seq Len match the generation scripts ---
TRAIN_IMGS = os.path.join(PROCESSED_DIR, 'grape_train_images.pt') 
TRAIN_CLIN = os.path.join(PROCESSED_DIR, 'grape_train_clinical.pt') 
MAX_SEQ_LEN = 9 # <--- CORRECTED MAX SEQ LEN

# Dynamic Imports (assuming the encoders/fusion are in a 'modules' folder)
try:
    from spatial_encoder import SpatialEncoder
    from clinical_encoder import ClinicalEncoder
    from fusion_layer import FusionLayer
except ImportError:
    print("\n[ERROR] Could not import modules. Ensure all modules are accessible.")
    sys.exit(1)


# --- (Helper Functions remain the same, simplified for brevity) ---
def print_header(title):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

def print_neat_vector(tensor, values_per_line=8):
    # ... (implementation remains the same)
    pass

def print_step(module_name, input_shape, output_shape, tensor_data):
    # ... (implementation remains the same)
    pass
# -----------------------------------------------------------------------------


def verify_system():
    print_header("MMI-SST-FORMER: ARCHITECTURE VERIFICATION REPORT")
    
    # --- 1. DATA LOADING ---
    print("\n[STEP 1] LOADING DATA")
    try:
        # Load tensors saved by the pre-processing scripts
        images_seq_full = torch.load(TRAIN_IMGS) # Shape: (B, S=9, 3, 224, 224)
        clinical_seq_full = torch.load(TRAIN_CLIN) # Shape: (B, S=9, F)
        
        # Get feature count
        n_feats = clinical_seq_full.shape[-1]
        
        # Define Batch Size and Visits
        BATCH_SIZE_SIM = 4
        NUM_SAMPLES_FLAT = BATCH_SIZE_SIM * MAX_SEQ_LEN

        # Prepare Image Batch (Flattens 5D -> 4D for Encoders)
        batch_img_seq = images_seq_full[:BATCH_SIZE_SIM]
        batch_img_flat = batch_img_seq.contiguous().view(NUM_SAMPLES_FLAT, 3, 224, 224)
        
        # Prepare Clinical Batch (Flattens 3D -> 2D for Encoders)
        batch_clin_seq = clinical_seq_full[:BATCH_SIZE_SIM]
        batch_clin_flat = batch_clin_seq.contiguous().view(NUM_SAMPLES_FLAT, n_feats)
        
        print(f"  ✓ Data Loaded Successfully (Actual Tensors Used)")
        print(f"  ✓ Clinical Feature Count (n_feats): {n_feats}")
        print(f"  ✓ Simulated Batch Size (visits): {NUM_SAMPLES_FLAT}")
        
    except Exception as e:
        print(f"  [ERROR] Data Loading Failed. Check file names and existence: {e}")
        return

    # --- 2. INITIALIZATION ---
    print("\n[STEP 2] INITIALIZING MODULES")
    SPATIAL_DIM = 512
    CLINICAL_DIM = 128
    FUSED_DIM = 256
    
    spatial = SpatialEncoder(output_dim=SPATIAL_DIM)
    clinical = ClinicalEncoder(input_dim=n_feats, output_dim=CLINICAL_DIM)
    fusion = FusionLayer(spatial_dim=SPATIAL_DIM, clinical_dim=CLINICAL_DIM, fused_dim=FUSED_DIM)
    
    # --- 3. EXECUTION LOG ---
    print_header("EXECUTION LOG (Flat Pass & Sequence Re-Shaping)")

    # A. SPATIAL ENCODER
    sp_out_flat = spatial(batch_img_flat)
    print_step("SPATIAL ENCODER", input_shape=list(batch_img_flat.shape), 
               output_shape=list(sp_out_flat.shape), tensor_data=sp_out_flat)

    # B. CLINICAL ENCODER
    cl_out_flat = clinical(batch_clin_flat)
    print_step("CLINICAL ENCODER", input_shape=list(batch_clin_flat.shape), 
               output_shape=list(cl_out_flat.shape), tensor_data=cl_out_flat)

    # C. RE-SHAPING FOR FUSION (CRITICAL STEP)
    # Convert flat encoder outputs back to sequences: (N_flat, D) -> (B, S, D)
    sp_out_seq = sp_out_flat.view(BATCH_SIZE_SIM, MAX_SEQ_LEN, -1)
    cl_out_seq = cl_out_flat.view(BATCH_SIZE_SIM, MAX_SEQ_LEN, -1)
    
    print("\n--- SHAPE TRANSFORMATION (FLAT -> SEQUENCE) ---")
    print(f"  ► SPATIAL (Flat {sp_out_flat.shape}) -> Sequence {list(sp_out_seq.shape)}")
    print(f"  ► CLINICAL (Flat {cl_out_flat.shape}) -> Sequence {list(cl_out_seq.shape)}")
    
    # D. FUSION LAYER (Run on Sequence Data)
    final_seq_fused = fusion(sp_out_seq, cl_out_seq)
    print_step("FUSION LAYER (Sequence Output)", 
               input_shape=f"Sequence (S={MAX_SEQ_LEN}) of Fused Vecs",
               output_shape=list(final_seq_fused.shape), 
               tensor_data=final_seq_fused.view(-1, FUSED_DIM)) # Flatten for print_neat_vector
               
    print(f"\n{'='*80}")
    print("VERIFICATION COMPLETE: Feature extraction pipeline is fully functional and dimensionally correct.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    verify_system()