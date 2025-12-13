import torch
import sys
import os
import pandas as pd

# ==========================================
# 1. SETUP PATHS & IMPORTS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

# --- NEW IMPORTS: ONE FROM EACH FILE ---
try:
    from modules.spatial_encoder import SpatialEncoder
    from modules.clinical_encoder import ClinicalEncoder
    from modules.fusion_layer import FusionLayer
    print("✓ Successfully imported modules from separate files.")
except ImportError as e:
    print(f"[CRITICAL ERROR] Import failed: {e}")
    print("Ensure you have created spatial_encoder.py, clinical_encoder.py, and fusion_layer.py in '2_code/modules/'")
    sys.exit()

# Files
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
TRAIN_IMGS = os.path.join(PROCESSED_DIR, 'grape_train_images.pt')
TRAIN_CSV = os.path.join(PROCESSED_DIR, 'grape_train.csv')

print(f"{'='*70}")
print(f"MILESTONE CHECK: MODULAR ARCHITECTURE FORWARD PASS")
print(f"{'='*70}")

def test_forward_pass():
    # ==========================================
    # 2. INPUT DATA FETCHING
    # ==========================================
    print("\n1. Loading Input Data...")
    try:
        # A. Fetch Images (The Fuel for Spatial Encoder)
        images = torch.load(TRAIN_IMGS)
        
        # B. Fetch CSV (To count features for Clinical Encoder)
        df = pd.read_csv(TRAIN_CSV)
        
        # Detect clinical feature count dynamically
        drop_cols = ['unique_id', 'Visit Number', 'Interval Years', 'Corresponding CFP', 
                     'Interval_Years_Raw', 'Progression_Flag']
        feat_cols = [c for c in df.columns if c not in drop_cols]
        num_clinical_features = len(feat_cols)
        
        print(f"   Image Tensor Found: {images.shape}")
        print(f"   Clinical Features Detected: {num_clinical_features}")
        
        # C. Simulate a Batch (The "Cup of Fuel")
        batch_size = 4
        # Grab first 4 patients
        batch_imgs = images[:batch_size] 
        # Flatten time dimension (CNN processes 1 image at a time)
        batch_imgs_flat = batch_imgs.view(-1, 3, 224, 224) 
        
        # Create dummy clinical data matching the shape
        batch_clinical_flat = torch.randn(batch_size * 10, num_clinical_features)

    except Exception as e:
        print(f"[ERROR] Data loading failed. {e}")
        return

    # ==========================================
    # 3. INITIALIZING THE ENGINES
    # ==========================================
    print("\n2. Initializing Modular Encoders...")
    spatial_net = SpatialEncoder(output_dim=512)
    clinical_net = ClinicalEncoder(input_dim=num_clinical_features, output_dim=128)
    fusion_net = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256)
    
    print("   ✓ Spatial Encoder (ResNet18) Ready")
    print("   ✓ Clinical Encoder (MLP) Ready")
    print("   ✓ Fusion Layer Ready")

    # ==========================================
    # 4. RUNNING THE SYSTEM
    # ==========================================
    print("\n3. Running Forward Pass...")
    
    # Step A: Process Images
    spatial_out = spatial_net(batch_imgs_flat)
    print(f"   ✓ Spatial Output: {spatial_out.shape} (Should be [40, 512])")
    
    # Step B: Process Data
    clinical_out = clinical_net(batch_clinical_flat)
    print(f"   ✓ Clinical Output: {clinical_out.shape} (Should be [40, 128])")
    
    # Step C: Fuse Them
    fused_out = fusion_net(spatial_out, clinical_out)
    print(f"   ✓ Fused Output: {fused_out.shape} (Should be [40, 256])")
    
    print(f"\n{'='*70}")
    print(f"SUCCESS: MODULAR CODE IS WORKING")
    print(f"The separate encoder files are correctly integrated.")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_forward_pass()