import torch
import sys
import os
import pandas as pd

# Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

from modules.feature_encoders import SpatialEncoder, ClinicalEncoder, FusionLayer

# Files
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
TRAIN_IMGS = os.path.join(PROCESSED_DIR, 'grape_train_images.pt')
TRAIN_CSV = os.path.join(PROCESSED_DIR, 'grape_train.csv')

print(f"{'='*70}")
print(f"MILESTONE CHECK: ARCHITECTURE FORWARD PASS")
print(f"{'='*70}")

def test_forward_pass():
    print("\n1. Loading Data...")
    try:
        images = torch.load(TRAIN_IMGS)
        df = pd.read_csv(TRAIN_CSV)
        
        # Detect clinical feature count dynamically
        drop_cols = ['unique_id', 'Visit Number', 'Interval Years', 'Corresponding CFP', 
                     'Interval_Years_Raw', 'Progression_Flag']
        feat_cols = [c for c in df.columns if c not in drop_cols]
        num_clinical_features = len(feat_cols)
        
        print(f"   Image Tensor: {images.shape}")
        print(f"   Clinical Features: {num_clinical_features}")
        
        # Prepare Batch
        batch_size = 4
        batch_imgs = images[:batch_size] 
        batch_imgs_flat = batch_imgs.view(-1, 3, 224, 224) # Flatten time
        
        batch_clinical_flat = torch.randn(batch_size * 10, num_clinical_features)

    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print("\n2. Initializing Models...")
    spatial_net = SpatialEncoder(output_dim=512)
    clinical_net = ClinicalEncoder(input_dim=num_clinical_features, output_dim=128)
    fusion_net = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256)

    print("\n3. Running Forward Pass...")
    spatial_out = spatial_net(batch_imgs_flat)
    print(f"   ✓ Spatial Output: {spatial_out.shape}")
    
    clinical_out = clinical_net(batch_clinical_flat)
    print(f"   ✓ Clinical Output: {clinical_out.shape}")
    
    fused_out = fusion_net(spatial_out, clinical_out)
    print(f"   ✓ Fused Output: {fused_out.shape}")
    
    print(f"\n{'='*70}")
   
    print(f"{'='*70}")

if __name__ == "__main__":
    test_forward_pass()