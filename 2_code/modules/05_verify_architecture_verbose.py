import torch
import sys
import os
import pandas as pd

# Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer

PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
TRAIN_IMGS = os.path.join(PROCESSED_DIR, 'grape_train_images.pt')
TRAIN_CSV = os.path.join(PROCESSED_DIR, 'grape_train.csv')

print(f"{'='*70}")
print(f"MODULE: SYSTEM VERIFICATION (VERBOSE)")
print(f"{'='*70}")

def test_verbose():
    # --- INPUT ---
    print("\n=== INPUT DATA ===")
    images = torch.load(TRAIN_IMGS)
    df = pd.read_csv(TRAIN_CSV)
    
    # Get Feature Count
    drop = ['unique_id', 'Visit Number', 'Interval Years', 'Corresponding CFP', 'Interval_Years_Raw', 'Progression_Flag']
    feats = [c for c in df.columns if c not in drop]
    n_feats = len(feats)
    
    # Create Batch
    batch_img = images[:4].view(-1, 3, 224, 224)
    batch_clin = torch.randn(40, n_feats)
    
    print(f"   > Image Batch Shape: {batch_img.shape}")
    print(f"   > Clinical Batch Shape: {batch_clin.shape}")

    # --- INTERMEDIATE ---
    print("\n=== INTERMEDIATE DELIVERABLES (ENCODERS) ===")
    spatial = SpatialEncoder(output_dim=512)
    clinical = ClinicalEncoder(input_dim=n_feats, output_dim=128)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256)
    
    sp_out = spatial(batch_img)
    cl_out = clinical(batch_clin)
    
    print(f"   > Spatial Encoder Output: {sp_out.shape}")
    print(f"     (Sample Values: {sp_out[0][:3].detach().numpy()}...)")
    print(f"   > Clinical Encoder Output: {cl_out.shape}")
    print(f"     (Sample Values: {cl_out[0][:3].detach().numpy()}...)")

    # --- OUTPUT ---
    print("\n=== FINAL OUTPUT (FUSION) ===")
    final = fusion(sp_out, cl_out)
    print(f"   > Fused Vector Shape: {final.shape}")
    print(f"   > Status: READY FOR TRANSFORMER")

if __name__ == "__main__":
    test_verbose()