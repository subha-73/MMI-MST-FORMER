import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import sys
import os
import numpy as np

# --- ADD THIS CORRECTLY ---
import warnings
# Filter out the specific PyTorch warning
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. SETUP
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

# Import Modules
# NOTE: If you used the 'aug' files, change these imports to _aug versions
from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer
from modules.single_scale_transformer import SingleScaleTransformer
from modules.dataset_loader import GrapeDataset

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Check which model you want to inspect:
# Option A: The standard one
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_single_scale_model.pth')
# Option B: The augmented one (Uncomment if you trained script 11)
# MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_single_scale_aug_model.pth')

def print_neat_vector(tensor, title):
    vals = tensor[0, 0].detach().cpu().numpy() # Patient 0, Visit 0
    print(f"\n   [DEBUG] {title}")
    print(f"   Shape: {tensor.shape}")
    print(f"   {'-'*60}")
    for i in range(0, 16, 8):
        chunk = vals[i : i+8]
        line_str = " ".join([f"{float(x):7.4f}" for x in chunk])
        print(f"      [{i:03d}]: {line_str}")
    print(f"   {'-'*60}")

def inspect_flow():
    print(f"{'='*60}")
    print(f"DIAGNOSTIC: TRANSFORMER DATA FLOW INSPECTION")
    print(f"{'='*60}")

    # 1. LOAD DATA (Just need 1 batch)
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_test_images.pt')
    csv_path = os.path.join(processed_dir, 'grape_test.csv')
    
    # We use validation data to see how it handles "unseen" examples
    dataset = GrapeDataset(tensor_path, csv_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_data = random_split(dataset, [train_size, val_size])
    
    loader = DataLoader(val_data, batch_size=4, shuffle=False)
    clinical_dim = dataset.get_clinical_dim()

    # 2. LOAD MODEL
    print(f"\n[STEP 1] Loading Weights from: {os.path.basename(MODEL_PATH)}")
    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clinical_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model file not found. Did you finish training?")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    spatial.load_state_dict(checkpoint['spatial'])
    clinical.load_state_dict(checkpoint['clinical'])
    fusion.load_state_dict(checkpoint['fusion'])
    transformer.load_state_dict(checkpoint['transformer'])
    print("   ✓ Weights loaded successfully.")

    # 3. RUN ONE PASS
    print("\n[STEP 2] Running Forward Pass on Patient #0...")
    spatial.eval()
    clinical.eval()
    fusion.eval()
    transformer.eval()

    # Get one batch
    images, clin_data, labels = next(iter(loader))
    images, clin_data = images.to(DEVICE), clin_data.to(DEVICE)

    with torch.no_grad():
        # A. Encoders & Fusion
        b, v, c, h, w = images.shape
        spatial_feats = spatial(images.view(-1, c, h, w))
        clin_feats = clinical(clin_data).unsqueeze(1).expand(-1, v, -1).reshape(-1, 128)
        fused_feats = fusion(spatial_feats, clin_feats).view(b, v, -1)
        
        # PRINT 1: INPUT TO TRANSFORMER
        print_neat_vector(fused_feats, "INPUT: Fused Features (Entering Transformer)")

        # B. Transformer Internal
        # We call the sub-components manually to see inside
        # 1. Positional Encoding
        x_pos = transformer.pos_encoder(fused_feats)
        
        # 2. Transformer Encoder (The "Intermediate Deliverable")
        memory = transformer.transformer_encoder(x_pos)
        
        # PRINT 2: INTERMEDIATE
        print_neat_vector(memory, "INTERMEDIATE: Contextualized Memory (Transformer Output)")

        # 3. Classification
        cls_token = torch.mean(memory, dim=1)
        logits = transformer.classifier(cls_token)
        probs = torch.softmax(logits, dim=1)

        # PRINT 3: OUTPUT
        print(f"\n   [DEBUG] FINAL OUTPUT (Logits & Probabilities)")
        print(f"   {'-'*60}")
        print(f"   True Label: {labels[0].item()} ({'Progression' if labels[0]==1 else 'Stable'})")
        print(f"   Raw Logits: [{logits[0,0]:.4f}, {logits[0,1]:.4f}]")
        print(f"   Probs:      [Stable: {probs[0,0]*100:.1f}%, Prog: {probs[0,1]*100:.1f}%]")
        print(f"   {'-'*60}")

if __name__ == "__main__":
    inspect_flow()