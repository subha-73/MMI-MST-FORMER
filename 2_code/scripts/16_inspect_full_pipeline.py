import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import sys
import os
import numpy as np

# --- SILENCE WARNINGS ---
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. SETUP
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

# Import Modules
# NOTE: If using augmented training, ensure these match your saved model's class definitions
from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer
from modules.single_scale_transformer import SingleScaleTransformer
from modules.dataset_loader import GrapeDataset

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_single_scale_model.pth')

def print_neat_vector(tensor, title):
    # Handle different shapes. We always want to see Patient 0.
    if tensor.dim() == 3: # [Batch, Visits, Feats] -> Show Visit 0
        vals = tensor[0, 0].detach().cpu().numpy()
        shape_msg = f"Patient 0, Visit 0 (Shape: {tensor.shape})"
    elif tensor.dim() == 2: # [Batch, Feats] -> Show Patient 0
        vals = tensor[0].detach().cpu().numpy()
        shape_msg = f"Patient 0 (Shape: {tensor.shape})"
    else:
        vals = tensor.detach().cpu().numpy()
        shape_msg = str(tensor.shape)

    print(f"\n   [DEBUG] {title}")
    print(f"   Context: {shape_msg}")
    print(f"   {'-'*60}")
    # Print first 16 values
    for i in range(0, min(16, len(vals)), 8):
        chunk = vals[i : i+8]
        line_str = " ".join([f"{float(x):7.4f}" for x in chunk])
        print(f"      [{i:03d}]: {line_str}")
    print(f"   {'-'*60}")

def inspect_full_chain():
    print(f"{'='*60}")
    print(f"FULL PIPELINE X-RAY: EVERY MODULE OUTPUT")
    print(f"{'='*60}")

    # 1. LOAD DATA
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_test_images.pt')
    csv_path = os.path.join(processed_dir, 'grape_test.csv')
    
    dataset = GrapeDataset(tensor_path, csv_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_data = random_split(dataset, [train_size, val_size])
    
    loader = DataLoader(val_data, batch_size=4, shuffle=False)
    clinical_dim = dataset.get_clinical_dim()

    # 2. LOAD MODEL
    print(f"\n[STEP 1] Loading Weights...")
    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clinical_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    spatial.load_state_dict(checkpoint['spatial'])
    clinical.load_state_dict(checkpoint['clinical'])
    fusion.load_state_dict(checkpoint['fusion'])
    transformer.load_state_dict(checkpoint['transformer'])
    print("   ✓ Model Loaded.")

    # 3. RUN FORWARD PASS
    print("\n[STEP 2] Running Inference Chain...")
    spatial.eval()
    clinical.eval()
    fusion.eval()
    transformer.eval()

    images, clin_data, labels = next(iter(loader))
    images, clin_data = images.to(DEVICE), clin_data.to(DEVICE)

    with torch.no_grad():
        b, v, c, h, w = images.shape
        
        # --- MODULE A: SPATIAL ENCODER ---
        flat_images = images.view(-1, c, h, w)
        spatial_feats = spatial(flat_images) 
        # Reshape back to [Batch, Visits, 512] for viewing
        spatial_view = spatial_feats.view(b, v, -1)
        print_neat_vector(spatial_view, "MODULE A: Spatial Encoder Output (The Eye)")

        # --- MODULE B: CLINICAL ENCODER ---
        clin_feats = clinical(clin_data)
        # Reshape to [Batch, 1, 128] for consistency in viewing
        print_neat_vector(clin_feats, "MODULE B: Clinical Encoder Output (The Stats)")

        # --- MODULE C: FUSION LAYER ---
        clin_feats_expanded = clin_feats.unsqueeze(1).expand(-1, v, -1).reshape(-1, 128)
        fused_feats = fusion(spatial_feats, clin_feats_expanded)
        fused_view = fused_feats.view(b, v, -1)
        print_neat_vector(fused_view, "MODULE C: Fusion Layer Output (Eye + Stats)")

        # --- MODULE D: TRANSFORMER ---
        # 1. Positional Encoding + Encoder Layers
        # We manually call the internal parts to see the memory
        src_pos = transformer.pos_encoder(fused_view)
        memory = transformer.transformer_encoder(src_pos)
        print_neat_vector(memory, "MODULE D: Transformer Memory (Temporal Context)")

        # 2. Pooling (The Summary)
        cls_token = torch.mean(memory, dim=1)
        print_neat_vector(cls_token, "MODULE E: Pooling Layer (The Patient Summary)")

        # --- MODULE F: CLASSIFIER ---
        logits = transformer.classifier(cls_token)
        probs = torch.softmax(logits, dim=1)

        print(f"\n   [DEBUG] MODULE F: Final Decision")
        print(f"   {'-'*60}")
        print(f"   True Label: {labels[0].item()}")
        print(f"   Probabilities: [Stable: {probs[0,0]*100:.2f}% | Progression: {probs[0,1]*100:.2f}%]")
        print(f"   {'-'*60}")

if __name__ == "__main__":
    inspect_full_chain()