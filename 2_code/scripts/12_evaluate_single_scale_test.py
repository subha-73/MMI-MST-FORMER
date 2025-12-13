import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

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
# NOTE: Ensure these match your saved model (Standard vs Aug)
from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer
from modules.single_scale_transformer import SingleScaleTransformer
from modules.dataset_loader import GrapeDataset

# CONFIG
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_single_scale_model.pth')

def evaluate_best_model():
    print(f"{'='*60}")
    print(f"EVALUATION REPORT: BEST SINGLE SCALE MODEL (FULL TEST SET)")
    print(f"{'='*60}")

    # --- 1. LOAD DATA ---
    print("\n[STEP 1] Loading Data...")
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    
    # POINTING TO TEST DATA
    tensor_path = os.path.join(processed_dir, 'grape_test_images.pt')
    csv_path = os.path.join(processed_dir, 'grape_test.csv')

    # Load Full Dataset (No Split!)
    dataset = GrapeDataset(tensor_path, csv_path)
    clinical_dim = dataset.get_clinical_dim()
    
    # Use the WHOLE dataset for evaluation
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"   ✓ Evaluation Samples: {len(dataset)} (Should be 40)")

    # --- 2. LOAD ARCHITECTURE ---
    print("\n[STEP 2] Loading Best Model Weights...")
    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clinical_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256).to(DEVICE)

    # Load State Dict
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    spatial.load_state_dict(checkpoint['spatial'])
    clinical.load_state_dict(checkpoint['clinical'])
    fusion.load_state_dict(checkpoint['fusion'])
    transformer.load_state_dict(checkpoint['transformer'])
    
    print(f"   ✓ Weights Loaded Successfully")

    # --- 3. RUN EVALUATION ---
    print("\n[STEP 3] Running Inference...")
    spatial.eval()
    clinical.eval()
    fusion.eval()
    transformer.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    val_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, clin_data, labels in val_loader:
            images, clin_data, labels = images.to(DEVICE), clin_data.to(DEVICE), labels.to(DEVICE)
            
            # Forward Pass
            b, v, c, h, w = images.shape
            spatial_feats = spatial(images.view(-1, c, h, w))
            clin_feats = clinical(clin_data).unsqueeze(1).expand(-1, v, -1).reshape(-1, 128)
            fused_feats = fusion(spatial_feats, clin_feats).view(b, v, -1)
            
            _, logits = transformer(fused_feats)
            
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # --- 4. CALCULATE METRICS ---
    avg_loss = val_loss / len(val_loader)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro') # Macro average is fairer for imbalanced data
    cm = confusion_matrix(all_targets, all_preds)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS FOR BEST MODEL")
    print(f"{'='*60}")
    print(f"1. Loss:        {avg_loss:.4f}")
    print(f"2. Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"3. F1 Score:    {f1:.4f}")
    print(f"\n4. Confusion Matrix:")
    print(f"   [TN  FP]")
    print(f"   [FN  TP]")
    print(f"{cm}")
    
    print(f"\n5. Detailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=['Stable (0)', 'Progression (1)']))
    print(f"{'='*60}")

if __name__ == "__main__":
    evaluate_best_model()