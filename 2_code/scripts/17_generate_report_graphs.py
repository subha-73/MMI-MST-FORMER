import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score

# ==========================================
# 1. SETUP
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))
OUTPUT_DIR = os.path.join(ROOT_DIR, '3_results', 'graphs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import Modules
from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer
from modules.single_scale_transformer import SingleScaleTransformer
from modules.dataset_loader import GrapeDataset

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_single_scale_model.pth')
TEST_TENSOR = os.path.join(ROOT_DIR, '1_data', 'processed', 'grape_test_images.pt')
TEST_CSV = os.path.join(ROOT_DIR, '1_data', 'processed', 'grape_test.csv')

def generate_graphs():
    print(f"{'='*60}")
    print(f"GENERATING REPORT GRAPHS")
    print(f"Saving to: {OUTPUT_DIR}")
    print(f"{'='*60}")

    # 1. LOAD DATA & MODEL
    dataset = GrapeDataset(TEST_TENSOR, TEST_CSV)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    clinical_dim = dataset.get_clinical_dim()

    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clinical_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    spatial.load_state_dict(checkpoint['spatial'])
    clinical.load_state_dict(checkpoint['clinical'])
    fusion.load_state_dict(checkpoint['fusion'])
    transformer.load_state_dict(checkpoint['transformer'])
    
    spatial.eval(); clinical.eval(); fusion.eval(); transformer.eval()

    # 2. GET PREDICTIONS
    print("Running Inference on Test Set...")
    all_targets = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, clin_data, labels in loader:
            images, clin_data = images.to(DEVICE), clin_data.to(DEVICE)
            b, v, c, h, w = images.shape
            
            spatial_feats = spatial(images.view(-1, c, h, w))
            clin_feats = clinical(clin_data).unsqueeze(1).expand(-1, v, -1).reshape(-1, 128)
            fused = fusion(spatial_feats, clin_feats).view(b, v, -1)
            
            _, logits = transformer(fused)
            probs = torch.softmax(logits, dim=1)[:, 1] # Probability of Class 1 (Progression)
            preds = torch.argmax(logits, dim=1)
            
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 3. GENERATE GRAPHS

    # --- GRAPH A: CONFUSION MATRIX HEATMAP ---
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stable', 'Progression'], yticklabels=['Stable', 'Progression'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Data)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
    print("   [+] Saved confusion_matrix.png")
    plt.close()

    # --- GRAPH B: ROC CURVE ---
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300)
    print("   [+] Saved roc_curve.png")
    plt.close()

    # --- GRAPH C: METRICS SUMMARY BAR CHART ---
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    metrics = ['Accuracy', 'F1 Score (Macro)', 'AUC']
    values = [acc, f1, roc_auc]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800'])
    plt.ylim(0, 1.0)
    plt.title('Model Performance Metrics')
    
    # Add numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_bar_chart.png'), dpi=300)
    print("   [+] Saved metrics_bar_chart.png")
    plt.close()

    print(f"\nDone! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    generate_graphs()