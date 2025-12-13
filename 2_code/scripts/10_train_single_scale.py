import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
# --- ADD THIS CORRECTLY ---
import warnings
# Filter out the specific PyTorch warning
warnings.filterwarnings("ignore", category=FutureWarning)
# ==========================================
# 1. SETUP PATHS & IMPORTS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

# Import Modules
from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer
from modules.single_scale_transformer import SingleScaleTransformer
from modules.dataset_loader import GrapeDataset

# ==========================================
# 2. CONFIGURATION
# ==========================================
BATCH_SIZE = 8       
LEARNING_RATE = 1e-4 
EPOCHS = 50          
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save Path
SAVE_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_single_scale_model.pth')
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# --- FIXED VISUALIZATION FUNCTION ---
def print_neat_vector(tensor, epoch_num):
    # tensor shape is [Batch, Visits, 256]
    # We want Patient 0, Visit 0 -> Shape [256]
    vals = tensor[0, 0].detach().cpu().numpy()
    
    print(f"\n   [DEBUG] Fusion Vector Sample (Epoch {epoch_num} - Patient 0, Visit 0):")
    print(f"   {'-'*60}")
    # Print first 16 values (2 rows of 8)
    for i in range(0, 16, 8):
        chunk = vals[i : i+8]
        # Iterate through the chunk and format each scalar number
        line_str = " ".join([f"{float(x):7.4f}" for x in chunk])
        print(f"      [{i:03d}]: {line_str}")
    print(f"   {'-'*60}")

# ==========================================
# 3. TRAINING FUNCTION
# ==========================================
def train_model():
    print(f"{'='*60}")
    print(f"TRAINING PROTOCOL: SINGLE SCALE FUSION TRANSFORMER")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    # --- STEP 1: LOAD DATA ---
    print("\n[STEP 1] Loading Data...")
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_train_images.pt')
    csv_path = os.path.join(processed_dir, 'grape_train.csv')

    dataset = GrapeDataset(tensor_path, csv_path)
    clinical_dim = dataset.get_clinical_dim()
    
    # Split Data (80% Train, 20% Val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"   ✓ Clinical Features: {clinical_dim}")
    print(f"   ✓ Training Samples: {len(train_data)}")
    print(f"   ✓ Validation Samples: {len(val_data)}")

    # --- STEP 2: INITIALIZE FULL PIPELINE ---
    print("\n[STEP 2] Initializing Architecture...")
    # A. Encoders
    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clinical_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    
    # B. Transformer
    transformer = SingleScaleTransformer(input_dim=256).to(DEVICE)

    # --- STEP 3: OPTIMIZER ---
    optimizer = optim.Adam(
        list(spatial.parameters()) + 
        list(clinical.parameters()) + 
        list(fusion.parameters()) + 
        list(transformer.parameters()), 
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    # --- STEP 4: TRAINING LOOP ---
    print("\n[STEP 3] Starting Training Loop...")
    
    for epoch in range(EPOCHS):
        # A. TRAIN PHASE
        spatial.train()
        clinical.train()
        fusion.train()
        transformer.train()
        
        train_loss = 0
        all_preds, all_targets = [], []
        
        for batch_idx, (images, clin_data, labels) in enumerate(train_loader):
            images, clin_data, labels = images.to(DEVICE), clin_data.to(DEVICE), labels.to(DEVICE)
            
            # --- FORWARD PASS ---
            b, v, c, h, w = images.shape
            flat_images = images.view(-1, c, h, w)
            spatial_feats = spatial(flat_images) 
            
            clin_feats = clinical(clin_data) 
            clin_feats_expanded = clin_feats.unsqueeze(1).expand(-1, v, -1).reshape(-1, 128)
            
            fused_feats = fusion(spatial_feats, clin_feats_expanded)
            
            # Reshape for Transformer [Batch, Visits, 256]
            transformer_input = fused_feats.view(b, v, -1)
            
            # ** VISUALIZATION (Corrected) **
            if batch_idx == 0:
                print_neat_vector(transformer_input, epoch+1)

            # Transformer
            _, logits = transformer(transformer_input)

            # --- BACKPROP ---
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        # B. VALIDATION PHASE
        transformer.eval() 
        spatial.eval()
        clinical.eval()
        fusion.eval()
        
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for images, clin_data, labels in val_loader:
                images, clin_data, labels = images.to(DEVICE), clin_data.to(DEVICE), labels.to(DEVICE)
                
                b, v, c, h, w = images.shape
                spatial_feats = spatial(images.view(-1, c, h, w))
                clin_feats = clinical(clin_data).unsqueeze(1).expand(-1, v, -1).reshape(-1, 128)
                fused_feats = fusion(spatial_feats, clin_feats).view(b, v, -1)
                
                _, logits = transformer(fused_feats)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        # C. METRICS & SAVING
        train_acc = accuracy_score(all_targets, all_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        avg_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | Val F1: {val_f1:.4f}", end="")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'spatial': spatial.state_dict(),
                'clinical': clinical.state_dict(),
                'fusion': fusion.state_dict(),
                'transformer': transformer.state_dict(),
                'f1_score': val_f1,
                'epoch': epoch
            }, SAVE_PATH)
            print(f"  >>> SAVED (New Best F1)")
        else:
            print("") 

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE.")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Best Model Saved to: {SAVE_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_model()