import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. SETUP PATHS & IMPORTS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

# --- IMPORTANT: UPDATED IMPORTS TO MATCH YOUR FILENAMES ---
from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer

# Pointing to the NEW files you created:
from modules.single_scale_transformer_aug import SingleScaleTransformer 
from modules.dataset_loader_aug import GrapeDataset

# ==========================================
# 2. CONFIGURATION
# ==========================================
BATCH_SIZE = 8       
LEARNING_RATE = 1e-4 
EPOCHS = 50          
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_single_scale_aug_model.pth')
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# Helper for neat printing
def print_neat_vector(tensor, epoch_num):
    vals = tensor[0, 0].detach().cpu().numpy()
    print(f"\n   [DEBUG] Fusion Vector Sample (Epoch {epoch_num} - Patient 0, Visit 0):")
    print(f"   {'-'*60}")
    for i in range(0, 16, 8):
        chunk = vals[i : i+8]
        line_str = " ".join([f"{float(x):7.4f}" for x in chunk])
        print(f"      [{i:03d}]: {line_str}")
    print(f"   {'-'*60}")

# ==========================================
# 3. TRAINING FUNCTION
# ==========================================
def train_model():
    print(f"{'='*60}")
    print(f"TRAINING PROTOCOL: SINGLE SCALE (AUGMENTED + HIGH DROPOUT)")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    # --- STEP 1: LOAD DATA ---
    print("\n[STEP 1] Loading Data...")
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_train_images.pt')
    csv_path = os.path.join(processed_dir, 'grape_train.csv')

    # 1. Determine Splits (Using Validation Set Logic)
    # We use a dummy loader just to get the length/indices
    full_dataset_ref = GrapeDataset(tensor_path, csv_path, augment=False)
    indices = list(range(len(full_dataset_ref)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # 2. Create Loaders
    # Train Loader: augment=True (Flips + Noise)
    train_ds_full = GrapeDataset(tensor_path, csv_path, augment=True)
    train_data = Subset(train_ds_full, train_idx)
    
    # Val Loader: augment=False (Clean Data)
    val_ds_full = GrapeDataset(tensor_path, csv_path, augment=False)
    val_data = Subset(val_ds_full, val_idx)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    clinical_dim = full_dataset_ref.get_clinical_dim()
    
    print(f"   ✓ Clinical Features: {clinical_dim}")
    print(f"   ✓ Training Samples: {len(train_data)} (Augmented)")
    print(f"   ✓ Validation Samples: {len(val_data)} (Original)")

    # --- STEP 2: INITIALIZE ---
    print("\n[STEP 2] Initializing Architecture...")
    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clinical_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    
    # Initialize Transformer with HIGHER DROPOUT (0.5)
    transformer = SingleScaleTransformer(input_dim=256, dropout=0.5).to(DEVICE)

    # --- STEP 3: OPTIMIZER ---
    optimizer = optim.Adam(
        list(spatial.parameters()) + 
        list(clinical.parameters()) + 
        list(fusion.parameters()) + 
        list(transformer.parameters()), 
        lr=LEARNING_RATE
    )
    
    class_weights = torch.tensor([1.0, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_val_f1 = 0.0

    # --- STEP 4: TRAINING ---
    print("\n[STEP 3] Starting Training Loop...")
    for epoch in range(EPOCHS):
        spatial.train()
        clinical.train()
        fusion.train()
        transformer.train()
        
        train_loss = 0
        all_preds, all_targets = [], []
        
        for batch_idx, (images, clin_data, labels) in enumerate(train_loader):
            images, clin_data, labels = images.to(DEVICE), clin_data.to(DEVICE), labels.to(DEVICE)
            
            b, v, c, h, w = images.shape
            spatial_feats = spatial(images.view(-1, c, h, w))
            clin_feats = clinical(clin_data) 
            clin_feats_expanded = clin_feats.unsqueeze(1).expand(-1, v, -1).reshape(-1, 128)
            fused_feats = fusion(spatial_feats, clin_feats_expanded)
            transformer_input = fused_feats.view(b, v, -1)
            
            if batch_idx == 0:
                print_neat_vector(transformer_input, epoch+1)

            _, logits = transformer(transformer_input)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        # Validation
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

        # Metrics
        train_acc = accuracy_score(all_targets, all_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        avg_loss = train_loss / len(train_loader)
        
        # Update Scheduler
        scheduler.step(val_f1)

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
    print(f"TRAINING COMPLETE. Best Validation F1: {best_val_f1:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_model()