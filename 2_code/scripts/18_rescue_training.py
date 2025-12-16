import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os
import csv  # <--- NEW: To save the history
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# SETUP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer
from modules.single_scale_transformer import SingleScaleTransformer
from modules.dataset_loader import GrapeDataset

# CONFIG
BATCH_SIZE = 8
LEARNING_RATE = 5e-5 
EPOCHS = 40 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'balanced_single_scale_model.pth')
LOG_PATH = os.path.join(ROOT_DIR, '3_results', 'training_log.csv') # <--- NEW FILE

def train_rescue_model():
    print(f"{'='*60}")
    print(f"RESCUE MISSION: BALANCED TRAINING (With CSV Logging)")
    print(f"{'='*60}")

    # 1. LOAD DATA
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_train_images.pt')
    csv_path = os.path.join(processed_dir, 'grape_train.csv')

    dataset = GrapeDataset(tensor_path, csv_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. INITIALIZE
    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=dataset.get_clinical_dim(), output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256).to(DEVICE)

    # 3. OPTIMIZER
    optimizer = optim.Adam(
        list(spatial.parameters()) + list(clinical.parameters()) + 
        list(fusion.parameters()) + list(transformer.parameters()), 
        lr=LEARNING_RATE,
        weight_decay=1e-4 
    )

    # 4. BALANCED WEIGHTS
    class_weights = torch.tensor([1.0, 3.0]).to(DEVICE) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0.0

    # --- PREPARE LOG FILE ---
    # Create the file and write headers
    with open(LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_acc', 'val_f1'])

    print("\nEpoch | Train Loss | Val Acc | Val F1 | Status")
    print("-" * 55)

    for epoch in range(EPOCHS):
        # TRAIN
        spatial.train(); clinical.train(); fusion.train(); transformer.train()
        train_loss = 0
        
        for images, clin_data, labels in train_loader:
            images, clin_data, labels = images.to(DEVICE), clin_data.to(DEVICE), labels.to(DEVICE)
            
            b, v, c, h, w = images.shape
            spatial_feats = spatial(images.view(-1, c, h, w))
            clin_feats = clinical(clin_data).unsqueeze(1).expand(-1, v, -1).reshape(-1, 128)
            fused = fusion(spatial_feats, clin_feats).view(b, v, -1)
            
            _, logits = transformer(fused)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # VALIDATION
        spatial.eval(); clinical.eval(); fusion.eval(); transformer.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for images, clin_data, labels in val_loader:
                images, clin_data, labels = images.to(DEVICE), clin_data.to(DEVICE), labels.to(DEVICE)
                b, v, c, h, w = images.shape
                spatial_feats = spatial(images.view(-1, c, h, w))
                clin_feats = clinical(clin_data).unsqueeze(1).expand(-1, v, -1).reshape(-1, 128)
                fused = fusion(spatial_feats, clin_feats).view(b, v, -1)
                
                _, logits = transformer(fused)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        # METRICS
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        avg_loss = train_loss / len(train_loader)
        
        # --- SAVE TO CSV ---
        with open(LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, val_acc, val_f1])
        # -------------------

        status = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            status = "★ SAVED"
            torch.save({
                'spatial': spatial.state_dict(),
                'clinical': clinical.state_dict(),
                'fusion': fusion.state_dict(),
                'transformer': transformer.state_dict(),
                'f1_score': val_f1,
                'epoch': epoch
            }, SAVE_PATH)

        print(f"{epoch+1:03d}   | {avg_loss:.4f}     | {val_acc:.2f}    | {val_f1:.4f} | {status}")

    print(f"\n[DONE] Log saved to: {LOG_PATH}")

if __name__ == "__main__":
    train_rescue_model()