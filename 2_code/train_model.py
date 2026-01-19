import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys

# --- 1. Path & Module Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'modules'))

from modules.mmi_mst_former import MMI_MST_Former
from modules.dataset_loader import create_dataloaders_fixed 

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Hardware: {device}")

    # Paths
    data_dir = os.path.join(SCRIPT_DIR, '..', '1_data', 'processed')
    tensor_path = os.path.join(data_dir, 'grape_test_images1.pt')
    csv_path = os.path.join(data_dir, 'grape_test.csv')

    # --- 2. Correct Dataloading (Fixed Split) ---
    # 70% Train, 15% Val, 15% Test. Total must be 1.0
    train_loader, val_loader, _ = create_dataloaders_fixed(
        tensor_path=tensor_path,
        csv_path=csv_path,
        batch_size=4,
        train_split=0.7, 
        val_split=0.15,
        max_seq_len=9
    )

    # --- 3. Model Initialization ---
    model = MMI_MST_Former(num_clinical_features=67, num_vf_points=61).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_mae = float('inf')
    epochs = 100

    print(f"🚀 Training: Clamping & Padding Protection Active.")

    for epoch in range(epochs):
        model.train()
        train_abs_err, train_count = 0.0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            images = batch['images'].to(device)
            clinical = batch['clinical'].to(device)
            time_intervals = batch['time_intervals'].to(device)
            target_vf = batch['target_vf'].to(device)
            valid_len = batch['valid_len'].to(device)

            optimizer.zero_grad()
            preds = model(images, clinical, time_intervals) # [B, 61]
            
            # --- THE SAFETY FIX: INDEX CLAMPING ---
            batch_size_curr = images.size(0)
            b_idx = torch.arange(batch_size_curr, device=device)
            
            # Clamp to prevent IndexError and ensure we stay within the valid window
            # We subtract 1 because valid_len is 1-based count, index is 0-based
            target_idx = torch.clamp(valid_len - 1, min=0, max=target_vf.shape[1]-1).long()
            
            batch_preds = preds 
            batch_targets = target_vf[b_idx, target_idx]

            # Loss calculation only on real forecasting pairs
            loss = criterion(batch_preds, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                train_abs_err += torch.abs(batch_preds - batch_targets).sum().item()
                train_count += batch_targets.numel()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # --- 4. VALIDATION PHASE ---
        model.eval()
        val_abs_err, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                clinical = batch['clinical'].to(device)
                time_intervals = batch['time_intervals'].to(device)
                target_vf = batch['target_vf'].to(device)
                valid_len = batch['valid_len'].to(device)

                preds = model(images, clinical, time_intervals)
                
                b_idx = torch.arange(images.size(0), device=device)
                target_idx = torch.clamp(valid_len - 1, min=0, max=target_vf.shape[1]-1).long()
                
                v_preds = preds
                v_targets = target_vf[b_idx, target_idx]
                
                val_abs_err += torch.abs(v_preds - v_targets).sum().item()
                val_count += v_targets.numel()

        # Metrics
        epoch_train_mae = train_abs_err / (train_count + 1e-8)
        epoch_val_mae = val_abs_err / (val_count + 1e-8)
        scheduler.step(epoch_val_mae)

        print(f"\n📈 Epoch {epoch+1}: Train MAE: {epoch_train_mae:.4f} | Val MAE: {epoch_val_mae:.4f}")

        if epoch_val_mae < best_val_mae:
            best_val_mae = epoch_val_mae
            torch.save(model.state_dict(), "best_mmi_mst_model.pth")
            print(f"⭐ New Best MAE! Model Saved.")

    print("\n✅ Training Complete.")

if __name__ == "__main__":
    train_model()