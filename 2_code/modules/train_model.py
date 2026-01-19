import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import sys

# --- 1. Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'modules'))

# Import the new architecture and your loader
from mmi_mst_former import MMI_MST_Former
from dataset_loader import GrapeDataset

def train_model():
    # --- 2. Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    batch_size = 4
    learning_rate = 3e-4
    
    # Data Paths (Adjusted to your project structure)
    data_dir = os.path.join(SCRIPT_DIR, '..', '1_data', 'processed')
    tensor_path = os.path.join(data_dir, 'grape_test_images1.pt')
    csv_path = os.path.join(data_dir, 'grape_test.csv')

    # --- 3. Initialize Model ---
    # Parameters matched to your QUICK_START.md
    model = MMI_MST_Former(
        img_size=224,
        patch_size=16,
        num_scales=3,
        d_model=256,
        num_heads=8,
        num_clinical_features=67, # Matches your 61 VF + 6 Clinical
        num_vf_points=61
    ).to(device)

    # --- 4. Load Data ---
    dataset = GrapeDataset(tensor_path, csv_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # --- 5. Optimizer & Loss ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss() # For 61-point VF regression

    print(f"🚀 Training started on {device}...")

    # --- 6. Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, clin, tar_vf in pbar:
            imgs, clin, tar_vf = imgs.to(device), clin.to(device), tar_vf.to(device)
            
            # Extract Time Intervals for the Time-Aware Attention
            # Assuming 'Interval_Norm' is index 5 in your clinical features
            time_intervals = clin[:, :, 5] 

            optimizer.zero_grad()
            
            # Forward pass (return_attention=False during training for speed)
            preds = model(imgs, clin, time_intervals, return_attention=False)
            
            # Loss: Comparing predicted VF (last visit) to target VF (last visit)
            loss = criterion(preds, tar_vf[:, -1, :])
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- 7. Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, clin, tar_vf in val_loader:
                imgs, clin, tar_vf = imgs.to(device), clin.to(device), tar_vf.to(device)
                time_intervals = clin[:, :, 5]
                preds = model(imgs, clin, time_intervals)
                val_loss += criterion(preds, tar_vf[:, -1, :]).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_mmi_mst_model.pth")
            print("⭐ New Best Model Saved!")

if __name__ == "__main__":
    train_model()