import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# SETUP PATHS
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
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_forecasting_model.pth')

# --- TOGGLE DEBUG PRINTING ---
DEBUG_TENSORS = False # Set to True to print all tensor shapes/outputs for Batch 1
# -----------------------------

def print_tensor_info(name, tensor):
    if DEBUG_TENSORS:
        print(f"       [DEBUG] {name}: Shape {tensor.shape}, Min {tensor.min().item():.3f}, Max {tensor.max().item():.3f}")

def train_forecaster():
    print("="*60)
    print("TRAINING PROTOCOL: TRUE TIME-AWARE FORECASTING")
    print(f"Device: {DEVICE}")
    print("="*60)

    # 1. LOAD DATA
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_train_images1.pt') 
    csv_path = os.path.join(processed_dir, 'grape_train.csv')          

    dataset = GrapeDataset(tensor_path, csv_path)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # 2. INITIALIZE MODELS
    clin_dim = dataset.get_clinical_dim()
    out_dim = dataset.get_output_dim()

    print(f"   [INFO] Clinical Inputs: {clin_dim} features")
    print(f"   [INFO] Forecasting Output: {out_dim} VF Points (V_t -> V_t+1)")

    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clin_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256, output_dim=out_dim).to(DEVICE)

    # 3. OPTIMIZER & LOSS
    optimizer = optim.Adam(
        list(spatial.parameters()) + list(clinical.parameters()) + 
        list(fusion.parameters()) + list(transformer.parameters()), 
        lr=LEARNING_RATE
    )
    
    criterion = nn.MSELoss()      # Loss (MSE)
    metric_mae = nn.L1Loss()      # Metric (MAE)

    # 4. TRAINING LOOP
    best_val_mae = float('inf')
    
    print("\n[STEP 3] Starting Forecasting Training...")
    print("         (Time shift applied. MAE will be higher than before, but correct.)")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- TRAIN ---
        spatial.train(); clinical.train(); fusion.train(); transformer.train()
        train_loss = 0
        train_mae = 0
        
        print(f"\n   > Epoch {epoch+1} Started...")
        
        for batch_idx, (images, clin_data, targets) in enumerate(train_loader):
            images, clin_data, targets = images.to(DEVICE), clin_data.to(DEVICE), targets.to(DEVICE)
            b, s, c, h, w = images.shape
            
            optimizer.zero_grad()
            
            # --- DEBUGGING START ---
            if DEBUG_TENSORS and batch_idx == 0:
                print("\n   --- DEBUG OUTPUT (Batch 1, Forward Pass) ---")
                print_tensor_info("Input Images", images)
                print_tensor_info("Input Clinical", clin_data)
                print_tensor_info("Target VF", targets)
            # --- DEBUGGING END ---

            # 1. Spatial Encoder (CNN)
            spatial_feats = spatial(images.view(-1, c, h, w))
            print_tensor_info("1. Spatial Feats (512)", spatial_feats)

            # 2. Clinical Encoder (MLP)
            clin_feats = clinical(clin_data.view(-1, clin_dim))
            print_tensor_info("2. Clinical Feats (128)", clin_feats)

            # 3. Fusion Layer
            fused = fusion(spatial_feats.view(b, s, -1), clin_feats.view(b, s, -1))
            print_tensor_info("3. Fused Vector (256)", fused)

            # 4. Transformer
            predictions = transformer(fused)
            print_tensor_info("4. Predictions (61)", predictions)

            # Mask zero-padded visits and single-visit patients
            # Loss is ONLY computed where the target is NOT all zero (i.e., where there was a next visit)
            mask = (targets.abs().sum(dim=2) > 0)
            
            if mask.sum() > 0:
                loss = criterion(predictions[mask], targets[mask])
                mae = metric_mae(predictions[mask], targets[mask])
            else:
                # If the entire batch is invalid (unlikely), skip backward pass
                continue 
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += mae.item()
            
            # --- HEARTBEAT PRINT ---
            if (batch_idx + 1) % 5 == 0:
                print(f"     [Batch {batch_idx+1}/{len(train_loader)}] Current MAE: {mae.item():.4f}")

        # --- VALIDATE ---
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for images, clin_data, targets in val_loader:
                images, clin_data, targets = images.to(DEVICE), clin_data.to(DEVICE), targets.to(DEVICE)
                b, s, c, h, w = images.shape
                
                spatial_feats = spatial(images.view(-1, c, h, w))
                clin_feats = clinical(clin_data.view(-1, clin_dim))
                fused = fusion(spatial_feats.view(b,s,-1), clin_feats.view(b,s,-1))
                preds = transformer(fused)
                
                mask = (targets.abs().sum(dim=2) > 0)
                if mask.sum() > 0:
                    loss = criterion(preds[mask], targets[mask])
                    mae = metric_mae(preds[mask], targets[mask])
                else:
                    continue
                    
                val_loss += loss.item()
                val_mae += mae.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        avg_val_mae = val_mae / len(val_loader)
        epoch_time = time.time() - start_time
        
        # Print all required metrics
        print(f"   > Epoch {epoch+1} Finished in {epoch_time:.0f}s | Train MSE: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f}")

        # Save Efficient Weights (Best Val MAE)
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save({
                'spatial': spatial.state_dict(),
                'clinical': clinical.state_dict(),
                'fusion': fusion.state_dict(),
                'transformer': transformer.state_dict(),
                'epoch': epoch,
                'val_mae': best_val_mae
            }, SAVE_PATH)
            print(f"    >>> SAVED (New Best Val MAE: {best_val_mae:.4f})")

    print("\nTRAINING COMPLETE.")

if __name__ == "__main__":
    train_forecaster()
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os
import numpy as np
import time
import warnings

# Filter out FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# SETUP PATHS
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
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_forecasting_model.pth')

def train_forecaster():
    print("="*60)
    print("TRAINING PROTOCOL: VF FORECASTING (REGRESSION)")
    print(f"Device: {DEVICE}")
    print("="*60)

    # 1. LOAD DATA (Updated Filenames)
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_train_images1.pt') 
    csv_path = os.path.join(processed_dir, 'grape_train.csv')          

    dataset = GrapeDataset(tensor_path, csv_path)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # 2. INITIALIZE MODELS
    clin_dim = dataset.get_clinical_dim()
    out_dim = dataset.get_output_dim()

    print(f"   [INFO] Clinical Inputs: {clin_dim} features")
    print(f"   [INFO] Forecasting Output: {out_dim} VF Points")

    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clin_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256, output_dim=out_dim).to(DEVICE)

    # 3. OPTIMIZER & LOSS
    optimizer = optim.Adam(
        list(spatial.parameters()) + list(clinical.parameters()) + 
        list(fusion.parameters()) + list(transformer.parameters()), 
        lr=LEARNING_RATE
    )
    
    criterion = nn.MSELoss()      # Loss
    metric_mae = nn.L1Loss()      # Metric (MAE)

    # 4. TRAINING LOOP
    best_val_mae = float('inf')
    
    print("\n[STEP 3] Starting Forecasting Training...")
    print("         (Prints progress every 5 batches...)")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- TRAIN ---
        spatial.train(); clinical.train(); fusion.train(); transformer.train()
        train_loss = 0
        train_mae = 0
        
        print(f"\n   > Epoch {epoch+1} Started...")
        
        for batch_idx, (images, clin_data, targets) in enumerate(train_loader):
            images, clin_data, targets = images.to(DEVICE), clin_data.to(DEVICE), targets.to(DEVICE)
            b, v, c, h, w = images.shape
            
            # Forward Pass
            spatial_feats = spatial(images.view(-1, c, h, w))
            clin_feats = clinical(clin_data.view(-1, clin_dim))
            fused = fusion(spatial_feats.view(b, v, -1), clin_feats.view(b, v, -1))
            predictions = transformer(fused)
            
            # Mask zero-padded visits
            mask = (targets.abs().sum(dim=2) > 0)
            
            if mask.sum() > 0:
                loss = criterion(predictions[mask], targets[mask])
                mae = metric_mae(predictions[mask], targets[mask])
            else:
                loss = criterion(predictions, targets)
                mae = metric_mae(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += mae.item()
            
            # --- HEARTBEAT PRINT ---
            if (batch_idx + 1) % 5 == 0:
                print(f"     [Batch {batch_idx+1}/{len(train_loader)}] Current MAE: {mae.item():.4f}")

        # --- VALIDATE ---
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for images, clin_data, targets in val_loader:
                images, clin_data, targets = images.to(DEVICE), clin_data.to(DEVICE), targets.to(DEVICE)
                b, v, c, h, w = images.shape
                
                spatial_feats = spatial(images.view(-1, c, h, w))
                clin_feats = clinical(clin_data.view(-1, clin_dim))
                fused = fusion(spatial_feats.view(b,v,-1), clin_feats.view(b,v,-1))
                preds = transformer(fused)
                
                mask = (targets.abs().sum(dim=2) > 0)
                if mask.sum() > 0:
                    loss = criterion(preds[mask], targets[mask])
                    mae = metric_mae(preds[mask], targets[mask])
                else:
                    loss = criterion(preds, targets)
                    mae = metric_mae(preds, targets)
                    
                val_loss += loss.item()
                val_mae += mae.item()

        avg_loss = train_loss / len(train_loader)
        avg_mae = val_mae / len(val_loader)
        epoch_time = time.time() - start_time
        
        print(f"   > Epoch {epoch+1} Finished in {epoch_time:.0f}s | Train MSE: {avg_loss:.4f} | Val MAE: {avg_mae:.4f}")

        if avg_mae < best_val_mae:
            best_val_mae = avg_mae
            torch.save({
                'spatial': spatial.state_dict(),
                'clinical': clinical.state_dict(),
                'fusion': fusion.state_dict(),
                'transformer': transformer.state_dict(),
                'epoch': epoch,
                'val_mae': best_val_mae
            }, SAVE_PATH)
            print(f"    >>> SAVED (New Best MAE: {best_val_mae:.4f})")

    print("\nTRAINING COMPLETE.")

if __name__ == "__main__":
    train_forecaster()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os
import numpy as np

import warnings
# Filter out the specific PyTorch warning
warnings.filterwarnings("ignore", category=FutureWarning)

# SETUP PATHS
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
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_forecasting_model.pth')

def train_forecaster():
    print("="*60)
    print("TRAINING PROTOCOL: VF FORECASTING (REGRESSION)")
    print(f"Device: {DEVICE}")
    print("="*60)

    # 1. LOAD DATA (Updated Filenames)
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_train_images1.pt') # Using plural as per previous log
    csv_path = os.path.join(processed_dir, 'grape_train.csv')          # Using the name you provided

    dataset = GrapeDataset(tensor_path, csv_path)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # 2. INITIALIZE MODELS
    clin_dim = dataset.get_clinical_dim()
    out_dim = dataset.get_output_dim()

    print(f"   [INFO] Clinical Inputs: {clin_dim} features")
    print(f"   [INFO] Forecasting Output: {out_dim} VF Points")

    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clin_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256, output_dim=out_dim).to(DEVICE)

    # 3. OPTIMIZER & LOSS
    optimizer = optim.Adam(
        list(spatial.parameters()) + list(clinical.parameters()) + 
        list(fusion.parameters()) + list(transformer.parameters()), 
        lr=LEARNING_RATE
    )
    
    criterion = nn.MSELoss()      # Loss
    metric_mae = nn.L1Loss()      # Metric (MAE)

    # 4. TRAINING LOOP
    best_val_mae = float('inf')
    
    print("\n[STEP 3] Starting Forecasting Training...")
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        spatial.train(); clinical.train(); fusion.train(); transformer.train()
        train_loss = 0
        train_mae = 0
        
        for images, clin_data, targets in train_loader:
            images, clin_data, targets = images.to(DEVICE), clin_data.to(DEVICE), targets.to(DEVICE)
            b, v, c, h, w = images.shape
            
            spatial_feats = spatial(images.view(-1, c, h, w))
            clin_feats = clinical(clin_data.view(-1, clin_dim))
            fused = fusion(spatial_feats.view(b, v, -1), clin_feats.view(b, v, -1))
            predictions = transformer(fused)
            
            # Mask zero-padded visits
            mask = (targets.abs().sum(dim=2) > 0)
            
            if mask.sum() > 0:
                loss = criterion(predictions[mask], targets[mask])
                mae = metric_mae(predictions[mask], targets[mask])
            else:
                loss = criterion(predictions, targets)
                mae = metric_mae(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += mae.item()

        # --- VALIDATE ---
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for images, clin_data, targets in val_loader:
                images, clin_data, targets = images.to(DEVICE), clin_data.to(DEVICE), targets.to(DEVICE)
                b, v, c, h, w = images.shape
                
                spatial_feats = spatial(images.view(-1, c, h, w))
                clin_feats = clinical(clin_data.view(-1, clin_dim))
                fused = fusion(spatial_feats.view(b,v,-1), clin_feats.view(b,v,-1))
                preds = transformer(fused)
                
                mask = (targets.abs().sum(dim=2) > 0)
                if mask.sum() > 0:
                    loss = criterion(preds[mask], targets[mask])
                    mae = metric_mae(preds[mask], targets[mask])
                else:
                    loss = criterion(preds, targets)
                    mae = metric_mae(preds, targets)
                    
                val_loss += loss.item()
                val_mae += mae.item()

        avg_loss = train_loss / len(train_loader)
        avg_mae = val_mae / len(val_loader)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss (MSE): {avg_loss:.4f} | Val MAE: {avg_mae:.4f}")

        if avg_mae < best_val_mae:
            best_val_mae = avg_mae
            torch.save({
                'spatial': spatial.state_dict(),
                'clinical': clinical.state_dict(),
                'fusion': fusion.state_dict(),
                'transformer': transformer.state_dict(),
                'epoch': epoch,
                'val_mae': best_val_mae
            }, SAVE_PATH)
            print(f"    >>> SAVED (New Best MAE: {best_val_mae:.4f})")

    print("\nTRAINING COMPLETE.")

if __name__ == "__main__":
    train_forecaster()
"""