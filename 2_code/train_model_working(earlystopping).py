import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import os
import sys

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'modules'))

from modules.mmi_mst_former import MMI_MST_Former
from modules.dataset_loader import GrapeDataset


# ============================================
# BALANCED SOFTMAX CROSS-ENTROPY LOSS (Paper Eq. 6)
# ============================================

class BalancedSoftmaxCrossEntropy(nn.Module):
    """
    Temperature-controlled Balanced Softmax Cross-Entropy loss.
    Addresses class imbalance in long-tailed distributions.
    Paper Eq. 6: L = -Σ_i y_i log(n_i^τ * e^y_i / Σ_j n_j^τ * e^y_j)
    """
    def __init__(self, num_classes, class_counts=None, temperature=2.0):
        """
        Args:
            num_classes: number of classes
            class_counts: [num_classes] tensor with sample counts per class
            temperature: τ parameter controlling class weight scaling
        """
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        
        if class_counts is not None:
            # Compute class weights based on counts
            class_counts = torch.tensor(class_counts, dtype=torch.float32)
            self.register_buffer('class_weights', class_counts / class_counts.sum())
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, num_classes] model outputs
            targets: [B] class labels
        
        Returns:
            loss: scalar
        """
        if self.class_weights is not None:
            # Apply temperature scaling to class weights
            scaled_weights = self.class_weights ** self.temperature
            # Normalize
            scaled_weights = scaled_weights / scaled_weights.sum()
            
            # Compute loss with scaling
            log_softmax = torch.log_softmax(logits, dim=1)
            
            # Get loss for target class
            loss = torch.zeros(logits.shape[0], device=logits.device)
            for i, target in enumerate(targets):
                loss[i] = -log_softmax[i, target] * scaled_weights[target]
            
            return loss.mean()
        else:
            # Standard cross-entropy if no class weights
            return nn.functional.cross_entropy(logits, targets)


# ============================================
# MAIN TRAINING LOOP
# ============================================

def train_model():
    """
    Complete training pipeline with proper indexing and loss functions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")
    
    # Data paths
    data_dir = os.path.join(SCRIPT_DIR, '..', '1_data', 'processed')
    tensor_path = os.path.join(data_dir, 'grape_test_images1.pt')
    csv_path = os.path.join(data_dir, 'grape_test.csv')

    # ============================================
    # MODEL INITIALIZATION
    # ============================================
    print("\n📊 Initializing MMI-MST-Former Model...")
    model = MMI_MST_Former(
        img_size=224,
        patch_size=16,
        num_scales=3,
        d_model=256,
        num_heads=8,
        num_clinical_features=67,
        num_vf_points=61,
        dropout=0.1,
        alpha=0.5,
        beta=0.5
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # ============================================
    # DATA LOADING
    # ============================================
    print("\n📂 Loading dataset...")
    dataset = GrapeDataset(
        tensor_path=tensor_path,
        csv_path=csv_path,
        max_seq_len=9,
        overlap_stride=1
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples: {len(val_ds)}")

    # ============================================
    # LOSS & OPTIMIZER
    # ============================================
    # MSE Loss for VF point regression
    criterion = nn.MSELoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # ============================================
    # TRAINING CONFIGURATION
    # ============================================
    num_epochs = 100
    best_val_mae = float('inf')
    patience = 20
    patience_counter = 0

    print(f"\n🚀 Starting Training ({num_epochs} epochs)...")
    print(f"   Batch size: 4")
    print(f"   Learning rate: 3e-4")
    print(f"   Loss: MSELoss")

    # ============================================
    # TRAINING LOOP
    # ============================================
    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            images = batch['images'].to(device)  # [B, L, 3, 224, 224]
            clinical = batch['clinical'].to(device)  # [B, L, 67]
            time_intervals = batch['time_intervals'].to(device)  # [B, L]
            target_vf = batch['target_vf'].to(device)  # [B, L, 61]
            valid_len = batch['valid_len'].to(device)  # [B]
            
            optimizer.zero_grad()
            
            # Forward pass
            vf_preds = model(images, clinical, time_intervals)  # [B, 61]
            
            # Extract targets at correct positions
            # For simplicity, use LAST VALID TIMESTEP's VF as target
            batch_indices = torch.arange(images.size(0), device=device)
            last_valid_idx = (valid_len - 1).long()  # -1 because target is shifted
            last_valid_idx = torch.clamp(last_valid_idx, min=0, max=target_vf.shape[1]-1)
            
            target_at_last = target_vf[batch_indices, last_valid_idx]  # [B, 61]
            
            # Compute loss
            loss = criterion(vf_preds, target_at_last)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                mae = torch.abs(vf_preds - target_at_last).mean().item()
            
            train_loss += loss.item() * images.size(0)
            train_mae += mae * images.size(0)
            train_count += images.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae:.4f}'
            })
        
        train_loss /= train_count
        train_mae /= train_count
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0
        val_count = 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in pbar_val:
                images = batch['images'].to(device)
                clinical = batch['clinical'].to(device)
                time_intervals = batch['time_intervals'].to(device)
                target_vf = batch['target_vf'].to(device)
                valid_len = batch['valid_len'].to(device)
                
                # Forward pass
                vf_preds = model(images, clinical, time_intervals)
                
                # Extract targets
                batch_indices = torch.arange(images.size(0), device=device)
                last_valid_idx = (valid_len - 1).long()
                last_valid_idx = torch.clamp(last_valid_idx, min=0, max=target_vf.shape[1]-1)
                target_at_last = target_vf[batch_indices, last_valid_idx]
                
                # Metrics
                loss = criterion(vf_preds, target_at_last)
                mae = torch.abs(vf_preds - target_at_last).mean().item()
                rmse = torch.sqrt(torch.pow(vf_preds - target_at_last, 2).mean()).item()
                
                val_loss += loss.item() * images.size(0)
                val_mae += mae * images.size(0)
                val_rmse += rmse * images.size(0)
                val_count += images.size(0)
            
            val_loss /= val_count
            val_mae /= val_count
            val_rmse /= val_count
        
        # Learning rate scheduling
        scheduler.step()
        
        # --- LOGGING & CHECKPOINTING ---
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss:     {train_loss:.4f}  |  MAE: {train_mae:.4f}")
        print(f"   Val Loss:       {val_loss:.4f}  |  MAE: {val_mae:.4f}  |  RMSE: {val_rmse:.4f}")
        print(f"   Learning Rate:  {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model based on validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "best_mmi_mst_model.pth")
            print(f"   ✅ New best MAE! Model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   ⚠️ No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    torch.save(model.state_dict(), "final_mmi_mst_model.pth")
    print("\n✅ Training Complete!")
    print(f"   Best validation MAE: {best_val_mae:.4f}")


if __name__ == "__main__":
    train_model()
'''
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# --- STEP 1: RESOLVE FOLDER STRUCTURE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # 2_code
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) # Project Root

# Add /modules/ to path for imports
sys.path.append(os.path.join(SCRIPT_DIR, 'modules'))

# Define DATA PATHS based on your Audit Script location
DATA_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
TENSOR_PATH = os.path.join(DATA_DIR, 'grape_test_images1.pt') # From your audit script
CSV_PATH = os.path.join(DATA_DIR, 'grape_test.csv')           # From your audit script

# --- STEP 2: IMPORTS ---
try:
    from spatial_encoder import SpatialEncoder
    from clinical_encoder import ClinicalEncoder
    from fusion_layer import FusionLayer
    from mst_transformer import MSTFormer
    from dataset_loader import GrapeDataset
except ImportError as e:
    print(f"[ERROR] Import failed: {e}. Check your /modules/ folder.")
    sys.exit()

def train_one_epoch(models, loader, optimizer, criterion, device):
    for m in models.values(): m.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        imgs, clin, tar_vf = [b.to(device) for b in batch]
        optimizer.zero_grad()

        # Scale 1: Hierarchical Grid Processing
        visit_features = []
        for i in range(imgs.shape[1]):
            s_map = models['spatial'](imgs[:, i]) # 7x7 Grid
            c_emb = models['clinical'](clin[:, i])
            f_grid = models['fusion'](s_map, c_emb)
            visit_features.append(f_grid.unsqueeze(1))
        
        sequence = torch.cat(visit_features, dim=1) # [B, 9, 256, 7, 7]
        pred_vf, _ = models['mst'](sequence) # Hierarchical MST logic

        # Target is the next visit (index -1 of sequence)
        loss = criterion(pred_vf, tar_vf[:, -1, :])
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    return running_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Hierarchical MST-Former Init ---")
    print(f"Using Data from: {DATA_DIR}")

    if not os.path.exists(TENSOR_PATH):
        print(f"CRITICAL ERROR: {TENSOR_PATH} not found!")
        return

    # CONFIG: 6 Clinical + 61 VF = 67
    models = {
        'spatial': SpatialEncoder().to(device),
        'clinical': ClinicalEncoder(input_dim=67).to(device),
        'fusion': FusionLayer().to(device),
        'mst': MSTFormer().to(device)
    }

    dataset = GrapeDataset(TENSOR_PATH, CSV_PATH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    optimizer = optim.Adam([p for m in models.values() for p in m.parameters()], lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(50):
        loss = train_one_epoch(models, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")
        torch.save({k: v.state_dict() for k, v in models.items()}, "best_glaucoma_model.pth")

if __name__ == "__main__":
    main()
'''
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import math

# --- 1. SETUP DEBUGGING ---
print("----------------------------------------------------------------")
print("   STARTING GRAVITY (GLAUCOMA) FORECASTING TRAINING ENGINE      ")
print("----------------------------------------------------------------")

# Add the current directory to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- IMPORT MODULES ---
try:
    from modules.dataset_loader import GrapeDataset
    from modules.spatial_encoder import SpatialEncoder
    from modules.clinical_encoder import ClinicalEncoder
    from modules.fusion_layer import FusionLayer
    from modules.mst_transformer import MSTFormer
    print("[OK] All Modules Imported Successfully.")
except ImportError as e:
    print(f"\n[CRITICAL ERROR] Import Failed: {e}")
    print("Ensure you have 'dataset_loader.py', 'mst_transformer.py', etc. inside the 'modules' folder.\n")
    sys.exit(1)

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Processing Device: {DEVICE}")

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 50

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Check for data in parallel folder
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', '1_data', 'processed'))
SAVE_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', '3_results', 'models'))
os.makedirs(SAVE_DIR, exist_ok=True)

def calculate_metrics(pred, target):
    """
    Helper to calculate MAE and RMSE for the Visual Field (61 points).
    """
    mse = nn.MSELoss()(pred, target)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(pred - target))
    return mae.item(), rmse.item()

def train():
    # --- 1. DATA LOADING ---
    train_tensor_path = os.path.join(DATA_DIR, 'grape_train_images.pt')
    train_csv_path = os.path.join(DATA_DIR, 'grape_train.csv')

    print(f"[INFO] Looking for Data at: {DATA_DIR}")
    
    if not os.path.exists(train_tensor_path) or not os.path.exists(train_csv_path):
        print(f"\n[ERROR] Data NOT found!")
        print(f"   Missing: {train_tensor_path}")
        print("   Please check your '1_data/processed' folder location.")
        return

    print("[INFO] Loading Dataset... (This might take a moment)")
    train_dataset = GrapeDataset(train_tensor_path, train_csv_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[OK] Data Loaded. Total Patients: {len(train_dataset)}")

    # --- 2. INITIALIZE MODELS ---
    print("[INFO] Initializing MST-Former Architecture...")
    spatial_enc = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical_enc = ClinicalEncoder(input_dim=train_dataset.get_clinical_dim(), output_dim=128).to(DEVICE)
    fusion_layer = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    mst_former = MSTFormer(input_dim=256, output_dim=train_dataset.get_output_dim()).to(DEVICE)

    # --- 3. OPTIMIZER ---
    all_params = list(spatial_enc.parameters()) + \
                 list(clinical_enc.parameters()) + \
                 list(fusion_layer.parameters()) + \
                 list(mst_former.parameters())
                 
    optimizer = optim.Adam(all_params, lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()

    # --- 4. TRAINING LOOP ---
    print(f"\n[INFO] Starting Training Loop for {EPOCHS} Epochs...")
    best_rmse = float('inf')

    for epoch in range(EPOCHS):
        spatial_enc.train(); clinical_enc.train(); fusion_layer.train(); mst_former.train()
        
        running_loss = 0.0
        running_mae_vf = 0.0
        running_rmse_vf = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack (Handle 5 items)
            images, clin_data, time_steps, target_vf, target_gap = batch_data
            
            # Move to Device
            images = images.to(DEVICE)          # [B, Seq, 3, 224, 224]
            clin_data = clin_data.to(DEVICE)    # [B, Seq, Feats]
            time_steps = time_steps.to(DEVICE)  # [B, Seq]
            target_vf = target_vf.to(DEVICE)    # [B, Seq, 61]
            target_gap = target_gap.to(DEVICE)  # [B, Seq, 1]
            
            optimizer.zero_grad()
            
            # -- Forward Pass --
            b, s, c, h, w = images.shape
            
            # Spatial
            img_flat = images.view(b * s, c, h, w)
            spatial_emb = spatial_enc(img_flat).view(b, s, -1)
            
            # Clinical
            clin_flat = clin_data.view(b * s, -1)
            clinical_emb = clinical_enc(clin_flat).view(b, s, -1)
            
            # Fusion
            fused_emb = fusion_layer(spatial_emb, clinical_emb)
            
            # MST-Former
            pred_vf, pred_gap = mst_former(fused_emb, time_steps)
            
            # -- Calculate Loss --
            final_target_vf = target_vf[:, -1, :] 
            final_target_gap = target_gap[:, -1, :]
            
            loss_vf = criterion_mse(pred_vf, final_target_vf)
            loss_time = criterion_mse(pred_gap, final_target_gap)
            
            # Weighted Loss (Focus mainly on VF accuracy)
            loss = loss_vf + (0.2 * loss_time)
            
            loss.backward()
            optimizer.step()
            
            # -- Metrics Tracking --
            running_loss += loss.item()
            
            mae, rmse = calculate_metrics(pred_vf, final_target_vf)
            running_mae_vf += mae
            running_rmse_vf += rmse
            
        # -- Epoch Summary --
        avg_loss = running_loss / len(train_loader)
        avg_mae = running_mae_vf / len(train_loader)
        avg_rmse = running_rmse_vf / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | VF MAE: {avg_mae:.4f} dB | VF RMSE: {avg_rmse:.4f} dB")
        
        # -- Save Best Model --
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            save_path = os.path.join(SAVE_DIR, 'best_mst_model.pth')
            torch.save({
                'spatial': spatial_enc.state_dict(),
                'clinical': clinical_enc.state_dict(),
                'fusion': fusion_layer.state_dict(),
                'mst_former': mst_former.state_dict(),
                'epoch': epoch,
                'rmse': best_rmse
            }, save_path)
            print(f"   >>> New Best Model Saved! (RMSE: {best_rmse:.4f})")

    print("\n----------------------------------------------------------------")
    print("   TRAINING COMPLETE")
    print(f"   Best Model Saved at: {os.path.join(SAVE_DIR, 'best_mst_model.pth')}")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    train()
'''