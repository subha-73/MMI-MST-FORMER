import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm
import warnings
# --- FIX 1: ADD REQUIRED SKLEARN IMPORTS ---
from sklearn.metrics import mean_absolute_error, r2_score 

# Add this line to suppress the specific FutureWarning globally (as requested)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. SETUP PATHS and IMPORTS (CLEANED & FIXED)
# ==========================================

# Determine the project root for output directories (relative to the 2_code folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) 

# --- CRITICAL FIX: Add the '2_code' directory to sys.path ---
# This ensures that Python can find 'modules.mmi_sst_former' and 'data.dataset_loader'.
CODE_DIR = os.path.join(ROOT_DIR, '2_code')
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

try:
    # Now, import using the folder names within 2_code
    from modules.mmi_sst_former import MMI_SST_Former
    from data.dataset_loader import get_data_loaders
    
    # Also ensure the individual encoder modules are available in the module path
    if os.path.join(CODE_DIR, 'modules') not in sys.path:
        sys.path.append(os.path.join(CODE_DIR, 'modules'))

except ImportError as e:
    print(f"[CRITICAL ERROR] Failed to import core modules. Please ensure __init__.py files exist. Error: {e}")
    sys.exit(1)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. HYPERPARAMETERS AND CONFIGURATION
# ==========================================

class Config:
    # Data & I/O
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    OUTPUT_DIR = os.path.join(ROOT_DIR, '3_output', 'checkpoints')
    
    # Training
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Model (MMI-SST-Former)
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_HEADS = 8
    DROPOUT_RATE = 0.1
    FUSED_DIM = 256

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
writer = SummaryWriter(os.path.join(ROOT_DIR, '3_output', 'runs'))

# ==========================================
# 3. TRAINING STEP FUNCTION (FULLY FIXED)
# ==========================================

def train_one_epoch(model, data_loader, optimizer, criterion):
    """Handles the forward pass, loss calculation, and backpropagation for one epoch."""
    model.train()
    total_valid_predictions = 0
    total_loss_sum = 0
    
    # We pass 'i' (batch index) through enumerate
    for i, batch in enumerate(tqdm(data_loader, desc="Training")):
        # 1. Prepare Data
        images = batch['image_seq'].to(DEVICE)
        clinical = batch['clinical_seq'].to(DEVICE)
        seq_mask = batch['seq_mask'].to(DEVICE)
        target_vf = batch['target_vf'].to(DEVICE)
        target_mask = batch['target_mask'].to(DEVICE) 

        # 2. Forward Pass
        optimizer.zero_grad()
        predictions = model(images, clinical, seq_mask)

        # 3. Calculate Loss (Only on Valid Steps)
        predictions_flat = predictions.view(-1)
        target_vf_flat = target_vf.view(-1)
        
        mask_expanded = target_mask.unsqueeze(-1).repeat(1, 1, target_vf.shape[-1]).view(-1)
        
        valid_predictions = predictions_flat[mask_expanded == 1]
        valid_targets = target_vf_flat[mask_expanded == 1]
        
        if len(valid_predictions) == 0:
            continue
            
        loss = criterion(valid_predictions, valid_targets)
        
        # 4. Backpropagation
        loss.backward()
        optimizer.step()
        
        # --- NEW: BATCH METRICS AND PRINTING ---
        current_loss = loss.item()
        
        # FIX 2: Apply .detach() to prevent RuntimeError: Can't call numpy() on Tensor that requires grad.
        # Calculate Batch MAE (MAE is the most clinically relevant metric)
        batch_mae = mean_absolute_error(
            valid_targets.detach().cpu().numpy(),  # .detach() added
            valid_predictions.detach().cpu().numpy() # .detach() added
        )
        
        # Print Batch Loss and MAE using tqdm.write for clean output
        tqdm.write(
            f"  [Batch {i+1}/{len(data_loader)}] Loss: {current_loss:.4f} | MAE: {batch_mae:.4f}"
        )
        # --- END NEW: BATCH METRICS AND PRINTING ---
        
        total_loss_sum += current_loss * len(valid_predictions)
        total_valid_predictions += len(valid_predictions)
        
    # Return average loss per valid prediction step
    return total_loss_sum / total_valid_predictions if total_valid_predictions > 0 else 0


# ==========================================
# 4. METRICS HELPER FUNCTION (NEW)
# ==========================================

def calculate_metrics(predictions, targets):
    """Calculates MAE and R2 for the entire validation set."""
    # Note: Predictions and targets passed here are already detached and on CPU from validate_model
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    mae = mean_absolute_error(targets_np, preds_np)
    r2 = r2_score(targets_np, preds_np)
    
    return mae, r2

# ==========================================
# 5. VALIDATION STEP FUNCTION (UPDATED)
# ==========================================

def validate_model(model, data_loader, criterion):
    """Evaluates the model on the validation set and returns loss, MAE, and R2."""
    model.eval()
    total_loss_sum = 0
    total_valid_predictions = 0
    
    # Lists to store all valid predictions and targets for metrics calculation
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # 1. Prepare Data
            images = batch['image_seq'].to(DEVICE)
            clinical = batch['clinical_seq'].to(DEVICE)
            seq_mask = batch['seq_mask'].to(DEVICE)
            target_vf = batch['target_vf'].to(DEVICE)
            target_mask = batch['target_mask'].to(DEVICE)

            # 2. Forward Pass
            predictions = model(images, clinical, seq_mask)

            # 3. Calculate Loss and Collect Metric Data (Masked)
            predictions_flat = predictions.view(-1)
            target_vf_flat = target_vf.view(-1)
            
            mask_expanded = target_mask.unsqueeze(-1).repeat(1, 1, target_vf.shape[-1]).view(-1)
            
            valid_predictions = predictions_flat[mask_expanded == 1]
            valid_targets = target_vf_flat[mask_expanded == 1]
            
            if len(valid_predictions) == 0:
                continue

            loss = criterion(valid_predictions, valid_targets)
            
            total_loss_sum += loss.item() * len(valid_predictions)
            total_valid_predictions += len(valid_predictions)
            
            # Store data for overall epoch metrics
            # Detach is not strictly needed here due to torch.no_grad, but good practice
            all_preds.append(valid_predictions.cpu()) 
            all_targets.append(valid_targets.cpu())
                
    if total_valid_predictions == 0:
        return 0, 0, 0 # Handle case with no valid steps
        
    # Final Loss
    mean_val_loss = total_loss_sum / total_valid_predictions

    # Calculate MAE and R2
    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    val_mae, val_r2 = calculate_metrics(all_preds_tensor, all_targets_tensor)
    
    return mean_val_loss, val_mae, val_r2 # Return all three metrics


# ==========================================
# 6. MAIN EXECUTION LOOP (UPDATED)
# ==========================================

def main():
    # --- FIX 3: Print device once in main() ---
    print(f"Using device: {DEVICE}")
    print("-" * 50)
    print("MMI-SST-FORMER: STARTING TRAINING")
    print("-" * 50)
    
    # 1. Load Data and Get Dimensions
    try:
        train_loader, val_loader, _, VF_DIM, CLINICAL_INPUT_DIM = get_data_loaders(
            batch_size=Config.BATCH_SIZE, 
            num_workers=Config.NUM_WORKERS
        )
        print(f"\nModel Params: VF_DIM={VF_DIM}, CLIN_INPUT_DIM={CLINICAL_INPUT_DIM}")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Data loading failed. Check preprocessing files: {e}")
        return

    # 2. Initialize Model
    model = MMI_SST_Former(
        clinical_input_dim=CLINICAL_INPUT_DIM,
        vf_output_dim=VF_DIM,
        fused_dim=Config.FUSED_DIM,
        num_heads=Config.TRANSFORMER_HEADS,
        num_layers=Config.TRANSFORMER_LAYERS,
        dropout=Config.DROPOUT_RATE
    ).to(DEVICE)

    # 3. Define Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        # FIX 4: Corrected the WEIGHT_DECAY typo
        weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.SmoothL1Loss(reduction='mean')
    
    # --- FIX 5: Save based on MAE, not Loss ---
    best_val_mae = float('inf') 

    # 4. Training Loop
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\n--- EPOCH {epoch}/{Config.NUM_EPOCHS} ---")
        
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Training Loss (SmoothL1): {train_loss:.4f}")
        writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # Validation
        val_loss, val_mae, val_r2 = validate_model(model, val_loader, criterion)
        
        # Print and Log Metrics (NEW OUTPUTS)
        print(f"Validation Loss (SmoothL1): {val_loss:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        print(f"Validation R2 Score: {val_r2:.4f}")

        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metric/MAE', val_mae, epoch)
        writer.add_scalar('Metric/R2', val_r2, epoch)
        
        # 5. Checkpoint Saving (Now based on minimizing MAE)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            checkpoint_path = os.path.join(Config.OUTPUT_DIR, f"best_model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae, # Saved the MAE instead of loss
            }, checkpoint_path)
            print(f"-> SAVED BEST MODEL with MAE: {best_val_mae:.4f}")
            
    print("\nTRAINING COMPLETE.")
    writer.close()


if __name__ == "__main__":
    main()