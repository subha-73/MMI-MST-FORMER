import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
import os
import sys
from tqdm import tqdm
import warnings

# --- SETUP PATHS ---
warnings.filterwarnings("ignore", category=FutureWarning)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) 
CODE_DIR = os.path.join(ROOT_DIR, '2_code')
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

try:
    # We continue to use the MMI_SST_Former class name, assuming this is the class name in your module file
    from modules.mmi_sst_former import MMI_SST_Former
    from data.dataset_loader import get_data_loaders
except ImportError as e:
    print(f"[CRITICAL ERROR] Failed to import core modules. Error: {e}")
    sys.exit(1)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIGURATION (Must match train.py) ---
class Config:
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, '3_output', 'checkpoints')
    # === CRITICAL UPDATE REQUIRED ===
    # *** REPLACE 'X' WITH THE ACTUAL EPOCH NUMBER OF YOUR BEST SAVED MODEL ***
    BEST_MODEL_FILENAME = 'best_model_epoch_49.pth' # <--- !!! UPDATE THIS NAME !!!
    
    # Model configuration
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_HEADS = 8
    DROPOUT_RATE = 0.1
    FUSED_DIM = 256

# --- METRICS HELPER (Calculates MAE and R2) ---
def calculate_metrics(predictions, targets):
    """Calculates MAE and R2 for the entire test set."""
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    mae = mean_absolute_error(targets_np, preds_np)
    r2 = r2_score(targets_np, preds_np)
    
    return mae, r2

# ==========================================
# MAIN EVALUATION FUNCTION
# ==========================================

def evaluate_model():
    print(f"Using device: {DEVICE}")
    print("-" * 50)
    print("MMI-SST-FORMER: STARTING TEST EVALUATION")
    print("-" * 50)
    
    # 1. Load Data and Get Dimensions
    try:
        # Get_data_loaders returns train, val, test loaders. We need the 3rd one (test).
        _, _, test_loader, VF_DIM, CLINICAL_INPUT_DIM = get_data_loaders(
            batch_size=Config.BATCH_SIZE, 
            num_workers=Config.NUM_WORKERS
        )
        print(f"Loaded Test Set with {len(test_loader)} batches.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Data loading failed for evaluation: {e}")
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

    # 3. Load Checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, Config.BEST_MODEL_FILENAME)
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        print("Please update Config.BEST_MODEL_FILENAME with your best model file name.")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Ensure model is set to evaluation mode
        model.eval() 
        print(f"Successfully loaded model checkpoint from epoch {checkpoint.get('epoch', 'N/A')}.")
    except Exception as e:
        print(f"[ERROR] Failed to load model state dictionary: {e}")
        return

    # 4. Evaluation Loop (Collecting all predictions)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image_seq'].to(DEVICE)
            clinical = batch['clinical_seq'].to(DEVICE)
            seq_mask = batch['seq_mask'].to(DEVICE)
            target_vf = batch['target_vf'].to(DEVICE)
            target_mask = batch['target_mask'].to(DEVICE)

            # Forward pass
            predictions = model(images, clinical, seq_mask)

            # --- Extract Valid Predictions and Targets (Masked) ---
            predictions_flat = predictions.view(-1)
            target_vf_flat = target_vf.view(-1)
            
            # This handles the masking of sequence padding and invalid steps
            mask_expanded = target_mask.unsqueeze(-1).repeat(1, 1, target_vf.shape[-1]).view(-1)
            
            valid_predictions = predictions_flat[mask_expanded == 1]
            valid_targets = target_vf_flat[mask_expanded == 1]
            
            if len(valid_predictions) == 0:
                continue
                
            all_preds.append(valid_predictions.cpu())
            all_targets.append(valid_targets.cpu())

    # 5. Final Metrics Calculation
    if len(all_preds) == 0:
        print("[ERROR] No valid predictions were collected. Evaluation failed.")
        return

    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    
    test_mae, test_r2 = calculate_metrics(all_preds_tensor, all_targets_tensor)
    
    print("-" * 50)
    print("--- FINAL TEST SET PERFORMANCE ---")
    print(f"Model: Single-Scale SST-Former")
    print(f"Test MAE (Mean Absolute Error): {test_mae:.4f}")
    print(f"Test R2 Score: {test_r2:.4f}")
    print("-" * 50)
    
    # Optionally save the predictions (for debugging/plotting)
    torch.save({
        'predictions': all_preds_tensor, 
        'targets': all_targets_tensor
    }, os.path.join(ROOT_DIR, '3_output', 'test_results.pth'))
    print(f"Saved raw predictions and targets to 3_output/test_results.pth")

if __name__ == "__main__":
    evaluate_model()