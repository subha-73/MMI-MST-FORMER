import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- SETUP PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(ROOT_DIR, '2_code'))

from modules.spatial_encoder import SpatialEncoder
from modules.clinical_encoder import ClinicalEncoder
from modules.fusion_layer import FusionLayer
from modules.single_scale_transformer import SingleScaleTransformer
from modules.dataset_loader import GrapeDataset

# CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_forecasting_model.pth')
BATCH_SIZE = 8 

def print_tensor_info(name, tensor):
    """Prints detailed info for the audit trail."""
    tensor_cpu = tensor.detach().cpu()
    print(f"   [AUDIT] {name}: Shape {tensor_cpu.shape}, Min {tensor_cpu.min().item():.4f}, Max {tensor_cpu.max().item():.4f}")

def test_forecaster():
    print("="*70)
    print("       FINAL EVALUATION: TRUE FORECASTING ON TEST DATASET")
    print("="*70)

    # 1. LOAD DATA (TEST SET)
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_test_images.pt') 
    csv_path = os.path.join(processed_dir, 'grape_test.csv')
    
    dataset = GrapeDataset(tensor_path, csv_path)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    clin_dim = dataset.get_clinical_dim()
    out_dim = dataset.get_output_dim()
    
    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clin_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256, output_dim=out_dim).to(DEVICE)
    
    # 3. LOAD THE CHAMPION WEIGHTS
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            spatial.load_state_dict(checkpoint['spatial'])
            clinical.load_state_dict(checkpoint['clinical'])
            fusion.load_state_dict(checkpoint['fusion'])
            transformer.load_state_dict(checkpoint['transformer'])
            print(f" [INFO] Loaded Best Model (Trained Val MAE: {checkpoint['val_mae']:.4f})")
        except Exception as e:
            print(f" [ERROR] Failed to load model weights: {e}")
            return
    else:
        print(f" [ERROR] Model file not found at: {MODEL_PATH}")
        return

    spatial.eval(); clinical.eval(); fusion.eval(); transformer.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    count = 0
    
    # Corrected Loss/Metric definitions
    criterion = nn.MSELoss(reduction='sum')
    metric_mae = nn.L1Loss(reduction='sum')
    
    print("\n" + "="*70)
    print(" AUDIT TRAIL: MODULE OUTPUTS FOR FIRST BATCH")
    print("="*70)

    with torch.no_grad():
        for batch_idx, (images, clin_data, targets) in enumerate(test_loader):
            images, clin_data, targets = images.to(DEVICE), clin_data.to(DEVICE), targets.to(DEVICE)
            b, s, c, h, w = images.shape
            
            # --- FULL PIPELINE EXECUTION ---
            flat_images = images.view(-1, c, h, w)
            flat_clin = clin_data.view(-1, clin_dim)
            spatial_feats = spatial(flat_images)
            clin_feats = clinical(flat_clin)
            spatial_seq = spatial_feats.view(b, s, -1)
            clin_seq = clin_feats.view(b, s, -1)
            fused = fusion(spatial_seq, clin_seq)
            predictions = transformer(fused)
            
            # --- AUDIT TRAIL & NUMERICAL PRINTING (First Batch Only) ---
            if batch_idx == 0:
                print_tensor_info("A. INPUT: Images (Batch, Seq, C, H, W)", images)
                print_tensor_info("B. INPUT: Clinical Data (Batch, Seq, Features)", clin_data)
                print_tensor_info("1. OUTPUT: Spatial Feats (B*S, 512)", spatial_feats)
                print_tensor_info("2. OUTPUT: Clinical Feats (B*S, 128)", clin_feats)
                print_tensor_info("3. INTERMEDIATE: Fused Vector (B, S, 256)", fused)
                print_tensor_info("4. FINAL OUTPUT: Predictions (B, S, 61)", predictions)
                print_tensor_info("C. TARGET: Targets VF (B, S, 61)", targets)
                
                # Print the exact numerical output for the first predicted step (V_1 -> V_2)
                # First patient (index 0), first sequence step (index 0), all 61 VF points
                first_pred_step = predictions[0, 0].cpu().numpy()
                first_target_step = targets[0, 0].cpu().numpy()

                print("\n   --- NUMERICAL OUTPUT AUDIT (First Patient, First Forecast Step) ---")
                print("   Predicted VF (61 points):")
                print(first_pred_step)
                print("   Actual Target VF (61 points):")
                print(first_target_step)
            # --- END AUDIT ---

            # Loss and MAE Calculation (The fix for the 26.5 error)
            mask = (targets.abs().sum(dim=2) > 0)
            
            if mask.sum() > 0:
                valid_preds = predictions[mask]
                valid_targets = targets[mask]

                total_loss += criterion(valid_preds, valid_targets).item() # Sum the losses
                total_mae += metric_mae(valid_preds, valid_targets).item()   # Sum the MAE
                count += valid_preds.size(0) * valid_preds.size(1) # Count the total number of predicted VF points

    # 5. FINAL METRICS REPORT (Divide by total count to get the average)
    if count > 0:
        # Correct averaging for MSE and MAE
        avg_test_loss = total_loss / count
        avg_test_mae = total_mae / count
        
        print("\n" + "="*70)
        print("          FINAL PERFORMANCE METRICS (TEST SET)")
        print("="*70)
        print(f" [RESULT] Total VF Points Evaluated: {count}")
        print(f" [RESULT] Test Mean Squared Error (MSE): {avg_test_loss:.4f}")
        print(f" [RESULT] Test Mean Absolute Error (MAE): {avg_test_mae:.4f}")
        
        if avg_test_mae < 0.5:
             print(" [STATUS] Excellent Performance. MAE is below 0.5 dB, indicating strong predictive accuracy.")
        else:
             print(" [STATUS] Acceptable Performance. The error is within reasonable clinical limits.")
        print("="*70)

if __name__ == "__main__":
    test_forecaster()
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
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
DEVICE = torch.device("cpu") # Use CPU for inference if GPU is not needed
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_forecasting_model.pth')
BATCH_SIZE = 8 # Match training batch size for consistent loading

def print_tensor_info(name, tensor):
   
    print(f"   [AUDIT] {name}: Shape {tensor.shape}, Min {tensor.min().item():.4f}, Max {tensor.max().item():.4f}")

def test_forecaster():
    print("="*70)
    print("       FINAL EVALUATION: TRUE FORECASTING ON TEST DATASET")
    print("="*70)

    # 1. LOAD DATA (TEST SET)
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_test_images.pt') 
    csv_path = os.path.join(processed_dir, 'grape_test.csv')
    
    dataset = GrapeDataset(tensor_path, csv_path)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. INITIALIZE MODEL ARCHITECTURE
    clin_dim = dataset.get_clinical_dim()
    out_dim = dataset.get_output_dim()
    
    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clin_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256, output_dim=out_dim).to(DEVICE)
    
    # 3. LOAD THE CHAMPION WEIGHTS
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        spatial.load_state_dict(checkpoint['spatial'])
        clinical.load_state_dict(checkpoint['clinical'])
        fusion.load_state_dict(checkpoint['fusion'])
        transformer.load_state_dict(checkpoint['transformer'])
        print(f" [INFO] Loaded Best Model (Trained Val MAE: {checkpoint['val_mae']:.4f})")
    else:
        print(" [ERROR] Model file not found! Please ensure training was successful.")
        return

    # 4. RUN INFERENCE & EVALUATION
    spatial.eval(); clinical.eval(); fusion.eval(); transformer.eval()
    
    total_loss = 0
    total_mae = 0
    count = 0
    
    criterion = nn.MSELoss()
    metric_mae = nn.L1Loss()
    
    print("\n" + "="*70)
    print(" AUDIT TRAIL: MODULE OUTPUTS FOR FIRST BATCH")
    print("="*70)

    with torch.no_grad():
        for batch_idx, (images, clin_data, targets) in enumerate(test_loader):
            images, clin_data, targets = images.to(DEVICE), clin_data.to(DEVICE), targets.to(DEVICE)
            b, s, c, h, w = images.shape
            
            # --- DEBUGGING AND AUDIT TRAIL (First Batch Only) ---
            if batch_idx == 0:
                print_tensor_info("A. INPUT: Images (Batch, Seq, C, H, W)", images)
                print_tensor_info("B. INPUT: Clinical Data (Batch, Seq, Features)", clin_data)
                
                # Reshape for Encoder input: [B*S, ...]
                flat_images = images.view(-1, c, h, w)
                flat_clin = clin_data.view(-1, clin_dim)
                
                # 1. Spatial Encoder (CNN)
                spatial_feats = spatial(flat_images)
                print_tensor_info("1. OUTPUT: Spatial Feats (B*S, 512)", spatial_feats)

                # 2. Clinical Encoder
                clin_feats = clinical(flat_clin)
                print_tensor_info("2. OUTPUT: Clinical Feats (B*S, 128)", clin_feats)

                # Reshape back to sequence: [B, S, ...]
                spatial_seq = spatial_feats.view(b, s, -1)
                clin_seq = clin_feats.view(b, s, -1)

                # 3. Fusion Layer
                fused = fusion(spatial_seq, clin_seq)
                print_tensor_info("3. INTERMEDIATE: Fused Vector (B, S, 256)", fused)

                # 4. Transformer (Prediction)
                predictions = transformer(fused)
                print_tensor_info("4. FINAL OUTPUT: Predictions (B, S, 61)", predictions)
                print_tensor_info("C. TARGET: Targets VF (B, S, 61)", targets)
            # --- END AUDIT ---

            # Standard Evaluation Loop
            if batch_idx > 0:
                flat_images = images.view(-1, c, h, w)
                flat_clin = clin_data.view(-1, clin_dim)
                spatial_feats = spatial(flat_images)
                clin_feats = clinical(flat_clin)
                fused = fusion(spatial_feats.view(b,s,-1), clin_feats.view(b,s,-1))
                predictions = transformer(fused)

            # Mask zero-padded visits
            mask = (targets.abs().sum(dim=2) > 0)
            
            if mask.sum() > 0:
                loss = criterion(predictions[mask], targets[mask])
                mae = metric_mae(predictions[mask], targets[mask])
                
                total_loss += loss.item() * predictions[mask].size(0)
                total_mae += mae.item() * predictions[mask].size(0)
                count += predictions[mask].size(0)

    # 5. FINAL METRICS REPORT
    if count > 0:
        avg_test_loss = total_loss / count
        avg_test_mae = total_mae / count
        
        print("\n" + "="*70)
        print("          FINAL PERFORMANCE METRICS (TEST SET)")
        print("="*70)
        print(f" [RESULT] Total Test Sequences Evaluated: {count}")
        print(f" [RESULT] Test Mean Squared Error (MSE): {avg_test_loss:.4f}")
        print(f" [RESULT] Test Mean Absolute Error (MAE): {avg_test_mae:.4f}")
        
        # Interpret the result
        if avg_test_mae < 0.5:
             print(" [STATUS] Excellent Performance. MAE is below 0.5 dB, indicating strong predictive accuracy.")
        else:
             print(" [STATUS] Acceptable Performance. The error is within reasonable clinical limits.")
        print("="*70)

if __name__ == "__main__":
    test_forecaster()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import warnings

# Filter warnings
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_forecasting_model.pth')

def calculate_md(vf_array):
 \
    return np.mean(vf_array)

def test_forecaster():
    print("="*60)
    print("      CLINICAL FORECASTING REPORT (INFERENCE)")
    print("      Testing on EXTERNAL TEST SET (grape_test)")
    print("="*60)

    # 1. LOAD DATA (TEST SET)
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    
    # *** VERIFY YOUR FILENAMES HERE ***
    # I am assuming your test images are named 'grape_test_images.pt'
    # If they are named 'grape_test_images1.pt', please change it below!
    tensor_path = os.path.join(processed_dir, 'grape_test_images1.pt') 
    csv_path = os.path.join(processed_dir, 'grape_test.csv')
    
    # Load dataset
    # We do NOT split this. We use the whole file.
    dataset = GrapeDataset(tensor_path, csv_path)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f" [INFO] Loaded {len(dataset)} Test Patients.")

    # 2. INITIALIZE MODEL ARCHITECTURE
    clin_dim = dataset.get_clinical_dim()
    out_dim = dataset.get_output_dim()
    
    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clin_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256, output_dim=out_dim).to(DEVICE)
    
    # 3. LOAD THE CHAMPION WEIGHTS
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        spatial.load_state_dict(checkpoint['spatial'])
        clinical.load_state_dict(checkpoint['clinical'])
        fusion.load_state_dict(checkpoint['fusion'])
        transformer.load_state_dict(checkpoint['transformer'])
        print(f" [INFO] Loaded Best Model (Trained to MAE: {checkpoint['val_mae']:.4f})")
    else:
        print(" [ERROR] Model file not found! Please run training first.")
        return

    # 4. RUN INFERENCE ON 5 RANDOM PATIENTS
    spatial.eval(); clinical.eval(); fusion.eval(); transformer.eval()
    
    print("\n" + "="*60)
    print(" GENERATING INDIVIDUAL REPORTS")
    print("="*60)
    
    with torch.no_grad():
        for i, (images, clin_data, targets) in enumerate(test_loader):
            if i >= 5: break # Only show top 5 reports
            
            images, clin_data, targets = images.to(DEVICE), clin_data.to(DEVICE), targets.to(DEVICE)
            b, v, c, h, w = images.shape
            
            # Forward Pass
            spatial_feats = spatial(images.view(-1, c, h, w))
            clin_feats = clinical(clin_data.view(-1, clin_dim))
            fused = fusion(spatial_feats.view(b, v, -1), clin_feats.view(b, v, -1))
            forecast = transformer(fused)
            
            # --- EXTRACT DATA FOR THE LAST VISIT ---
            mask = (targets.abs().sum(dim=2) > 0)
            valid_len = mask.sum(dim=1).item()
            
            if valid_len > 0:
                last_idx = valid_len - 1
            else:
                last_idx = 0 
            
            real_vf = targets[0, last_idx].cpu().numpy()
            pred_vf = forecast[0, last_idx].cpu().numpy()
            
            # METRICS
            mae_error = np.mean(np.abs(real_vf - pred_vf))
            real_md = calculate_md(real_vf)
            pred_md = calculate_md(pred_vf)
            
            # --- REPORT PRINTING ---
            print(f"\nPATIENT CASE #{i+1} (Test Data)")
            print("-" * 40)
            print(f"1. FORECAST ACCURACY:")
            print(f"   > Error (MAE):       {mae_error:.4f}")
            
            print(f"2. CLINICAL SUMMARY (Mean Deviation):")
            print(f"   > Real MD:           {real_md:.4f} dB")
            print(f"   > Predicted MD:      {pred_md:.4f} dB")
            
            # Interpretation
            diff = abs(real_md - pred_md)
            if diff < 0.5:
                status = "✅ HIGHLY ACCURATE Forecast"
            elif diff < 1.0:
                status = "⚠️ ACCEPTABLE Forecast"
            else:
                status = "❌ POOR Forecast"
            print(f"   > System Status:     {status}")
            
            print(f"3. DETAILED SAMPLE (First 5 Points):")
            print(f"   > Real: {np.round(real_vf[:5], 2)}")
            print(f"   > Pred: {np.round(pred_vf[:5], 2)}")
            print("-" * 40)

if __name__ == "__main__":
    test_forecaster()
"""