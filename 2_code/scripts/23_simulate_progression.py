import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
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
DEVICE = torch.device("cpu") # CPU is sufficient for inference
MODEL_PATH = os.path.join(ROOT_DIR, '3_results', 'models', 'best_forecasting_model.pth')
SAVE_PLOT_PATH = os.path.join(ROOT_DIR, '3_results', '5_year_forecast_simulation.png')
FORECAST_YEARS = [1, 3, 5] # The user input for the simulation

def calculate_md(vf_array):
    """Calculates Mean Deviation (Average of the 61 points)"""
    return np.mean(vf_array)

def load_and_initialize_model(clin_dim, out_dim):
    """Loads the model components with trained weights."""
    spatial = SpatialEncoder(output_dim=512).to(DEVICE)
    clinical = ClinicalEncoder(input_dim=clin_dim, output_dim=128).to(DEVICE)
    fusion = FusionLayer(spatial_dim=512, clinical_dim=128, fused_dim=256).to(DEVICE)
    transformer = SingleScaleTransformer(input_dim=256, output_dim=out_dim).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        spatial.load_state_dict(checkpoint['spatial'])
        clinical.load_state_dict(checkpoint['clinical'])
        fusion.load_state_dict(checkpoint['fusion'])
        transformer.load_state_dict(checkpoint['transformer'])
        print(f" [INFO] Loaded Best Model (Val MAE: {checkpoint['val_mae']:.4f})")
    else:
        raise FileNotFoundError("Best model weights not found. Check MODEL_PATH.")

    spatial.eval(); clinical.eval(); fusion.eval(); transformer.eval()
    return spatial, clinical, fusion, transformer

def run_simulation():
    print("="*60)
    print(" 5-YEAR GLAUCOMA PROGRESSION SIMULATION (RECURSIVE)")
    print("="*60)

    # 1. LOAD DATA
    processed_dir = os.path.join(ROOT_DIR, '1_data', 'processed')
    tensor_path = os.path.join(processed_dir, 'grape_test_images.pt') # Use Test Data
    csv_path = os.path.join(processed_dir, 'grape_test.csv')
    
    dataset = GrapeDataset(tensor_path, csv_path)
    # Batch size 1, Shuffle False for predictable patient ID retrieval (if needed)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    clin_dim = dataset.get_clinical_dim()
    out_dim = dataset.get_output_dim()
    
    try:
        spatial, clinical, fusion, transformer = load_and_initialize_model(clin_dim, out_dim)
    except FileNotFoundError as e:
        print(f" [ERROR] {e}")
        return

    # 2. SELECT STARTING POINT (A patient with at least one sequence step)
    # We will simply take the first valid patient in the test set.
    
    # Extract the last REAL visit as the starting point for prediction
    
    images_0, clin_data_0, targets_0 = next(iter(loader)) 
    images_0, clin_data_0, targets_0 = images_0.to(DEVICE), clin_data_0.to(DEVICE), targets_0.to(DEVICE)

    # Find the index of the last valid visit in the targets (the current state)
    mask = (targets_0.abs().sum(dim=2) > 0)
    last_real_idx = mask.sum(dim=1).item() - 1
    
    if last_real_idx < 0:
        print(" [ERROR] Cannot find a multi-visit patient to start simulation.")
        return

    # Extract the features of the LAST REAL VISIT (V_current)
    # This visit will be the 'Year 0' starting point for the simulation
    current_image = images_0[0, last_real_idx].unsqueeze(0).to(DEVICE) # [1, 3, 224, 224]
    current_clin = clin_data_0[0, last_real_idx].unsqueeze(0).to(DEVICE) # [1, 67]
    current_vf = targets_0[0, last_real_idx].cpu().detach().numpy() # [61] - The true current VF # [61] - The true current VF

    print(f" [SIM] Starting Prediction from Patient's Last Known Visit (MD: {calculate_md(current_vf):.3f} dB)")

    # 3. RUN RECURSIVE PREDICTION (Forecasting the future)
    # Store results for plotting
    simulation_mds = {'Year 0 (Real)': calculate_md(current_vf)}
    
    # We will use the average interval time in the dataset, which is what the model learned from.
    # For a fixed time prediction (e.g., 1, 3, 5 years), we need a normalized interval that 
    # corresponds to 1 year of progression.
    
    # Assuming the 'Interval_Norm' feature (index 5) is normalized such that 1.0 = avg interval (e.g., 1 year)
    # We will use the difference in the normalized interval from the start point.
    
    # Get the normalized value of the last real visit's interval
    base_interval = current_clin[0, 5].item() 
    
    # Create the prediction sequence
    
    # Start prediction from the features of V_current
    image_input = current_image
    vf_input = current_vf 
    
    for year in FORECAST_YEARS:
        
        # --- PREPARE NEXT STEP INPUT ---
        # The key idea: the model is trained to predict the next step based on the interval.
        # To simulate 1, 3, 5 years, we need to create an input feature that represents that time jump.

        # We need the normalized value for the total time jump (Year 'y')
        # We will assume a simple denormalization factor for this demo (e.g., 1.0 = 1 year)
        # NOTE: In a real system, you would need the exact mean/std of your 'Interval_Norm' feature.
        
        # We'll stick to 1-year jumps and recurse to cover 3 and 5 years.

        # Prepare the input for ONE YEAR JUMP:
        # We need the *latest predicted VF* and the *latest image* for the input
        
        # For simplicity in this recursive simulation, we will reuse the last real image, 
        # as the image changes very slowly.
        
        # --- RECURSION LOOP (1-year jumps) ---
        
        # We will loop in 1-year increments until we reach the target year
        current_year = 0
        predicted_vf_latest = current_vf.copy()
        
        for i in range(year):
            
            # --- PREPARE THE INPUT TENSOR (Simulating the next visit) ---
            # Input features must be [1, 9, ...] and padded
            
            # 1. Update the time input to represent a 1-year jump (Normalized)
            # We assume a fixed normalized jump of 1.0 represents 1 year.
            sim_clin = current_clin.clone()
            sim_clin[0, 5] = 1.0 # Set Interval_Norm to 1.0 (representing a 1-year gap)
            
            # 2. Update the VF input with the latest prediction (The most crucial step)
            sim_clin[0, 6:] = torch.tensor(predicted_vf_latest).to(DEVICE) # VF features start at index 6

            # 3. Assemble the Sequence Tensors (S=9, but only using S=1 for prediction)
            images_in = torch.zeros(1, 9, 3, 224, 224).to(DEVICE)
            clin_in = torch.zeros(1, 9, clin_dim).to(DEVICE)
            
            # Place the current image/clinical data into the first sequence slot
            images_in[0, 0] = image_input[0] 
            clin_in[0, 0] = sim_clin[0]

            # 4. Forward Pass (Prediction for the next step)
            spatial_feats = spatial(images_in.view(-1, 3, 224, 224))
            clin_feats = clinical(clin_in.view(-1, clin_dim))
            fused = fusion(spatial_feats.view(1, 9, -1), clin_feats.view(1, 9, -1))
            
            # The prediction for the next step is the first output of the transformer (index 0)
            next_pred_vf = transformer(fused)[0, 0].cpu().detach().numpy() 
            
            # 5. Prepare for Next Loop (Update the state)
            predicted_vf_latest = next_pred_vf
            
            current_year += 1
            if current_year == year:
                 # Store the result only at the target year
                simulation_mds[f'Year {year} (Predicted)'] = calculate_md(predicted_vf_latest)
                print(f" [SIM] Predicted MD after {year} years: {calculate_md(predicted_vf_latest):.3f} dB")
                break


    # 4. PLOT AND REPORT
    md_values = list(simulation_mds.values())
    labels = list(simulation_mds.keys())
    
    plt.figure(figsize=(8, 6))
    
    # Find the predicted change
    change = md_values[-1] - md_values[0]
    
    plt.plot(labels, md_values, marker='o', linestyle='-', color='teal')
    plt.title(f"Glaucoma Progression Forecast: 5-Year Simulation\nTotal Predicted Change: {change:.3f} dB", fontsize=14)
    plt.xlabel("Time Point", fontsize=12)
    plt.ylabel("Mean Deviation (dB)", fontsize=12)
    plt.grid(True, alpha=0.5)
    
    # Highlight the starting point (Real)
    plt.scatter(labels[0], md_values[0], color='red', s=100, label='Starting Point (Real)')
    
    # Annotate values
    for i, md in enumerate(md_values):
        plt.annotate(f'{md:.2f} dB', (labels[i], md), textcoords="offset points", xytext=(-5, 10), ha='center')

    plt.legend()
    os.makedirs(os.path.dirname(SAVE_PLOT_PATH), exist_ok=True)
    plt.savefig(SAVE_PLOT_PATH)
    plt.close()

    print("\n" + "="*60)
    print(f" [SUCCESS] Multi-Year Simulation Plot saved to: {SAVE_PLOT_PATH}")
    print(" This plot demonstrates the model's ability to recursively predict.")
    print("="*60)


if __name__ == "__main__":
    run_simulation()