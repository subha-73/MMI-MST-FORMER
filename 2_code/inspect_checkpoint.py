import torch
import os
import sys
import numpy as np

# --- SETUP PATHS ---
# Determine the project root (assuming this script is in 2_code/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) 

# --- CONFIGURATION ---
RESULTS_PATH = os.path.join(ROOT_DIR, '3_output', 'test_results.pth')

def inspect_results():
    """Loads the test results file and prints the contents."""
    print("-" * 50)
    print("LOADING AND INSPECTING RAW PREDICTIONS")
    print("-" * 50)

    if not os.path.exists(RESULTS_PATH):
        print(f"[ERROR] Results file not found at: {RESULTS_PATH}")
        print("Please ensure evaluate.py was run successfully!")
        return

    try:
        # Load the dictionary saved by torch.save()
        results = torch.load(RESULTS_PATH)
        
        predictions_tensor = results.get('predictions')
        targets_tensor = results.get('targets')

        if predictions_tensor is None or targets_tensor is None:
            print("[ERROR] 'predictions' or 'targets' key missing from the saved file.")
            return

        print(f"File loaded successfully from: {RESULTS_PATH}")
        print(f"Total number of predicted VF points (across all test patients/visits): {predictions_tensor.shape[0]}")
        print(f"Total number of true VF points: {targets_tensor.shape[0]}")
        
        # --- Display a sample of the raw predicted VF values (ZLF/TD) ---
        
        # We assume the VF dimension is 61 (the number of points). 
        VF_DIM = 61
        
        if predictions_tensor.shape[0] >= VF_DIM:
            # Extract the first complete predicted VF vector (61 points)
            sample_prediction = predictions_tensor[:VF_DIM].numpy()
            sample_target = targets_tensor[:VF_DIM].numpy()
            
            print("\n--- SAMPLE PREDICTION (First 61 VF Points) ---")
            # Format the output to be readable
            print(f"Predicted VF Vector (dB):\n{np.array2string(sample_prediction, precision=3, separator=', ')}")
            
            print("\n--- CORRESPONDING TRUE TARGET VALUES ---")
            print(f"True VF Vector (dB):\n{np.array2string(sample_target, precision=3, separator=', ')}")
            
            # Calculate and print the MAE for just this single sample
            sample_mae = np.mean(np.abs(sample_prediction - sample_target))
            print(f"\nMAE for this sample vector: {sample_mae:.4f} dB")
        else:
            print("[INFO] Not enough data points saved to display a full 61-point sample.")

    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load or process the file: {e}")

if __name__ == "__main__":
    inspect_results()