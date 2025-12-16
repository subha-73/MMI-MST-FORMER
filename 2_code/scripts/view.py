import torch

# Path to your .pth file
pth_path = r"D:\MMI-MST-FORMER\3_output\test_results.pth"

# Load checkpoint
checkpoint = torch.load(pth_path, map_location='cpu')

# Show keys stored in the file
print("Keys in checkpoint:")
for key in checkpoint.keys():
    print(" -", key)

# Print basic info
print("\nSaved Epoch:", checkpoint.get('epoch'))
print("Best Validation MAE:", checkpoint.get('best_val_mae'))

# Inspect model state_dict
print("\nModel State Dict Info:")
state_dict = checkpoint['model_state_dict']
print("Total parameters:", len(state_dict))

# Print few layer names and shapes
print("\nSample Parameters:")
for name, param in list(state_dict.items())[:5]:
    print(f"{name} -> {param.shape}")
