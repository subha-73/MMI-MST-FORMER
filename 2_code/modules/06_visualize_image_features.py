import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import os
import sys

# ==========================================
# 1. SETUP
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
TRAIN_IMGS = os.path.join(PROCESSED_DIR, 'grape_train_images.pt')

print(f"{'='*70}")
print(f"MODULE: VISUALIZE SPATIAL FEATURES")
print(f"{'='*70}")

def visualize_features():
    # --- INPUT ---
    print("\n[INPUT] Loading Tensor...")
    try:
        full_tensor = torch.load(TRAIN_IMGS)
        print(f"   Source File: {TRAIN_IMGS}")
        print(f"   Full Tensor Shape: {full_tensor.shape}")
        
        # Select Patient 0, Visit 0
        img_tensor = full_tensor[0, 0] # Shape: [3, 224, 224]
        print(f"   Selected Sample: Patient 0, Visit 0")
        print(f"   Input Image Shape: {img_tensor.shape}")
    except Exception as e:
        print(f"[ERROR] Could not load input: {e}")
        return

    # --- PROCESSING (INTERMEDIATE) ---
    print("\n[PROCESSING] Extracting Feature Maps...")
    # Load ResNet
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    first_layer = resnet.conv1
    
    with torch.no_grad():
        input_batch = img_tensor.unsqueeze(0) # [1, 3, 224, 224]
        activations = first_layer(input_batch) # [1, 64, 112, 112]
        
    print(f"   Layer: ResNet18 Conv1")
    print(f"   Activation Output Shape: {activations.shape}")
    print(f"   (64 different filters detected)")

    # --- OUTPUT (DELIVERABLE) ---
    print("\n[OUTPUT] Generating Heatmap Grid...")
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('Intermediate Deliverable: Spatial Encoder Features', fontsize=16)

    for i in range(16):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        feature_map = activations[0, i].numpy()
        ax.imshow(feature_map, cmap='jet')
        ax.axis('off')
        ax.set_title(f'Filter {i+1}')

    plt.tight_layout()
    
    save_path = os.path.join(ROOT_DIR, 'deliverable_spatial_features.png')
    plt.savefig(save_path)
    print(f"   ✓ SAVED FILE: {save_path}")
    print("   ✓ Displaying Plot...")
    plt.show()

if __name__ == "__main__":
    visualize_features()