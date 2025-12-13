import torch
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# ==========================================
# 1. SETUP
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
TRAIN_IMGS = os.path.join(PROCESSED_DIR, 'grape_train_images.pt')

print(f"{'='*70}")
print(f"MODULE: VISUALIZE DEEP SPATIAL FEATURES (LAYER 4)")
print(f"{'='*70}")

# Denormalization transform to convert tensor back to PIL image for plotting
# Mean and Std from ImageNet (used in your preprocessing)
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def visualize_deeper_features():
    # --- INPUT ---
    print("\n[INPUT] Loading Tensor...")
    try:
        full_tensor = torch.load(TRAIN_IMGS)
        # Select Patient 0, Visit 0
        img_tensor = full_tensor[0, 0] # Shape: [3, 224, 224]
        print(f"   Selected Sample: Patient 0, Visit 0")
        print(f"   Input Image Shape: {img_tensor.shape}")
    except Exception as e:
        print(f"[ERROR] Could not load input: {e}")
        return

    # --- PROCESSING (INTERMEDIATE) ---
    print("\n[PROCESSING] Extracting and Averaging Feature Maps from Layer 4...")
    
    # 1. Load ResNet (Pre-trained)
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.eval() # Set to evaluation mode

    # 2. Hook to capture feature maps
    feature_maps = {}
    def hook_fn(module, input, output):
        feature_maps['layer4'] = output.detach()

    # Register hook on the last convolutional block
    hook = resnet.layer4.register_forward_hook(hook_fn)

    # 3. Forward Pass
    with torch.no_grad():
        input_batch = img_tensor.unsqueeze(0) # [1, 3, 224, 224]
        resnet(input_batch)
    
    hook.remove() # Clean up hook

    # 4. Get activations and average them
    activations = feature_maps['layer4'] # Shape: [1, 512, 7, 7]
    print(f"   Layer: ResNet18 Layer 4")
    print(f"   Activation Output Shape: {activations.shape}")
    
    # Calculate the mean of all 512 feature maps to get a single summary map
    avg_activation = torch.mean(activations, dim=1, keepdim=True) 
    print(f"   Averaged Activation Shape: {avg_activation.shape}")

    # 5. Resize to original image size for visualization
    # Use bilinear interpolation to smooth the 7x7 map up to 224x224
    heatmap = torch.nn.functional.interpolate(avg_activation, size=(224, 224), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().numpy() # Convert to numpy array [224, 224]
    
    # Normalize heatmap to [0, 1] for plotting
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # --- OUTPUT (DELIVERABLE) ---
    print("\n[OUTPUT] Generating Overlay Heatmap...")
    
    # Prepare original image for plotting
    # Denormalize and convert to numpy (H, W, C)
    orig_img_tensor = inv_normalize(img_tensor)
    orig_img_np = orig_img_tensor.permute(1, 2, 0).numpy()
    orig_img_np = np.clip(orig_img_np, 0, 1) # Ensure values are in [0, 1]

    # Create plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Intermediate Deliverable: Deep Spatial Features (Layer 4)', fontsize=16)

    # Plot 1: Original Image
    ax[0].imshow(orig_img_np)
    ax[0].set_title('Original Input Image')
    ax[0].axis('off')

    # Plot 2: Heatmap Overlay
    # We plot the original image first, then overlay the heatmap with transparency
    ax[1].imshow(orig_img_np)
    cbar = ax[1].imshow(heatmap, cmap='jet', alpha=0.5) # alpha sets transparency
    ax[1].set_title('Averaged Feature Heatmap Overlay')
    ax[1].axis('off')
    
    # Add colorbar
    fig.colorbar(cbar, ax=ax[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    save_path = os.path.join(ROOT_DIR, 'deliverable_deep_spatial_features.png')
    plt.savefig(save_path) #
    print(f"   ✓ SAVED FILE: {save_path}")
    print("   ✓ Displaying Plot...")
    plt.show()

if __name__ == "__main__":
    visualize_deeper_features()