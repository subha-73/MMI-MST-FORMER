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

# Base directory for all heatmaps
BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, '3_results', 'heatmaps')
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

print(f"{'='*70}")
print(f"MODULE: GENERATING HEATMAPS FOR ALL IMAGES")
print(f"{'='*70}")

# Denormalization transform
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def visualize_all_images():
    # --- INPUT ---
    print("\n[INPUT] Loading full image tensor...")
    try:
        # Shape: [Num_Patients, Num_Visits, Channels, Height, Width]
        full_tensor = torch.load(TRAIN_IMGS)
        num_patients, num_visits, _, _, _ = full_tensor.shape
        print(f"   Source File: {TRAIN_IMGS}")
        print(f"   Found: {num_patients} Patients, {num_visits} Visits per patient")
        print(f"   Total Images to Process: {num_patients * num_visits}")
    except Exception as e:
        print(f"[ERROR] Could not load input: {e}")
        return

    # Load ResNet once
    print("\n[SETUP] Loading pre-trained ResNet18...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.eval()

    # --- PROCESSING LOOP ---
    print(f"\n[PROCESSING] Starting generation loop...")
    
    total_count = 0
    
    for p_idx in range(num_patients):
        # Create patient-specific folder
        patient_id = p_idx + 1
        patient_dir = os.path.join(BASE_OUTPUT_DIR, f'Patient_{patient_id:03d}')
        os.makedirs(patient_dir, exist_ok=True)
        
        print(f"   > Processing Patient {patient_id}/{num_patients}...")
        
        for v_idx in range(num_visits):
            visit_id = v_idx + 1
            
            # Get specific image tensor
            img_tensor = full_tensor[p_idx, v_idx]
            
            # --- Hook & Forward Pass ---
            feature_maps = {}
            def hook_fn(module, input, output):
                feature_maps['layer4'] = output.detach()
            
            hook = resnet.layer4.register_forward_hook(hook_fn)
            
            with torch.no_grad():
                resnet(img_tensor.unsqueeze(0))
            
            hook.remove()

            # --- Generate Heatmap ---
            activations = feature_maps['layer4']
            avg_activation = torch.mean(activations, dim=1, keepdim=True)
            
            heatmap = torch.nn.functional.interpolate(avg_activation, size=(224, 224), mode='bilinear', align_corners=False)
            heatmap = heatmap.squeeze().numpy()
            # Normalize to [0, 1] to avoid dark images
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            # --- Prepare Output Plot ---
            # Prepare original image
            orig_img_tensor = inv_normalize(img_tensor)
            orig_img_np = orig_img_tensor.permute(1, 2, 0).numpy()
            orig_img_np = np.clip(orig_img_np, 0, 1)

            # Create figure with no frame/axes
            fig = plt.figure(figsize=(4, 4))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            # Plot original + overlay
            ax.imshow(orig_img_np)
            ax.imshow(heatmap, cmap='jet', alpha=0.5)

            # Save individual file
            filename = f'Visit_{visit_id:02d}.png'
            save_path = os.path.join(patient_dir, filename)
            
            # Save without extra whitespace
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig) # Close memory
            
            total_count += 1
            
            # Optional: Print progress every 50 images to reduce spam
            if total_count % 50 == 0:
                 print(f"     [Progress: {total_count} images saved...]")

    print(f"\n{'='*70}")
    print(f"[DONE] Process complete.")
    print(f"Total Heatmaps Generated: {total_count}")
    print(f"Check the folder: {BASE_OUTPUT_DIR}")
    print(f"{'='*70}")

if __name__ == "__main__":
    visualize_all_images()