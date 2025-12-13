import torch
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
TRAIN_IMGS = os.path.join(PROCESSED_DIR, 'grape_train_images.pt')

print(f"{'='*70}")
print(f"MODULE: GLOBAL AVERAGE SPATIAL ATTENTION")
print(f"{'='*70}")

def visualize_global_average(sample_limit=100):
    # 1. Load Data
    print("\n[INPUT] Loading Tensor...")
    try:
        full_tensor = torch.load(TRAIN_IMGS) # [N, Visits, 3, 224, 224]
        # Flatten visits so we just have a list of images
        N, V, C, H, W = full_tensor.shape
        all_images = full_tensor.view(-1, C, H, W)
        
        # Limit to first 100 images to save time
        limit = min(sample_limit, all_images.shape[0])
        subset = all_images[:limit]
        print(f"   Analyzing first {limit} images to compute global average...")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # 2. Setup Model
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.eval()

    # 3. Accumulate Activations
    feature_maps = {}
    def hook_fn(module, input, output):
        feature_maps['layer4'] = output.detach()
    hook = resnet.layer4.register_forward_hook(hook_fn)

    # Accumulator for the heatmap
    summed_heatmap = torch.zeros((224, 224))

    print("   Processing images...")
    with torch.no_grad():
        for i in range(limit):
            img = subset[i].unsqueeze(0)
            resnet(img)
            
            # Get activation
            act = feature_maps['layer4'] # [1, 512, 7, 7]
            avg_act = torch.mean(act, dim=1, keepdim=True) # [1, 1, 7, 7]
            
            # Upscale to 224x224
            upscaled = torch.nn.functional.interpolate(avg_act, size=(224, 224), mode='bilinear', align_corners=False)
            summed_heatmap += upscaled.squeeze().cpu()
            
            if i % 20 == 0:
                print(f"     > Processed {i}/{limit}...")

    hook.remove()

    # 4. Average and Normalize
    avg_heatmap = summed_heatmap / limit
    avg_heatmap = (avg_heatmap - avg_heatmap.min()) / (avg_heatmap.max() - avg_heatmap.min())
    
    # 5. Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_heatmap, cmap='jet')
    plt.colorbar(label='Mean Activation Strength')
    plt.title(f'Global Average Attention (n={limit})\nWhere does the model usually look?', fontsize=14)
    plt.axis('off')

    save_path = os.path.join(ROOT_DIR, 'deliverable_global_attention.png')
    plt.savefig(save_path)
    print(f"\n[OUTPUT] Saved Global Map: {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_global_average()