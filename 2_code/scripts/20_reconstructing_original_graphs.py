import matplotlib.pyplot as plt
import os
import sys

# SETUP PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
OUTPUT_DIR = os.path.join(ROOT_DIR, '3_results', 'graphs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def reconstruct_graphs():
    print(f"Reconstructing graphs from extracted log data...")
    
    # === DATA EXTRACTED FROM YOUR TERMINAL OUTPUT ===
    epochs = list(range(1, 51))
    
    # Extracted Loss Values
    loss = [
        0.5203, 0.4864, 0.4616, 0.4325, 0.2676, 0.2678, 0.1841, 0.2436, 0.0671, 0.0703,
        0.0859, 0.0718, 0.0939, 0.0236, 0.0391, 0.0337, 0.0127, 0.0130, 0.1819, 0.2604,
        0.0938, 0.0890, 0.1093, 0.0443, 0.0170, 0.0221, 0.0633, 0.0230, 0.0476, 0.0073,
        0.0193, 0.0543, 0.0589, 0.0302, 0.0176, 0.0065, 0.0062, 0.0108, 0.1103, 0.0361,
        0.0391, 0.0233, 0.0069, 0.0288, 0.0083, 0.0038, 0.0087, 0.0115, 0.1022, 0.0309
    ]

    # Extracted Validation Accuracy
    val_acc = [
        0.73, 0.73, 0.73, 0.73, 0.70, 0.70, 0.38, 0.73, 0.73, 0.73,
        0.70, 0.57, 0.59, 0.70, 0.68, 0.70, 0.73, 0.73, 0.70, 0.70,
        0.68, 0.65, 0.73, 0.76, 0.76, 0.73, 0.70, 0.73, 0.70, 0.70,
        0.65, 0.70, 0.70, 0.68, 0.70, 0.76, 0.73, 0.43, 0.73, 0.65,
        0.70, 0.73, 0.70, 0.68, 0.70, 0.70, 0.65, 0.68, 0.70, 0.68
    ]

    # Extracted Validation F1 Scores
    val_f1 = [
        0.4219, 0.4219, 0.4219, 0.4219, 0.4127, 0.4127, 0.3784, 0.5027, 0.5027, 0.5027,
        0.4127, 0.4825, 0.4689, 0.4127, 0.5595, 0.4127, 0.5027, 0.5595, 0.6105, 0.4127,
        0.4032, 0.3934, 0.4219, 0.5801, 0.5195, 0.4219, 0.5800, 0.4219, 0.4127, 0.4127,
        0.5397, 0.4127, 0.4868, 0.4714, 0.4868, 0.5195, 0.5027, 0.4324, 0.4219, 0.5397,
        0.4868, 0.5027, 0.4127, 0.4032, 0.4127, 0.4127, 0.5036, 0.4032, 0.4868, 0.4032
    ]

    # === PLOT 1: TRAINING LOSS ===
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, label='Training Loss', color='red', linewidth=2)
    plt.title('Training Loss Dynamics (Single-Scale Model)')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    save_path1 = os.path.join(OUTPUT_DIR, 'original_training_loss.png')
    plt.savefig(save_path1)
    print(f"   [+] Saved: {save_path1}")

    # === PLOT 2: ACCURACY & F1 ===
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='green', linestyle='--')
    plt.plot(epochs, val_f1, label='Validation F1 Score', color='blue', linewidth=2)
    
    # Highlight the Best Epoch (19)
    best_epoch_idx = 18 # Index 18 is Epoch 19
    plt.scatter([19], [val_f1[best_epoch_idx]], color='red', s=100, zorder=5, label='Best Model (Epoch 19)')
    plt.annotate(f'Peak F1: {val_f1[best_epoch_idx]:.4f}', 
                 (19, val_f1[best_epoch_idx]), 
                 xytext=(19, val_f1[best_epoch_idx] + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 ha='center')

    plt.title('Validation Performance (Single-Scale Model)')
    plt.xlabel('Epochs')
    plt.ylabel('Score (0.0 - 1.0)')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    save_path2 = os.path.join(OUTPUT_DIR, 'original_performance_curve.png')
    plt.savefig(save_path2)
    print(f"   [+] Saved: {save_path2}")

    print("\nDone!")

if __name__ == "__main__":
    reconstruct_graphs()