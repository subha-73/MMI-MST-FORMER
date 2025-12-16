import pandas as pd
import numpy as np
import os
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Data Directories
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
ROI_DIR = os.path.join(ROOT_DIR, '1_data', 'raw', 'grape', 'ROI images')

# Parameters
IMG_SIZE = 224
MAX_SEQ_LEN = 9 # <--- CORRECT: Max visits for input sequence V1...V9
                # If a patient has 10 visits, V1-V9 are input, V10 is the target.

print(f"{'='*70}")
print(f"GRAPE GLAUCOMA - IMAGE TENSOR GENERATION (ROI)")
print(f"{'='*70}")
print(f"Max Sequence Length: {MAX_SEQ_LEN}")

# Standard ImageNet Normalization (Best for Transfer Learning)
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. IMAGE LOADER ENGINE
# ==========================================

def load_and_process_image(filename):
    """Reads image, resizes, and normalizes. Returns Tensor or None."""
    s_name = str(filename).strip()
    if not s_name or s_name.lower() == 'nan':
        return None

    # Check common extensions
    candidates = [
        s_name,
        s_name.replace('.jpg', '.png'),
        s_name.replace('.png', '.jpg'),
    ]
    
    final_path = None
    for cand in candidates:
        full_path = os.path.join(ROI_DIR, cand)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            final_path = full_path
            break
    
    if final_path is None:
        return None # File truly missing
    
    try:
        img = cv2.imread(final_path)
        if img is None: 
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform_pipeline(img)
        return img_tensor
        
    except Exception as e:
        print(f"Error loading {final_path}: {e}")
        return None

def create_patient_tensors(csv_filename, output_prefix):
    """
    Reads a processed CSV, loads images, and saves padded tensors.
    """
    csv_path = os.path.join(PROCESSED_DIR, csv_filename)
    if not os.path.exists(csv_path):
        print(f"\n[SKIP] Could not find {csv_filename}")
        return

    print(f"\nProcessing {csv_filename}...")
    df = pd.read_csv(csv_path)
    
    unique_ids = df['unique_id'].unique()
    
    all_patient_imgs = []   # Will hold tensors of shape (9, 3, 224, 224)
    all_patient_masks = []  # Will hold masks of shape (9)
    valid_pids = []         
    
    missing_files = 0
    total_files = 0
    
    for pid in tqdm(unique_ids, desc="Building Tensors"):
        visits = df[df['unique_id'] == pid].sort_values('Interval Years')
        
        # Enforce Max Sequence Length (V1...V9)
        visits = visits.head(MAX_SEQ_LEN)
        
        seq_tensors = []
        
        for _, row in visits.iterrows():
            fname = row['Corresponding CFP']
            total_files += 1
            
            img_tensor = load_and_process_image(fname)
            
            if img_tensor is None:
                # If image is missing, use a Black Image (Zeros)
                img_tensor = torch.zeros((3, IMG_SIZE, IMG_SIZE))
                missing_files += 1
                
            seq_tensors.append(img_tensor)
        
        if len(seq_tensors) == 0:
            continue
            
        # Stack visits into one tensor: (Seq_Len, 3, 224, 224)
        seq_tensor = torch.stack(seq_tensors)
        
        # --- PADDING LOGIC ---
        current_len = len(seq_tensors)
        pad_len = MAX_SEQ_LEN - current_len
        
        if pad_len > 0:
            # Create padding (zeros)
            padding = torch.zeros((pad_len, 3, IMG_SIZE, IMG_SIZE))
            
            # Concatenate: [Real_Data, Padding]
            final_seq = torch.cat([seq_tensor, padding], dim=0)
            
            # Create Mask: 1 for Real, 0 for Padding
            mask = torch.cat([torch.ones(current_len), torch.zeros(pad_len)], dim=0)
        else:
            final_seq = seq_tensor
            mask = torch.ones(MAX_SEQ_LEN)
            
        all_patient_imgs.append(final_seq)
        all_patient_masks.append(mask)
        valid_pids.append(pid)

    # Convert lists to PyTorch Tensors
    if len(all_patient_imgs) == 0:
        print("  [ERROR] No valid patients found!")
        return
        
    # Final Shape: (Num_Patients, 9, 3, 224, 224)
    big_tensor_imgs = torch.stack(all_patient_imgs)
    # Final Mask Shape: (Num_Patients, 9)
    big_tensor_masks = torch.stack(all_patient_masks)
    
    # Save to disk
    out_img_path = os.path.join(PROCESSED_DIR, f"{output_prefix}_images.pt")
    out_mask_path = os.path.join(PROCESSED_DIR, f"{output_prefix}_masks.pt")
    
    torch.save(big_tensor_imgs, out_img_path)
    torch.save(big_tensor_masks, out_mask_path)
    
    print(f"  ✓ Saved Tensors to: {out_img_path}")
    print(f"    Tensor Shape: {big_tensor_imgs.shape}")
    print(f"  ✓ Missing Images Handled: {missing_files}/{total_files} ({missing_files/total_files*100:.1f}%)")
    
    # Save the PIDs to match with DataLoader
    with open(os.path.join(PROCESSED_DIR, f"{output_prefix}_pids.txt"), 'w') as f:
        for pid in valid_pids:
            f.write(f"{pid}\n")
    print(f"  ✓ Saved Patient IDs ({len(valid_pids)}).")


# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    splits = [
        ('grape_train.csv', 'grape_train'),
        ('grape_val.csv',   'grape_val'),
        ('grape_test.csv',  'grape_test')
    ]
    
    for csv_file, out_prefix in splits:
        create_patient_tensors(csv_file, out_prefix)
        
    print(f"\n{'='*70}")
    print("IMAGE PROCESSING COMPLETE!")
    print(f"{'='*70}")
    

if __name__ == "__main__":
    main()