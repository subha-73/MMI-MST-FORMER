import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================

# NOTE: This assumes the PROCESSED_DIR structure is consistent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')

MAX_SEQ_LEN = 9 # V1...V9 are input

# ==========================================
# 2. DATASET DEFINITION
# ==========================================

class GlaucomaForecastingDataset(Dataset):
    """
    Pytorch Dataset for loading sequence data and preparing the V_t+1 target.
    Input sequence: V1 to V9 (MAX_SEQ_LEN)
    Target: V_t+1 (Visit 2 to Visit 10) for the VF features.
    """
    def __init__(self, split_prefix, processed_dir=PROCESSED_DIR):
        """
        Args:
            split_prefix (str): e.g., 'grape_train', 'grape_val', 'grape_test'
        """
        print(f"Loading {split_prefix} data...")
        
        self.split_prefix = split_prefix
        
        # 1. Load Tensors (Input Sequence: V1...V9)
        self.images = torch.load(os.path.join(processed_dir, f"{split_prefix}_images.pt"))
        self.clinical_features = torch.load(os.path.join(processed_dir, f"{split_prefix}_clinical.pt"))
        self.masks = torch.load(os.path.join(processed_dir, f"{split_prefix}_masks.pt"))
        
        # 2. Load Raw CSV (Needed to find the target V_t+1)
        self.df = pd.read_csv(os.path.join(processed_dir, f"{split_prefix}.csv"))
        
        # 3. Load PIDs (Ensures consistency with the generated tensors)
        with open(os.path.join(processed_dir, f"{split_prefix}_pids.txt"), 'r') as f:
            self.pids = [line.strip() for line in f]
        
        # 4. Get VF columns for the target
        vf_cols = [str(i) for i in range(61) if str(i) in self.df.columns]
        self.vf_cols = vf_cols
        
        # 5. Build Target Map (Key: unique_id, Value: Target sequence)
        self.target_map = self._create_target_sequence()
        
        # Verify all PIDs have a corresponding target
        assert len(self.pids) == len(self.target_map), \
            f"PID count mismatch: {len(self.pids)} vs {len(self.target_map)}"
        
        print(f"  ✓ Data loaded successfully for {split_prefix}. Total patients: {len(self.pids)}")
        print(f"  ✓ Input Sequence Shape: {self.images.shape}")
        print(f"  ✓ Target Sequence Shape: {self.target_map[self.pids[0]].shape}") # Example shape

    def _create_target_sequence(self):
        """
        Creates the target map where target is the VF data from V_t+1.
        
        If a patient has K visits:
        Input Sequence (V1...VK) -> Target Sequence (V2...V(K+1))
        
        Since MAX_SEQ_LEN = 9, the target sequence is V2...V10.
        If K < 10, target is padded with zeros.
        """
        
        vf_dim = len(self.vf_cols)
        target_map = {}
        
        for pid in self.pids:
            # 1. Get all visits for the patient from the full CSV
            visits = self.df[self.df['unique_id'] == pid].sort_values('Interval Years')
            
            # 2. Extract VF features from all visits (V1, V2, V3, ...)
            vf_features_all = visits[self.vf_cols].values
            
            # 3. Create the shifted target sequence: V2, V3, ..., V(K+1)
            # This is the full list of visits starting from the second one (index 1)
            # The last element is the target for the last input sequence (V9)
            target_visits = vf_features_all[1:, :] 
            
            # 4. Pad the targets to ensure the target sequence length is MAX_SEQ_LEN
            # V2...V10 needs to be length 9.
            current_target_len = target_visits.shape[0]
            pad_len = MAX_SEQ_LEN - current_target_len
            
            if pad_len > 0:
                # Pad target sequence with zeros
                padding = np.zeros((pad_len, vf_dim))
                final_target_seq = np.concatenate([target_visits, padding], axis=0)
            else:
                # If K >= 10, we only take V2...V10 (i.e., the first 9 targets)
                final_target_seq = target_visits[:MAX_SEQ_LEN, :]
            
            target_map[pid] = torch.tensor(final_target_seq, dtype=torch.float32)

        return target_map

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        
        # Input sequence features (V1...V9)
        images = self.images[idx] # (9, 3, 224, 224)
        clinical = self.clinical_features[idx] # (9, F)
        
        # Sequence mask (1 for real data, 0 for padding)
        mask = self.masks[idx] # (9)
        
        # Target (V2...V10)
        target_seq = self.target_map[pid] # (9, VF_DIM)

        # The model needs a final output to check which prediction steps are valid.
        # The prediction for V_t is only valid if V_{t+1} exists.
        # This is the target validity mask (mask for V2...V10)
        # It's essentially the original mask shifted by one.
        target_mask = torch.zeros_like(mask)
        # 1 means valid target. Since targets start at V2, the mask starts at the first real input.
        # The mask length is 9. If the original sequence was V1-V5 (mask [1,1,1,1,1,0,0,0,0])
        # The target sequence is V2-V6 (target_mask [1,1,1,1,1,0,0,0,0])
        valid_indices = torch.where(mask > 0)[0] # Indices where input is real
        if len(valid_indices) > 0:
             # The target mask is valid up to the last valid input index. 
             # Why? The prediction for V_t (last real input) is for V_t+1 (the first padded target).
             # We use the length of the *target* sequence (V2...VK+1) which is max 9.
             target_mask[:len(valid_indices)] = 1
             # CRITICAL CORRECTION: The target mask for the last *real* input must be 0 if V_{t+1} is missing.
             # If a patient only has K visits: V1...VK. Input mask is 1 up to K.
             # Target is V2...V(K+1). V(K+1) is only real if K > MAX_SEQ_LEN (which we cap at 9).
             # If K=5, input is V1..V5, target is V2..V6. V6 is padding.
             # The prediction for V5 (last real input) is for V6 (padding). Thus, target_mask[4] must be 0.
             
             # The prediction at sequence position 't' predicts V_{t+1}.
             # The prediction is valid if the V_{t+1} exists and is not padding.
             # In a sequence of length 9, the mask[t] is 1 if V_{t} is real.
             # The prediction at index 't' is for V_{t+1} (target at index 't').
             # Thus, the target is valid *up to one step before the end* of the real visits.
             
             # If visits=5: [V1, V2, V3, V4, V5, 0, 0, 0, 0]. Last real index: 4.
             # Targets: [V2, V3, V4, V5, P6, 0, 0, 0, 0].
             # Valid predictions: P1 (for V2), P2 (for V3), P3 (for V4), P4 (for V5).
             # Prediction for V5 (at index 4) is for P6 (padding) -> INVALID.
             target_mask[:len(valid_indices) - 1] = 1 # Valid up to the second-to-last real input
             
        
        return {
            'image_seq': images, 
            'clinical_seq': clinical, 
            'seq_mask': mask, 
            'target_vf': target_seq, 
            'target_mask': target_mask # Use this mask to compute loss only on valid targets
        }

def get_data_loaders(batch_size, num_workers=4):
    """
    Function to initialize and return all data loaders.
    """
    train_dataset = GlaucomaForecastingDataset('grape_train')
    val_dataset = GlaucomaForecastingDataset('grape_val')
    test_dataset = GlaucomaForecastingDataset('grape_test')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    # Return the dimension of the VF target for the regression head
    VF_DIM = len(train_dataset.vf_cols)
    CLINICAL_INPUT_DIM = train_dataset.clinical_features.shape[-1]
    
    return train_loader, val_loader, test_loader, VF_DIM, CLINICAL_INPUT_DIM

if __name__ == '__main__':
    # Quick Test to verify shapes
    try:
        train_loader, _, _, vf_dim, clin_dim = get_data_loaders(batch_size=2)
        print(f"\n[Test] VF Dimension: {vf_dim}, Clinical Input Dim: {clin_dim}")
        
        for batch in train_loader:
            print("\n[Test Batch Shapes]:")
            print(f"  Image Seq: {batch['image_seq'].shape}")      # [B, S, 3, 224, 224]
            print(f"  Clinical Seq: {batch['clinical_seq'].shape}") # [B, S, F_clin]
            print(f"  Seq Mask: {batch['seq_mask'].shape}")        # [B, S]
            print(f"  Target VF: {batch['target_vf'].shape}")      # [B, S, F_vf]
            print(f"  Target Mask: {batch['target_mask'].shape}")  # [B, S]
            break
            
    except FileNotFoundError:
        print("\n[ERROR] Ensure all preprocessing steps (01, 02, 03) have been run to create .pt files.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")