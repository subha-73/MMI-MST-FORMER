import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class GrapeDataset(Dataset):
    """
    Dataset loader for glaucoma progression with clinical data.
    Handles irregular sampling and returns sequences for forecasting.
    """
    def __init__(self, tensor_path, csv_path, max_seq_len=9, overlap_stride=1):
        """
        Args:
            tensor_path: path to pre-computed image tensors [num_patients, max_visits, 3, 224, 224]
            csv_path: path to clinical data CSV
            max_seq_len: maximum sequence length (fixed window)
            overlap_stride: stride for creating overlapping sequences (data augmentation)
        """
        print(f"   [LOADER] Target Tensor: {tensor_path}")
        self.images = torch.load(tensor_path, weights_only=True)
        
        print(f"   [LOADER] Target CSV: {csv_path}")
        self.clinical_df = pd.read_csv(csv_path)

        # Filter and sort by patient and visit
        self.clinical_df = self.clinical_df.sort_values(['unique_id', 'Visit Number'])
        counts = self.clinical_df['unique_id'].value_counts()
        valid_ids = counts[counts >= 2].index  # Need at least 2 visits
        self.clinical_df = self.clinical_df[self.clinical_df['unique_id'].isin(valid_ids)]
        
        self.grouped = self.clinical_df.groupby('unique_id', sort=False)
        self.patient_ids = list(self.grouped.groups.keys())
        self.images = self.images[:len(self.patient_ids)]

        # Visual field points and clinical features
        self.vf_cols = [str(i) for i in range(61)]
        self.clinical_feature_cols = [
            'Age', 'Gender', 'CCT', 'IOP', 'Category_of_Glaucoma', 'Interval_Norm'
        ]
        self.all_feature_cols = self.clinical_feature_cols + self.vf_cols  # 67 total
        
        self.max_seq_len = max_seq_len
        self.overlap_stride = overlap_stride
        
        # Create overlapping sequences for data augmentation
        self.sequences = self._create_sequences()
        
        print(f"   [LOADER] Created {len(self.sequences)} training sequences from {len(self.patient_ids)} patients")

    def _create_sequences(self):
        """
        Create overlapping sequences from each patient's data.
        Similar to paper's data augmentation in Section IV.B.1
        """
        sequences = []
        
        for patient_idx, pid in enumerate(self.patient_ids):
            patient_data = self.grouped.get_group(pid).sort_values('Visit Number')
            num_visits = len(patient_data)
            
            # Create overlapping windows
            for start_idx in range(0, max(1, num_visits - 1), self.overlap_stride):
                end_idx = min(start_idx + self.max_seq_len, num_visits)
                
                # Need at least 2 visits for input and target
                if end_idx - start_idx < 2:
                    continue
                
                sequences.append({
                    'patient_id': pid,
                    'patient_idx': patient_idx,
                    'start_visit': start_idx,
                    'end_visit': end_idx,
                    'num_visits': end_idx - start_idx
                })
        
        return sequences

    def __len__(self):
        return len(self.sequences)

    def pad_seq(self, t, max_len):
        """Pad sequence to max_len along first dimension."""
        curr_len = t.shape[0]
        if curr_len >= max_len:
            return t[:max_len]
        
        pad_size = max_len - curr_len
        padding = torch.zeros((pad_size, *t.shape[1:]), dtype=t.dtype, device=t.device)
        return torch.cat([t, padding], dim=0)

    def __getitem__(self, idx):
        """
        Returns:
            images: [max_seq_len, 3, 224, 224] - padded image sequence
            clinical_features: [max_seq_len, 67] - padded clinical + VF features
            time_intervals: [max_seq_len] - normalized time intervals from first visit
            valid_len: int - actual number of valid timesteps (before padding)
            target_vf: [max_seq_len, 61] - VF targets (shifted by 1)
        """
        seq_info = self.sequences[idx]
        pid = seq_info['patient_id']
        patient_idx = seq_info['patient_idx']
        start_visit = seq_info['start_visit']
        end_visit = seq_info['end_visit']
        
        # Get image sequence for this patient
        images_full = self.images[patient_idx]  # [max_visits, 3, 224, 224]
        
        # Get clinical data for this patient
        patient_data = self.grouped.get_group(pid).sort_values('Visit Number')
        patient_subset = patient_data.iloc[start_visit:end_visit]
        
        # Extract features
        clinical_data = torch.tensor(
            patient_subset[self.all_feature_cols].values,
            dtype=torch.float32
        )  # [num_visits, 67]
        
        vf_data = torch.tensor(
            patient_subset[self.vf_cols].values,
            dtype=torch.float32
        )  # [num_visits, 61]
        
        # Extract images
        seq_images = images_full[start_visit:end_visit]  # [num_visits, 3, 224, 224]
        
        # Compute time intervals from first visit (in this sequence)
        visit_numbers = patient_subset['Visit Number'].values
        time_deltas = visit_numbers - visit_numbers[0]
        time_intervals = torch.tensor(time_deltas, dtype=torch.float32)  # [num_visits]
        
        # Normalize time intervals to [0, 1] range
        max_delta = time_intervals.max()
        if max_delta > 0:
            time_intervals = time_intervals / max_delta
        
        # Create sequences for forecasting: t-1 is input, t is target
        # Input uses all but last visit
        input_images = seq_images[:-1]  # [num_visits-1, 3, 224, 224]
        input_clinical = clinical_data[:-1]  # [num_visits-1, 67]
        input_time = time_intervals[:-1]  # [num_visits-1]
        
        # Target is VF from next visit
        target_vf = vf_data[1:]  # [num_visits-1, 61]
        
        actual_len = input_images.shape[0]
        
        # Pad sequences to max_seq_len
        input_images = self.pad_seq(input_images, self.max_seq_len)
        input_clinical = self.pad_seq(input_clinical, self.max_seq_len)
        input_time = self.pad_seq(input_time, self.max_seq_len)
        target_vf = self.pad_seq(target_vf, self.max_seq_len)
        
        return {
            'images': input_images,
            'clinical': input_clinical,
            'time_intervals': input_time,
            'target_vf': target_vf,
            'valid_len': actual_len  # Number of valid (non-padded) timesteps
        }


# Example usage in DataLoader
if __name__ == "__main__":
    # For debugging
    dataset = GrapeDataset(
        tensor_path='path/to/grape_test_images1.pt',
        csv_path='path/to/grape_test.csv',
        max_seq_len=9,
        overlap_stride=1
    )
    
    sample = dataset[0]
    print(f"Images shape: {sample['images'].shape}")  # [9, 3, 224, 224]
    print(f"Clinical shape: {sample['clinical'].shape}")  # [9, 67]
    print(f"Time intervals shape: {sample['time_intervals'].shape}")  # [9]
    print(f"Target VF shape: {sample['target_vf'].shape}")  # [9, 61]
    print(f"Valid length: {sample['valid_len']}")
'''
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class GrapeDataset(Dataset):
    def __init__(self, tensor_path, csv_path):
        """
        Finalized Loader: Input (t) -> Target (t+1)
        """
        print(f"   [LOADER] Target Tensor: {tensor_path}")
        self.images = torch.load(tensor_path, weights_only=True)
        
        print(f"   [LOADER] Target CSV: {csv_path}")
        self.clinical_df = pd.read_csv(csv_path)

        # Ensure chronological order for forecasting
        if 'unique_id' in self.clinical_df.columns:
            self.clinical_df = self.clinical_df.sort_values(['unique_id', 'Visit Number'])
        
        self.grouped = self.clinical_df.groupby('unique_id', sort=False)
        self.patient_ids = list(self.grouped.groups.keys())
        
        # Syncing Image/CSV indices
        self.images = self.images[:len(self.patient_ids)]

        # 61 VF points + 6 clinical = 67 features
        self.vf_cols = [str(i) for i in range(61)]
        self.clinical_features = ['Age', 'Gender', 'CCT', 'IOP', 'Category_of_Glaucoma', 'Interval_Norm'] + self.vf_cols

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        images_full = self.images[idx] 
        patient_data = self.grouped.get_group(pid).sort_values('Visit Number')

        clin_data_full = torch.tensor(patient_data[self.clinical_features].values, dtype=torch.float32)
        vf_targets_full = torch.tensor(patient_data[self.vf_cols].values, dtype=torch.float32)

        MAX_SEQ_LEN = 9 
        # Forecasting shift: Input visits 0 to N-2
        if clin_data_full.shape[0] > 1:
            input_imgs, input_clin = images_full[:-1], clin_data_full[:-1]
            target_vf = vf_targets_full[1:]
        else:
            input_imgs, input_clin, target_vf = images_full, clin_data_full, vf_targets_full 

        # Fixed-size padding for the Hierarchical MST-Former
        def pad_seq(t, max_len):
            t = t[:max_len]
            p = torch.zeros((max_len - t.shape[0], *t.shape[1:]))
            return torch.cat([t, p], dim=0)

        return pad_seq(input_imgs, MAX_SEQ_LEN), pad_seq(input_clin, MAX_SEQ_LEN), pad_seq(target_vf, MAX_SEQ_LEN)
'''
'''
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class GrapeDataset(Dataset):
    def __init__(self, tensor_path, csv_path):
        """
        TRUE FORECASTING LOADER (MULTI-TASK): 
        Shifts data so Input (t) predicts Target (t+1) AND Time Gap (t+1).
        """
        print(f"   [LOADER] Loading Tensor: {tensor_path}")
        self.images = torch.load(tensor_path, weights_only=True)
        
        print(f"   [LOADER] Loading CSV: {csv_path}")
        self.clinical_df = pd.read_csv(csv_path)

        # Group and Sort
        if 'unique_id' in self.clinical_df.columns:
            self.clinical_df = self.clinical_df.sort_values(['unique_id', 'Visit Number'])
        self.grouped = self.clinical_df.groupby('unique_id', sort=False)
        self.patient_ids = list(self.grouped.groups.keys())
        
        # Sync Lengths
        if len(self.patient_ids) > self.images.shape[0]:
            self.patient_ids = self.patient_ids[:self.images.shape[0]]
        else:
            self.images = self.images[:len(self.patient_ids)]

        # Features
        # Ensure 'Interval_Norm' (or your time column) exists
        self.time_col = 'Interval_Norm' 
        
        feature_cols = ['Age', 'Gender', 'CCT', 'IOP', 'Category_of_Glaucoma', self.time_col]
        self.vf_cols = [str(i) for i in range(61)]
        feature_cols.extend(self.vf_cols)
        
        self.clinical_features = feature_cols
        self.target_cols = self.vf_cols

    def get_clinical_dim(self):
        return len(self.clinical_features)

    def get_output_dim(self):
        return len(self.target_cols)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # 1. Get All Data for Patient
        pid = self.patient_ids[idx]
        images_full = self.images[idx] 
        patient_data = self.grouped.get_group(pid)

        # 2. Convert to Tensors
        clin_data_full = torch.tensor(patient_data[self.clinical_features].values, dtype=torch.float32)
        vf_targets_full = torch.tensor(patient_data[self.target_cols].values, dtype=torch.float32)
        
        # Extract Time separately for Positional Encoding & Target Calculation
        # We assume Interval_Norm is cumulative (0, 6, 12...). If not, use cumsum()
        raw_time = torch.tensor(patient_data[self.time_col].values, dtype=torch.float32)

        # 3. FORECASTING LOGIC
        MAX_SEQ_LEN = 9 
        n_visits = clin_data_full.shape[0]

        if n_visits > 1:
            # INPUTS (0 to N-2)
            input_imgs = images_full[:-1]
            input_clin = clin_data_full[:-1]
            input_time = raw_time[:-1]
            
            # TARGETS (1 to N-1)
            target_vf  = vf_targets_full[1:]
            
            # TIME TARGET: The Gap (Delta T)
            # Gap = Time[t+1] - Time[t]
            target_gap = (raw_time[1:] - raw_time[:-1]).unsqueeze(1) # Shape [Seq, 1]
            
        else:
            # Padding case for single visit (Cannot forecast)
            input_imgs = images_full
            input_clin = clin_data_full
            input_time = raw_time
            target_vf = vf_targets_full 
            target_gap = torch.zeros(1, 1)

        # 4. FIXED SIZE PADDING / TRUNCATION
        def pad_tensor(tensor, target_len, dim=0):
            curr = tensor.shape[0]
            if curr >= target_len:
                return tensor[:target_len]
            else:
                pad_size = target_len - curr
                # Dynamic padding shape based on input dimensions
                shape = list(tensor.shape)
                shape[0] = pad_size
                padding = torch.zeros(*shape)
                return torch.cat([tensor, padding], dim=0)

        input_imgs = pad_tensor(input_imgs, MAX_SEQ_LEN)
        input_clin = pad_tensor(input_clin, MAX_SEQ_LEN)
        input_time = pad_tensor(input_time, MAX_SEQ_LEN) # Pad Time Steps too
        target_vf  = pad_tensor(target_vf, MAX_SEQ_LEN)
        target_gap = pad_tensor(target_gap, MAX_SEQ_LEN)

        # Return 5 items!
        return input_imgs, input_clin, input_time, target_vf, target_gap
'''
'''
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class GrapeDataset(Dataset):
    def __init__(self, tensor_path, csv_path):
        """
        TRUE FORECASTING LOADER: Shifts data so Input (t) predicts Target (t+1).
        """
        print(f"   [LOADER] Loading Tensor: {tensor_path}")
        self.images = torch.load(tensor_path, weights_only=True)
        
        print(f"   [LOADER] Loading CSV: {csv_path}")
        self.clinical_df = pd.read_csv(csv_path)

        # Group and Sort
        if 'unique_id' in self.clinical_df.columns:
            self.clinical_df = self.clinical_df.sort_values('unique_id')
        self.grouped = self.clinical_df.groupby('unique_id', sort=False)
        self.patient_ids = list(self.grouped.groups.keys())
        
        # Sync Lengths
        num_images = self.images.shape[0]
        num_csv = len(self.patient_ids)
        
        if num_csv != num_images:
            print(f"   [WARNING] Data Mismatch Detected! CSV Patients: {num_csv}, Image Patients: {num_images}. Truncating.")
            if num_csv > num_images:
                self.patient_ids = self.patient_ids[:num_images]
            else:
                self.images = self.images[:num_csv]
        else:
            print(f"   [OK] Data sizes match ({num_csv} patients).")

        # Features
        feature_cols = ['Age', 'Gender', 'CCT', 'IOP', 'Category_of_Glaucoma', 'Interval_Norm']
        self.vf_cols = [str(i) for i in range(61)]
        feature_cols.extend(self.vf_cols)
        
        self.clinical_features = feature_cols
        self.target_cols = self.vf_cols

    def get_clinical_dim(self):
        return len(self.clinical_features)

    def get_output_dim(self):
        return len(self.target_cols)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # 1. Get All Data for Patient
        pid = self.patient_ids[idx]
        images_full = self.images[idx] 
        patient_data = self.grouped.get_group(pid).sort_values('Visit Number')

        # 2. Convert to Tensors
        clin_data_full = torch.tensor(patient_data[self.clinical_features].values, dtype=torch.float32)
        vf_targets_full = torch.tensor(patient_data[self.target_cols].values, dtype=torch.float32)

        # 3. THE TIME SHIFT (Forecasting Logic)
        n_visits = clin_data_full.shape[0]
        
        # --- Define Output Sequence Length ---
        MAX_SEQ_LEN = 9 # V_t -> V_t+1 reduces max sequence from 10 to 9.

        if n_visits > 1:
            # INPUT: Visits 0 to N-2 (Last input is N-1)
            input_imgs = images_full[:-1]
            input_clin = clin_data_full[:-1]
            
            # TARGET: Visits 1 to N-1 (Prediction for the next visit)
            target_vf  = vf_targets_full[1:]
        else:
            # Cannot forecast, return single sequence for padding alignment
            input_imgs = images_full
            input_clin = clin_data_full
            target_vf = vf_targets_full 
            
            # If n_visits=1, the logic below pads this single element up to 9.

        # 4. FIXED SIZE PADDING / TRUNCATION (CRITICAL FIX)
        
        # Truncate to MAX_SEQ_LEN if necessary
        input_imgs = input_imgs[:MAX_SEQ_LEN]
        input_clin = input_clin[:MAX_SEQ_LEN]
        target_vf = target_vf[:MAX_SEQ_LEN]
        
        curr_seq = input_clin.shape[0]
        pad_len = MAX_SEQ_LEN - curr_seq
        
        # Handle the one-visit patient case where curr_seq might be 0/1 depending on logic:
        if curr_seq == 0 and n_visits == 1:
            # Special case for patient with 1 visit: ensures initial shape is right for padding
            curr_seq = 1 
            pad_len = MAX_SEQ_LEN - 1
            
            # Use the single element we have for the first slot
            input_imgs = images_full[:1]
            input_clin = clin_data_full[:1]
            target_vf = vf_targets_full[:1]


        if pad_len > 0:
            # Pad Inputs
            img_pad = torch.zeros((pad_len, *input_imgs.shape[1:]))
            input_imgs = torch.cat([input_imgs, img_pad], dim=0)
            
            clin_pad = torch.zeros((pad_len, input_clin.shape[1]))
            input_clin = torch.cat([input_clin, clin_pad], dim=0)
            
            # Pad Targets
            tar_pad = torch.zeros((pad_len, target_vf.shape[1]))
            target_vf = torch.cat([target_vf, tar_pad], dim=0)
        
        # FINAL CHECK: Ensure everything is the fixed size [9, ...]
        input_imgs = input_imgs[:MAX_SEQ_LEN]
        input_clin = input_clin[:MAX_SEQ_LEN]
        target_vf = target_vf[:MAX_SEQ_LEN]
            
        return input_imgs, input_clin, target_vf
'''
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class GrapeDataset(Dataset):
    def __init__(self, tensor_path, csv_path):
        
        #TRUE FORECASTING LOADER: Shifts data so Input (t) predicts Target (t+1).
        
        print(f"   [LOADER] Loading Tensor: {tensor_path}")
        self.images = torch.load(tensor_path, weights_only=True)
        
        print(f"   [LOADER] Loading CSV: {csv_path}")
        self.clinical_df = pd.read_csv(csv_path)

        # Group and Sort
        if 'unique_id' in self.clinical_df.columns:
            self.clinical_df = self.clinical_df.sort_values('unique_id')
        self.grouped = self.clinical_df.groupby('unique_id', sort=False)
        self.patient_ids = list(self.grouped.groups.keys())
        
        # Sync Lengths
        num_images = self.images.shape[0]
        num_csv = len(self.patient_ids)
        
        if num_csv != num_images:
            print(f"   [WARNING] Data Mismatch Detected! CSV Patients: {num_csv}, Image Patients: {num_images}. Truncating.")
            if num_csv > num_images:
                self.patient_ids = self.patient_ids[:num_images]
            else:
                self.images = self.images[:num_csv]
        else:
            print(f"   [OK] Data sizes match ({num_csv} patients).")

        # Features
        feature_cols = ['Age', 'Gender', 'CCT', 'IOP', 'Category_of_Glaucoma', 'Interval_Norm']
        self.vf_cols = [str(i) for i in range(61)]
        feature_cols.extend(self.vf_cols)
        
        self.clinical_features = feature_cols
        self.target_cols = self.vf_cols

    def get_clinical_dim(self):
        return len(self.clinical_features)

    def get_output_dim(self):
        return len(self.target_cols)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # 1. Get All Data for Patient
        pid = self.patient_ids[idx]
        images_full = self.images[idx] 
        patient_data = self.grouped.get_group(pid).sort_values('Visit Number')

        # 2. Convert to Tensors
        clin_data_full = torch.tensor(patient_data[self.clinical_features].values, dtype=torch.float32)
        vf_targets_full = torch.tensor(patient_data[self.target_cols].values, dtype=torch.float32)

        # 3. THE TIME SHIFT (Forecasting Logic)
        n_visits = clin_data_full.shape[0]
        
        # --- Define Output Sequence Length ---
        MAX_SEQ_LEN = 9 # V_t -> V_t+1 reduces max sequence from 10 to 9.

        if n_visits > 1:
            # INPUT: Visits 0 to N-2 (Last input is N-1)
            input_imgs = images_full[:-1]
            input_clin = clin_data_full[:-1]
            
            # TARGET: Visits 1 to N-1 (Prediction for the next visit)
            target_vf  = vf_targets_full[1:]
        else:
            # Cannot forecast, return single sequence for padding alignment
            input_imgs = images_full
            input_clin = clin_data_full
            target_vf = vf_targets_full 
            
            # If n_visits=1, the logic below pads this single element up to 9.

        # 4. FIXED SIZE PADDING / TRUNCATION (CRITICAL FIX)
        
        # Truncate to MAX_SEQ_LEN if necessary
        input_imgs = input_imgs[:MAX_SEQ_LEN]
        input_clin = input_clin[:MAX_SEQ_LEN]
        target_vf = target_vf[:MAX_SEQ_LEN]
        
        curr_seq = input_clin.shape[0]
        pad_len = MAX_SEQ_LEN - curr_seq
        
        # Handle the one-visit patient case where curr_seq might be 0/1 depending on logic:
        if curr_seq == 0 and n_visits == 1:
            # Special case for patient with 1 visit: ensures initial shape is right for padding
            curr_seq = 1 
            pad_len = MAX_SEQ_LEN - 1
            
            # Use the single element we have for the first slot
            input_imgs = images_full[:1]
            input_clin = clin_data_full[:1]
            target_vf = vf_targets_full[:1]


        if pad_len > 0:
            # Pad Inputs
            img_pad = torch.zeros((pad_len, *input_imgs.shape[1:]))
            input_imgs = torch.cat([input_imgs, img_pad], dim=0)
            
            clin_pad = torch.zeros((pad_len, input_clin.shape[1]))
            input_clin = torch.cat([input_clin, clin_pad], dim=0)
            
            # Pad Targets
            tar_pad = torch.zeros((pad_len, target_vf.shape[1]))
            target_vf = torch.cat([target_vf, tar_pad], dim=0)
        
        # FINAL CHECK: Ensure everything is the fixed size [9, ...]
        input_imgs = input_imgs[:MAX_SEQ_LEN]
        input_clin = input_clin[:MAX_SEQ_LEN]
        target_vf = target_vf[:MAX_SEQ_LEN]
            
        return input_imgs, input_clin, target_vf

#regression code
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

import warnings
# Filter out the specific PyTorch warning
warnings.filterwarnings("ignore", category=FutureWarning)

class GrapeDataset(Dataset):
    def __init__(self, tensor_path, csv_path):
       
        # 1. LOAD DATA
        print(f"   [LOADER] Loading Tensor: {tensor_path}")
        self.images = torch.load(tensor_path, weights_only=True) # [N_patients, Max_Visits, Channels, H, W]
        
        print(f"   [LOADER] Loading CSV: {csv_path}")
        self.clinical_df = pd.read_csv(csv_path)

        # 2. GROUP BY PATIENT
        # We assume the images are sorted by ID, so we sort the CSV by ID to match.
        if 'unique_id' in self.clinical_df.columns:
            self.clinical_df = self.clinical_df.sort_values('unique_id')
            
        self.grouped = self.clinical_df.groupby('unique_id', sort=False)
        self.patient_ids = list(self.grouped.groups.keys())
        
        # --- CRITICAL FIX: SYNC LENGTHS (Prevents IndexError) ---
        num_images = self.images.shape[0]
        num_csv = len(self.patient_ids)
        
        if num_csv != num_images:
            print(f"   [WARNING] Data Mismatch Detected!")
            print(f"             > CSV Patients:    {num_csv}")
            print(f"             > Image Patients:  {num_images}")
            
            if num_csv > num_images:
                print(f"             > ACTION: Truncating CSV to first {num_images} patients.")
                self.patient_ids = self.patient_ids[:num_images]
            else:
                print(f"             > ACTION: Truncating Images to first {num_csv} patients.")
                self.images = self.images[:num_csv]
        else:
            print(f"   [OK] Data sizes match ({num_csv} patients).")
        # --------------------------------------------------------

        # 3. DEFINE INPUT FEATURES
        # Based on your CSV snippet
        feature_cols = [
            'Age', 'Gender', 'CCT', 'IOP', 
            'Category_of_Glaucoma', 'Interval_Norm'
        ]
        
        # Add VF History (Columns 0-60) as INPUTS
        self.vf_cols = [str(i) for i in range(61)]
        feature_cols.extend(self.vf_cols) 
        
        self.clinical_features = feature_cols
        
        # 4. DEFINE TARGETS (Future VF)
        self.target_cols = self.vf_cols

    def get_clinical_dim(self):
        return len(self.clinical_features)

    def get_output_dim(self):
        return len(self.target_cols)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # 1. Get Patient ID
        pid = self.patient_ids[idx]
        
        # 2. Get Images
        images = self.images[idx] 
        
        # 3. Get Clinical Data
        patient_data = self.grouped.get_group(pid)
        
        # Important: Sort visits by time/number
        patient_data = patient_data.sort_values('Visit Number')

        # Extract Inputs
        clinical_data = patient_data[self.clinical_features].values
        clinical_tensor = torch.tensor(clinical_data, dtype=torch.float32)

        # Extract Targets
        vf_targets = patient_data[self.target_cols].values
        target_tensor = torch.tensor(vf_targets, dtype=torch.float32) 
        
        # 4. Padding Logic
        max_visits = images.shape[0]
        curr_visits = clinical_tensor.shape[0]
        
        if curr_visits < max_visits:
            pad_len = max_visits - curr_visits
            clin_pad = torch.zeros((pad_len, clinical_tensor.shape[1]))
            clinical_tensor = torch.cat([clinical_tensor, clin_pad], dim=0)
            
            target_pad = torch.zeros((pad_len, target_tensor.shape[1]))
            target_tensor = torch.cat([target_tensor, target_pad], dim=0)

        if curr_visits > max_visits:
            clinical_tensor = clinical_tensor[:max_visits]
            target_tensor = target_tensor[:max_visits]

        return images, clinical_tensor, target_tensor

#classification code
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

import warnings
# Filter out the specific PyTorch warning
warnings.filterwarnings("ignore", category=FutureWarning)

class GrapeDataset(Dataset):
    def __init__(self, tensor_path, csv_path):
        
        
        
        # 1. LOAD DATA
        print(f"   [LOADER] Loading Tensor: {tensor_path}")
        self.images = torch.load(tensor_path, weights_only=True) # [N_patients, Max_Visits, Channels, H, W]
        
        print(f"   [LOADER] Loading CSV: {csv_path}")
        self.clinical_df = pd.read_csv(csv_path)

        # 2. GROUP BY PATIENT
        # We assume the images are sorted by ID, so we sort the CSV by ID to match.
        if 'unique_id' in self.clinical_df.columns:
            self.clinical_df = self.clinical_df.sort_values('unique_id')
            
        self.grouped = self.clinical_df.groupby('unique_id', sort=False)
        self.patient_ids = list(self.grouped.groups.keys())
        
        # --- CRITICAL FIX: SYNC LENGTHS (Prevents IndexError) ---
        num_images = self.images.shape[0]
        num_csv = len(self.patient_ids)
        
        if num_csv != num_images:
            print(f"   [WARNING] Data Mismatch Detected!")
            print(f"             > CSV Patients:    {num_csv}")
            print(f"             > Image Patients:  {num_images}")
            
            if num_csv > num_images:
                print(f"             > ACTION: Truncating CSV to first {num_images} patients.")
                self.patient_ids = self.patient_ids[:num_images]
            else:
                print(f"             > ACTION: Truncating Images to first {num_csv} patients.")
                self.images = self.images[:num_csv]
        else:
            print(f"   [OK] Data sizes match ({num_csv} patients).")
        # --------------------------------------------------------

        # 3. DEFINE INPUT FEATURES
        # Based on your CSV snippet
        feature_cols = [
            'Age', 'Gender', 'CCT', 'IOP', 
            'Category_of_Glaucoma', 'Interval_Norm'
        ]
        
        # Add VF History (Columns 0-60) as INPUTS
        self.vf_cols = [str(i) for i in range(61)]
        feature_cols.extend(self.vf_cols) 
        
        self.clinical_features = feature_cols
        
        # 4. DEFINE TARGETS (Future VF)
        self.target_cols = self.vf_cols

    def get_clinical_dim(self):
        return len(self.clinical_features)

    def get_output_dim(self):
        return len(self.target_cols)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # 1. Get Patient ID
        pid = self.patient_ids[idx]
        
        # 2. Get Images
        images = self.images[idx] 
        
        # 3. Get Clinical Data
        patient_data = self.grouped.get_group(pid)
        
        # Important: Sort visits by time/number
        patient_data = patient_data.sort_values('Visit Number')

        # Extract Inputs
        clinical_data = patient_data[self.clinical_features].values
        clinical_tensor = torch.tensor(clinical_data, dtype=torch.float32)

        # Extract Targets
        vf_targets = patient_data[self.target_cols].values
        target_tensor = torch.tensor(vf_targets, dtype=torch.float32) 
        
        # 4. Padding Logic
        max_visits = images.shape[0]
        curr_visits = clinical_tensor.shape[0]
        
        if curr_visits < max_visits:
            pad_len = max_visits - curr_visits
            clin_pad = torch.zeros((pad_len, clinical_tensor.shape[1]))
            clinical_tensor = torch.cat([clinical_tensor, clin_pad], dim=0)
            
            target_pad = torch.zeros((pad_len, target_tensor.shape[1]))
            target_tensor = torch.cat([target_tensor, target_pad], dim=0)

        if curr_visits > max_visits:
            clinical_tensor = clinical_tensor[:max_visits]
            target_tensor = target_tensor[:max_visits]

        return images, clinical_tensor, target_tensor
    """