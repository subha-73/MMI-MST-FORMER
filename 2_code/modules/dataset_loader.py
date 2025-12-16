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
"""
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