import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class GrapeDataset(Dataset):
    def __init__(self, tensor_path, csv_path, augment=False):
        """
        augment=True: Applies random flips and noise (Use for Training)
        augment=False: Returns data exactly as is (Use for Validation)
        """
        self.augment = augment
        
        print(f"   [LOADER] Loading Tensor: {tensor_path}")
        try:
            self.images = torch.load(tensor_path)
            num_tensor_patients = self.images.shape[0]
            print(f"   [LOADER] Tensor loaded. Found {num_tensor_patients} unique patients.")
        except Exception as e:
            print(f"[ERROR] Could not load tensor at {tensor_path}: {e}")
            raise e
        
        # Load CSV and Group by Patient
        print(f"   [LOADER] Loading CSV: {csv_path}")
        raw_df = pd.read_csv(csv_path)
        self.df = raw_df.groupby('unique_id', as_index=False).first()
        
        # Validate Sizes
        num_csv_patients = len(self.df)
        if num_csv_patients != num_tensor_patients:
            limit = min(num_csv_patients, num_tensor_patients)
            self.df = self.df.iloc[:limit]
            self.images = self.images[:limit]
            print(f"   [FIX] Truncated both to {limit} samples.")
        
        # Clinical Columns
        drop_cols = ['unique_id', 'Visit Number', 'Interval Years', 'Corresponding CFP', 'Interval_Years_Raw', 'Progression_Flag']
        self.clinical_cols = [c for c in self.df.columns if c not in drop_cols]
        self.clinical_data = self.df[self.clinical_cols].values.astype(np.float32)
        
        # Labels
        if 'Progression_Flag' not in self.df.columns:
            raise ValueError("CSV must contain 'Progression_Flag' column.")
        self.labels = self.df['Progression_Flag'].values.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # A. Get Image Sequence [Visits, 3, 224, 224]
        image_seq = self.images[idx].clone() # Clone to avoid modifying original tensor
        
        # B. Get Clinical Data
        clinical_vec = torch.tensor(self.clinical_data[idx]).clone()
        
        # --- AUGMENTATION (Training Only) ---
        if self.augment:
            # 1. Image Augmentation: Random Horizontal Flip
            # Iterate through visits and flip with 50% chance
            if random.random() > 0.5:
                image_seq = torch.flip(image_seq, dims=[-1]) # Flip width dimension
            
            # 2. Clinical Augmentation: Add tiny noise (jitter)
            # Add noise +/- 1% of the value
            noise = torch.randn_like(clinical_vec) * 0.01
            clinical_vec += noise

        # C. Get Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image_seq, clinical_vec, label

    def get_clinical_dim(self):
        return len(self.clinical_cols)