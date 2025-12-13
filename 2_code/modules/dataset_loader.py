import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class GrapeDataset(Dataset):
    def __init__(self, tensor_path, csv_path):
        """
        Custom Dataset to load Images, Clinical Data, and Labels.
        """
        print(f"   [LOADER] Loading Tensor: {tensor_path}")
        try:
            self.images = torch.load(tensor_path)
            num_tensor_patients = self.images.shape[0]
            print(f"   [LOADER] Tensor loaded. Found {num_tensor_patients} unique patients.")
        except Exception as e:
            print(f"[ERROR] Could not load tensor at {tensor_path}: {e}")
            raise e
        
        # 1. Load CSV
        print(f"   [LOADER] Loading CSV: {csv_path}")
        raw_df = pd.read_csv(csv_path)
        
        # 2. GROUP BY PATIENT (The Fix)
        # We collapse the multiple visits into single patient rows to match the Tensor
        # sort=True ensures we match the order used during tensor creation
        self.df = raw_df.groupby('unique_id', as_index=False).first()
        
        num_csv_patients = len(self.df)
        print(f"   [LOADER] CSV collapsed. Found {num_csv_patients} unique patients.")
        
        # 3. Validation
        if num_csv_patients != num_tensor_patients:
            print(f"   [WARNING] Mismatch! Tensor has {num_tensor_patients}, CSV has {num_csv_patients}.")
            # We truncate to the smaller number to prevent crashing
            limit = min(num_csv_patients, num_tensor_patients)
            self.df = self.df.iloc[:limit]
            self.images = self.images[:limit]
            print(f"   [FIX] Truncated both to {limit} samples.")
        
        # 4. Process Clinical Columns
        drop_cols = ['unique_id', 'Visit Number', 'Interval Years', 'Corresponding CFP', 'Interval_Years_Raw', 'Progression_Flag']
        self.clinical_cols = [c for c in self.df.columns if c not in drop_cols]
        
        self.clinical_data = self.df[self.clinical_cols].values.astype(np.float32)
        
        # 5. Process Labels
        if 'Progression_Flag' not in self.df.columns:
            raise ValueError("CSV must contain 'Progression_Flag' column.")
        self.labels = self.df['Progression_Flag'].values.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_seq = self.images[idx]
        clinical_vec = torch.tensor(self.clinical_data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image_seq, clinical_vec, label

    def get_clinical_dim(self):
        return len(self.clinical_cols)