import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')

# Must match the image sequence length
MAX_SEQ_LEN = 9

print("=" * 70)
print("GRAPE GLAUCOMA - CLINICAL TENSOR GENERATION")
print("=" * 70)
print(f"Reading CSVs from: {PROCESSED_DIR}")

# ==========================================
# 2. SEQUENCE TENSOR GENERATION
# ==========================================

def get_clinical_feature_list(df):
    """
    Dynamically determine the list of numerical clinical features to use.
    """
    feature_candidates = [
        'IOP', 'Age', 'CCT', 'Interval_Norm', 'Gender',
        'Category_of_Glaucoma', 'Diagnosis',
        'Mean', 'S', 'N', 'I', 'T'  # RNFL features
    ]

    # Include all VF columns (0–60) if present
    vf_cols = [str(i) for i in range(61) if str(i) in df.columns]

    final_features = [c for c in feature_candidates + vf_cols if c in df.columns]
    return final_features


def create_clinical_tensors(csv_filename, output_prefix):
    """
    Reads a processed CSV, extracts clinical features, groups them
    into padded sequences, and saves tensors + labels.
    """
    csv_path = os.path.join(PROCESSED_DIR, csv_filename)
    if not os.path.exists(csv_path):
        print(f"\n[SKIP] Could not find {csv_filename}")
        return

    print(f"\nProcessing {csv_filename} ...")
    df = pd.read_csv(csv_path)

    # Feature selection
    clinical_features = get_clinical_feature_list(df)
    print(f"  ✓ Found {len(clinical_features)} clinical features")

    unique_ids = df['unique_id'].unique()

    all_patient_clin = []
    patient_labels = []

    for pid in tqdm(unique_ids, desc="Building Clinical Tensors"):
        visits = df[df['unique_id'] == pid].sort_values('Interval Years')

        # Enforce max sequence length
        visits = visits.head(MAX_SEQ_LEN)

        feature_matrix = visits[clinical_features].values
        label = visits['Progression_Flag'].iloc[0]

        current_len = len(visits)
        pad_len = MAX_SEQ_LEN - current_len

        if pad_len > 0:
            padding = np.zeros((pad_len, len(clinical_features)))
            final_seq = np.concatenate([feature_matrix, padding], axis=0)
        else:
            final_seq = feature_matrix

        all_patient_clin.append(
            torch.tensor(final_seq, dtype=torch.float32)
        )
        patient_labels.append(label)

    if len(all_patient_clin) == 0:
        print("  [ERROR] No valid clinical sequences found!")
        return

    # Stack tensors
    big_tensor_clin = torch.stack(all_patient_clin)
    big_tensor_labels = torch.tensor(
        patient_labels, dtype=torch.long
    ).unsqueeze(1)

    # Save
    out_clin_path = os.path.join(PROCESSED_DIR, f"{output_prefix}_clinical.pt")
    out_label_path = os.path.join(PROCESSED_DIR, f"{output_prefix}_labels.pt")

    torch.save(big_tensor_clin, out_clin_path)
    torch.save(big_tensor_labels, out_label_path)

    print(f"  ✓ Saved Clinical Tensor: {out_clin_path}")
    print(f"    Shape: {big_tensor_clin.shape}")
    print(f"  ✓ Saved Labels: {out_label_path}")
    print(f"    Shape: {big_tensor_labels.shape}")


# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    splits = [
        ('grape_train.csv', 'grape_train'),
        ('grape_val.csv', 'grape_val'),
        ('grape_test.csv', 'grape_test')
    ]

    for csv_file, out_prefix in splits:
        create_clinical_tensors(csv_file, out_prefix)

    print("=" * 70)
    print("CLINICAL SEQUENCE GENERATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
