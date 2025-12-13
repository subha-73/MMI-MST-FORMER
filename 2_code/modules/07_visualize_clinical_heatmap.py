import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. SETUP
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
CSV_PATH = os.path.join(PROCESSED_DIR, 'grape_train.csv')

print(f"{'='*70}")
print(f"MODULE: VISUALIZE CLINICAL DATA")
print(f"{'='*70}")

def visualize_clinical():
    # --- INPUT ---
    print("\n[INPUT] Loading CSV...")
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] File not found: {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    print(f"   Source File: {CSV_PATH}")
    print(f"   Total Rows: {len(df)}")
    print(f"   Total Columns: {len(df.columns)}")

    # --- PROCESSING ---
    print("\n[PROCESSING] Calculating Correlations...")
    cols_to_plot = ['Age', 'CCT', 'IOP', 'Mean', 'S', 'N', 'I', 'T'] 
    vf_samples = [c for c in df.columns if c in ['1', '15', '30', '45']]
    cols_to_plot += vf_samples
    
    existing_cols = [c for c in cols_to_plot if c in df.columns]
    subset = df[existing_cols]
    
    corr_matrix = subset.corr()
    print(f"   Correlation Matrix Size: {corr_matrix.shape}")
    print("   (Subset of key features selected for clarity)")

    # --- OUTPUT (DELIVERABLE) ---
    print("\n[OUTPUT] Generating Heatmap...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Intermediate Deliverable: Clinical Correlations', fontsize=16)
    
    save_path = os.path.join(ROOT_DIR, 'deliverable_clinical_heatmap.png')
    plt.savefig(save_path)
    print(f"   ✓ SAVED FILE: {save_path}")
    print("   ✓ Displaying Plot...")
    plt.show()

if __name__ == "__main__":
    visualize_clinical()