import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# 1. CONFIGURATION
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RAW_DATA_PATH = os.path.join(ROOT_DIR, '1_data', 'raw', 'grape')
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
EXCEL_FILE = 'VF_clinical_information.xlsx'

print(f"--- STARTING ENHANCED PRE-PROCESSING ---")
print(f"File: {EXCEL_FILE}")

# ==========================================
# 2. DATA LOADER UTILITIES
# ==========================================

def clean_header_name(val):
    s = str(val).strip()
    if s.lower() == 'nan' or s == '': return None
    try:
        f = float(s)
        if f.is_integer(): return str(int(f))
    except: pass
    return s

def load_sheet_robust(file_path, sheet_name):
    try:
        header_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=2)
    except Exception as e:
        raise ValueError(f"Could not read sheet '{sheet_name}': {e}")

    new_columns = []
    row0 = header_df.iloc[0]
    row1 = header_df.iloc[1]

    for r0, r1 in zip(row0, row1):
        h0 = clean_header_name(r0)
        h1 = clean_header_name(r1)
        if h1 is not None: new_columns.append(h1)
        elif h0 is not None: new_columns.append(h0)
        else: new_columns.append("UNKNOWN")

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=2)
    df.columns = new_columns
    df.columns = df.columns.astype(str)
    return df

def get_col(df, keyword):
    matches = [c for c in df.columns if keyword.lower() in c.lower()]
    return matches[0] if matches else None

# ==========================================
# 3. MAIN DATA ENGINE
# ==========================================

def load_grape_data():
    print("\n--- STEP 1: LOADING & MERGING ---")
    file_path = os.path.join(RAW_DATA_PATH, EXCEL_FILE)
    if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load Sheets
    base_df = load_sheet_robust(file_path, 'Baseline')
    follow_df = load_sheet_robust(file_path, 'Follow-up')

    # 2. Create Unique ID
    subj_base = get_col(base_df, 'Subject')
    lat_base = get_col(base_df, 'Laterality')
    base_df['unique_id'] = base_df[subj_base].astype(str) + '_' + base_df[lat_base].astype(str)

    subj_fol = get_col(follow_df, 'Subject')
    lat_fol = get_col(follow_df, 'Laterality')
    follow_df['unique_id'] = follow_df[subj_fol].astype(str) + '_' + follow_df[lat_fol].astype(str)

    # 3. Identify Feature Columns
    vf_cols_base = [str(i) for i in range(61) if str(i) in base_df.columns]
    rnfl_cols = [c for c in ['Mean', 'S', 'N', 'I', 'T'] if c in base_df.columns]
    
    # NEW: Identify Additional Static Features
    age_col = get_col(base_df, 'Age')
    gender_col = get_col(base_df, 'Gender')
    cct_col = get_col(base_df, 'CCT')
    cat_col = get_col(base_df, 'Category') # Finds 'Category of Glaucoma'

    extra_static_map = {
        age_col: 'Age', 
        gender_col: 'Gender', 
        cct_col: 'CCT', 
        cat_col: 'Category_of_Glaucoma'
    }
    
    print(f"  - VF Columns: {len(vf_cols_base)}")
    print(f"  - Found Extra Features: {[v for k,v in extra_static_map.items() if k]}")

    # 4. Progression Logic
    potential_labels = ['MD', 'PLR2', 'PLR3']
    found_labels = [c for c in potential_labels if c in base_df.columns]
    if found_labels:
        for col in found_labels:
            base_df[col] = pd.to_numeric(base_df[col], errors='coerce').fillna(0)
        base_df['Progression_Flag'] = base_df[found_labels].max(axis=1).astype(int)
    else:
        raise ValueError("CRITICAL: No Progression labels found!")

    # 5. Prepare Static Data
    base_df['Visit Number'] = 0.0
    base_df['Interval Years'] = 0.0
    
    # Rename extra columns in Baseline for consistency
    base_df.rename(columns=extra_static_map, inplace=True)
    
    # Select all static columns to propagate
    static_cols = ['unique_id', 'Progression_Flag', 'Age', 'Gender', 'CCT', 'Category_of_Glaucoma'] + rnfl_cols
    # Filter to ensure they exist
    static_cols = [c for c in static_cols if c in base_df.columns]
    
    patient_static = base_df[static_cols].copy()

    # 6. Normalize Dynamic Columns
    col_map = {'Interval': 'Interval Years', 'IOP': 'IOP', 'CFP': 'Corresponding CFP'}
    def normalize_cols(df):
        rename_dict = {}
        for key, target in col_map.items():
            found = get_col(df, key)
            if found: rename_dict[found] = target
        df.rename(columns=rename_dict, inplace=True)
        return df

    base_df = normalize_cols(base_df)
    follow_df = normalize_cols(follow_df)

    # 7. Stack & Merge
    common_cols = ['unique_id', 'Visit Number', 'Interval Years', 'IOP', 'Corresponding CFP']
    vf_cols_follow = [c for c in follow_df.columns if c in vf_cols_base]
    
    stack_base = base_df[[c for c in (common_cols + vf_cols_base) if c in base_df.columns]]
    stack_fol = follow_df[[c for c in (common_cols + vf_cols_follow) if c in follow_df.columns]]

    full_seq = pd.concat([stack_base, stack_fol], ignore_index=True)
    final_df = pd.merge(full_seq, patient_static, on='unique_id', how='left')
    final_df = final_df.sort_values(by=['unique_id', 'Interval Years']).reset_index(drop=True)

    return final_df, vf_cols_base, rnfl_cols

# ==========================================
# 4. REPORTING & CLEANING ENGINE
# ==========================================

def print_stats(df, label, features):
    print(f"\n[REPORT] DATA STATISTICS: {label}")
    print(f"{'Feature':<20} | {'Missing':<10} | {'Mean / Mode':<15} | {'Min':<10} | {'Max':<10}")
    print("-" * 75)
    
    for feat in features:
        if feat not in df.columns: continue
        
        missing = df[feat].isnull().sum()
        if pd.api.types.is_numeric_dtype(df[feat]):
            mean_val = f"{df[feat].mean():.2f}"
            min_val = f"{df[feat].min():.2f}"
            max_val = f"{df[feat].max():.2f}"
        else:
            mean_val = str(df[feat].mode()[0])[:10]
            min_val = "N/A"
            max_val = "N/A"
            
        print(f"{feat:<20} | {missing:<10} | {mean_val:<15} | {min_val:<10} | {max_val:<10}")

def clean_and_normalize(df, vf_cols, rnfl_cols):
    print("\n--- STEP 2: CLEANING & NORMALIZING ---")
    
    # Define features to track for the report
    track_feats = ['IOP', 'Age', 'CCT', 'Interval Years']
    if 'Mean' in rnfl_cols: track_feats.append('Mean') # RNFL Mean
    
    # 1. REPORT BEFORE CLEANING
    print_stats(df, "BEFORE PREPROCESSING", track_feats)

    # 2. SAVE RAW TIME
    df['Interval_Years_Raw'] = pd.to_numeric(df['Interval Years'], errors='coerce').fillna(0.0)
    
    # 3. ENCODE CATEGORICALS (Gender, Category)
    # Gender: M/F -> 0/1
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.upper().map({'M': 0, 'F': 1, 'MALE': 0, 'FEMALE': 1}).fillna(0)
    
    # Category: Text -> Numeric ID
    if 'Category_of_Glaucoma' in df.columns:
        le = LabelEncoder()
        # Convert to string, fill NaNs with 'Unknown'
        df['Category_of_Glaucoma'] = df['Category_of_Glaucoma'].astype(str).fillna('Unknown')
        df['Category_of_Glaucoma'] = le.fit_transform(df['Category_of_Glaucoma'])
        print(f"  - Encoded Glaucoma Categories: {list(le.classes_)}")

    # 4. FORCE NUMERIC
    numeric_feats = ['IOP', 'Age', 'CCT'] + vf_cols + rnfl_cols
    # Filter only existing
    numeric_feats = [c for c in numeric_feats if c in df.columns]

    for col in numeric_feats:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. INTERPOLATION (Time Series Fill)
    interp_cols = ['IOP'] + [c for c in vf_cols if c in df.columns]
    df[interp_cols] = df.groupby('unique_id')[interp_cols].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    
    # 6. GLOBAL FILL (Safety Net)
    df[numeric_feats] = df[numeric_feats].fillna(df[numeric_feats].mean())

    # 7. Z-SCORE NORMALIZATION
    # We normalize continuous variables: IOP, Age, CCT, RNFL, VF
    # We do NOT normalize Gender, Category, or Flags (they are categorical/binary)
    scaler = StandardScaler()
    df[numeric_feats] = scaler.fit_transform(df[numeric_feats])
    
    # Normalize Time Separately
    time_scaler = StandardScaler()
    df['Interval_Norm'] = time_scaler.fit_transform(df[['Interval_Years_Raw']])

    # 8. REPORT AFTER CLEANING
    print_stats(df, "AFTER PREPROCESSING (Normalized)", track_feats)
    
    return df

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    try:
        # Load
        df, vf_cols, rnfl_cols = load_grape_data()
        
        # Clean
        final_df = clean_and_normalize(df, vf_cols, rnfl_cols)
        
        # Save
        if not os.path.exists(PROCESSED_DIR): os.makedirs(PROCESSED_DIR)
        save_path = os.path.join(PROCESSED_DIR, 'grape_clinical_processed.csv')
        final_df.to_csv(save_path, index=False)
        
        print(f"\nSUCCESS! File saved to: {save_path}")
        print(f"Total Columns: {len(final_df.columns)}")
        print(f"Columns Added: Age, Gender, CCT, Category_of_Glaucoma")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()