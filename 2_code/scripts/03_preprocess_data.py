'''
import pandas as pd
import numpy as np
import os
import copy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Assuming the following directory structure relative to the script:
# ../../1_data/raw/grape/VF_clinical_information.xlsx
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RAW_DATA_PATH = os.path.join(ROOT_DIR, '1_data', 'raw', 'grape')
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
EXCEL_FILE = 'VF_clinical_information.xlsx'

print(f"{'='*70}")
print(f"GRAPE GLAUCOMA FORECASTING - CLINICAL PREPROCESSING (WITH SPLIT)")
print(f"{'='*70}")
print(f"File: {EXCEL_FILE}")

# ==========================================
# 2. DATA LOADER UTILITIES
# ==========================================

def clean_header_name(val):
    """Convert Excel float headers (1.0 -> 1)"""
    s = str(val).strip()
    if s.lower() == 'nan' or s == '': 
        return None
    try:
        f = float(s)
        if f.is_integer(): 
            return str(int(f))
    except: 
        pass
    return s

def load_sheet_robust(file_path, sheet_name):
    """Load Excel sheet with 2-row header structure"""
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
        if h1 is not None: 
            new_columns.append(h1)
        elif h0 is not None: 
            new_columns.append(h0)
        else: 
            new_columns.append("UNKNOWN")

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=2)
    df.columns = new_columns
    df.columns = df.columns.astype(str)
    return df

def get_col(df, keyword):
    """Find column by keyword (case-insensitive)"""
    matches = [c for c in df.columns if keyword.lower() in c.lower()]
    return matches[0] if matches else None

# ==========================================
# 3. MAIN DATA ENGINE (LOADER) - CRITICAL FIX APPLIED HERE
# ==========================================

def load_grape_data():
    """Load, merge, and prepare RAW GRAPE dataset"""
    print("\n--- STEP 1: LOADING & MERGING DATA ---")
    file_path = os.path.join(RAW_DATA_PATH, EXCEL_FILE)
    if not os.path.exists(file_path): 
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load Sheets
    base_df = load_sheet_robust(file_path, 'Baseline')
    follow_df = load_sheet_robust(file_path, 'Follow-up')
    print(f"✓ Loaded Baseline: {base_df.shape[0]} patients")
    print(f"✓ Loaded Follow-up: {follow_df.shape[0]} records")

    # Create Unique ID (Patient_Eye)
    subj_base = get_col(base_df, 'Subject')
    lat_base = get_col(base_df, 'Laterality')
    base_df['unique_id'] = base_df[subj_base].astype(str) + '_' + base_df[lat_base].astype(str)

    subj_fol = get_col(follow_df, 'Subject')
    lat_fol = get_col(follow_df, 'Laterality')
    follow_df['unique_id'] = follow_df[subj_fol].astype(str) + '_' + follow_df[lat_fol].astype(str)

    # Identify Feature Columns
    vf_cols_base = [str(i) for i in range(61) if str(i) in base_df.columns]
    rnfl_cols = [c for c in ['Mean', 'S', 'N', 'I', 'T'] if c in base_df.columns]
    
    # Find Static Features
    age_col = get_col(base_df, 'Age')
    gender_col = get_col(base_df, 'Gender')
    cct_col = get_col(base_df, 'CCT')
    cat_col = get_col(base_df, 'Category')
    dx_col = get_col(base_df, 'Diagnosis')

    extra_static_map = {
        age_col: 'Age', 
        gender_col: 'Gender', 
        cct_col: 'CCT', 
        cat_col: 'Category_of_Glaucoma',
        dx_col: 'Diagnosis'
    }
    extra_static_map = {k: v for k, v in extra_static_map.items() if k}
    
    print(f"  VF Columns Found: {len(vf_cols_base)}")
    print(f"  RNFL Columns: {rnfl_cols}")
    print(f"  Static Features: {list(extra_static_map.values())}")

    # =================================================================
    # START: CRITICAL PROGRESSION LOGIC FIX (Matching MST-Former Forecasting)
    # This Progression Flag is the final outcome for the patient's entire sequence.
    # =================================================================
    potential_labels = ['MD', 'PLR2', 'PLR3']
    found_labels = [c for c in potential_labels if c in base_df.columns]

    if found_labels:
        # 1. Ensure the flag is numeric and treat missing as 0 (Non-Progressed)
        for col in found_labels:
            base_df[col] = pd.to_numeric(base_df[col], errors='coerce').fillna(0)
            
        # 2. Determine the definitive patient outcome (1 if any indicator is positive)
        # We use .clip(upper=1) to ensure the flag is strictly binary (0 or 1).
        base_df['Progression_Flag'] = base_df[found_labels].max(axis=1).clip(upper=1).astype(int)
        
        # Confirmation
        print(f"  Progression Indicators used for final label: {found_labels}")
        prog_count = base_df['Progression_Flag'].sum()
        total_count = len(base_df)
        print(f"  Patient Progression Flag (Static Outcome) loaded successfully.")
        print(f"  Progression Ratio: {prog_count} / {total_count} = {prog_count/total_count*100:.2f}% (CONFIRMED: Severe Class Imbalance)")
    else:
        raise ValueError("CRITICAL: No Progression labels (MD, PLR2, PLR3) found!")
    # =================================================================
    # END: CRITICAL PROGRESSION LOGIC FIX
    # =================================================================

    # Prepare Static Data
    base_df['Visit Number'] = 0.0
    base_df['Interval Years'] = 0.0
    base_df.rename(columns=extra_static_map, inplace=True)
    
    static_cols = ['unique_id', 'Progression_Flag', 'Age', 'Gender', 'CCT', 
                   'Category_of_Glaucoma', 'Diagnosis'] + rnfl_cols
    static_cols = [c for c in static_cols if c in base_df.columns]
    
    # Store static info (like the final Progression Flag) per patient ID
    patient_static = base_df[static_cols].copy()

    # Normalize Dynamic Columns
    col_map = {'Interval': 'Interval Years', 'IOP': 'IOP', 'CFP': 'Corresponding CFP'}
    def normalize_cols(df):
        rename_dict = {}
        for key, target in col_map.items():
            found = get_col(df, key)
            if found: 
                rename_dict[found] = target
        df.rename(columns=rename_dict, inplace=True)
        return df

    base_df = normalize_cols(base_df)
    follow_df = normalize_cols(follow_df)

    # Stack & Merge
    common_cols = ['unique_id', 'Visit Number', 'Interval Years', 'IOP', 'Corresponding CFP']
    vf_cols_follow = [c for c in follow_df.columns if c in vf_cols_base]
    
    stack_base = base_df[[c for c in (common_cols + vf_cols_base) if c in base_df.columns]]
    stack_fol = follow_df[[c for c in (common_cols + vf_cols_follow) if c in follow_df.columns]]

    full_seq = pd.concat([stack_base, stack_fol], ignore_index=True)
    # Merge the static Progression Flag onto every visit record
    final_df = pd.merge(full_seq, patient_static, on='unique_id', how='left')
    final_df = final_df.sort_values(by=['unique_id', 'Interval Years']).reset_index(drop=True)

    return final_df, vf_cols_base, rnfl_cols

# ==========================================
# 4. PREPROCESSING ENGINE (CLASS-BASED TO PREVENT LEAKAGE)
# ==========================================

class ClinicalPreprocessor:
    def __init__(self, vf_cols, rnfl_cols):
        self.vf_cols = vf_cols
        self.rnfl_cols = rnfl_cols
        
        # Placeholders for statistics learned from Training Set
        self.numeric_means = {}
        self.scalers = {}  # {col_name: StandardScaler}
        self.encoders = {} # {col_name: LabelEncoder}
        self.numeric_feats = []
        
        # Track report features
        self.track_feats = ['IOP', 'Age', 'CCT', 'Interval Years']
        if 'Mean' in rnfl_cols: self.track_feats.append('Mean')

    def print_stats(self, df, label):
        """Print statistics for reporting"""
        print(f"\n[REPORT] {label}")
        print(f"{'Feature':<20} | {'Missing':<12} | {'Mean':<12} | {'Std':<12} | {'Min':<10} | {'Max':<10}")
        print("-" * 90)
        
        for feat in self.track_feats:
            if feat not in df.columns: continue
            
            missing = df[feat].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            
            if pd.api.types.is_numeric_dtype(df[feat]):
                mean_val = f"{df[feat].mean():.2f}"
                std_val = f"{df[feat].std():.2f}"
                min_val = f"{df[feat].min():.2f}"
                max_val = f"{df[feat].max():.2f}"
            else:
                mean_val = "N/A"
                std_val = "N/A"
                min_val = "N/A"
                max_val = "N/A"
                
            print(f"{feat:<20} | {missing:>3} ({missing_pct:>5.1f}%) | {mean_val:<12} | {std_val:<12} | {min_val:<10} | {max_val:<10}")

    def _prepare_raw(self, df):
        """Steps common to both Train and Test (Safe operations)"""
        df = df.copy()
        
        # 1. Save Raw Time
        df['Interval_Years_Raw'] = pd.to_numeric(df['Interval Years'], errors='coerce').fillna(0.0)
        
        # 2. Define Numeric Features
        self.numeric_feats = ['IOP', 'Age', 'CCT'] + self.vf_cols + self.rnfl_cols
        self.numeric_feats = [c for c in self.numeric_feats if c in df.columns]
        
        # 3. Force Numeric
        for col in self.numeric_feats:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df

    def _interpolate_per_patient(self, df):
        """Safe to do per split or globally, as it is within-patient"""
        interp_cols = ['IOP'] + [c for c in self.vf_cols if c in df.columns]
        interp_cols = [c for c in interp_cols if c in df.columns]
        
        df[interp_cols] = df.groupby('unique_id')[interp_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        return df

    def fit_transform(self, df):
        """LEARN stats from TRAIN set and apply them"""
        print("\n--- FITTING PREPROCESSOR ON TRAINING DATA ---")
        self.print_stats(df, "TRAIN SET - BEFORE PREPROCESSING")
        
        df = self._prepare_raw(df)
        
        # 1. Encode Categoricals (Learn Mappings)
        if 'Gender' in df.columns:
            # Simple map doesn't need fitting, but good to be explicit
            gender_map = {'M': 0, 'F': 1, 'MALE': 0, 'FEMALE': 1}
            df['Gender'] = df['Gender'].astype(str).str.upper().map(gender_map).fillna(0).astype(int)
            
        if 'Category_of_Glaucoma' in df.columns:
            le = LabelEncoder()
            # Convert to string, fit
            df['Category_of_Glaucoma'] = df['Category_of_Glaucoma'].astype(str).fillna('Unknown')
            df['Category_of_Glaucoma'] = le.fit_transform(df['Category_of_Glaucoma'])
            self.encoders['Category'] = le
            print(f"  ✓ Learned Categories: {list(le.classes_)}")

        if 'Diagnosis' in df.columns:
            le_diag = LabelEncoder()
            df['Diagnosis'] = df['Diagnosis'].astype(str).fillna('Unknown')
            df['Diagnosis'] = le_diag.fit_transform(df['Diagnosis'])
            self.encoders['Diagnosis'] = le_diag

        # 2. Interpolate
        df = self._interpolate_per_patient(df)
        
        # 3. Learn Global Means for Filling Gaps
        for col in self.numeric_feats:
            mean_val = df[col].mean()
            self.numeric_means[col] = mean_val
            df[col] = df[col].fillna(mean_val)
        print(f"  ✓ Learned global means from Training set")

        # 4. Learn Scaling (Z-Score)
        # Norm feats excludes flags/IDs
        norm_feats = [c for c in self.numeric_feats if c not in ['Progression_Flag']]
        self.scaler = StandardScaler()
        df[norm_feats] = self.scaler.fit_transform(df[norm_feats])
        
        # Learn Time Scaling
        self.time_scaler = StandardScaler()
        df['Interval_Norm'] = self.time_scaler.fit_transform(df[['Interval_Years_Raw']])
        
        print(f"  ✓ Learned scaling parameters from Training set")
        self.print_stats(df, "TRAIN SET - AFTER PREPROCESSING")
        return df

    def transform(self, df, dataset_name="TEST"):
        """APPLY saved stats to VAL/TEST set (No Peeking!)"""
        print(f"\n--- TRANSFORMING {dataset_name} DATA (Using Train Stats) ---")
        self.print_stats(df, f"{dataset_name} - BEFORE")
        
        df = self._prepare_raw(df)
        
        # 1. Apply Categoricals
        if 'Gender' in df.columns:
            gender_map = {'M': 0, 'F': 1, 'MALE': 0, 'FEMALE': 1}
            df['Gender'] = df['Gender'].astype(str).str.upper().map(gender_map).fillna(0).astype(int)
            
        # Helper to safely encode unseen labels
        def safe_transform(encoder, series):
            # Map unknown classes to the first class (usually 'Unknown')
            known_classes = set(encoder.classes_)
            series = series.apply(lambda x: x if x in known_classes else encoder.classes_[0])
            return encoder.transform(series)

        if 'Category_of_Glaucoma' in df.columns and 'Category' in self.encoders:
            df['Category_of_Glaucoma'] = df['Category_of_Glaucoma'].astype(str).fillna('Unknown')
            df['Category_of_Glaucoma'] = safe_transform(self.encoders['Category'], df['Category_of_Glaucoma'])

        if 'Diagnosis' in df.columns and 'Diagnosis' in self.encoders:
            df['Diagnosis'] = df['Diagnosis'].astype(str).fillna('Unknown')
            df['Diagnosis'] = safe_transform(self.encoders['Diagnosis'], df['Diagnosis'])

        # 2. Interpolate
        df = self._interpolate_per_patient(df)
        
        # 3. Fill Missing using TRAIN Means (Prevent Leakage)
        for col in self.numeric_feats:
            if col in self.numeric_means:
                df[col] = df[col].fillna(self.numeric_means[col])
        
        # 4. Apply Scaling using TRAIN Scaler
        norm_feats = [c for c in self.numeric_feats if c not in ['Progression_Flag']]
        df[norm_feats] = self.scaler.transform(df[norm_feats])
        
        # Apply Time Scaling
        df['Interval_Norm'] = self.time_scaler.transform(df[['Interval_Years_Raw']])
        
        self.print_stats(df, f"{dataset_name} - AFTER")
        return df

# ==========================================
# 5. DATASET SPLITTER
# ==========================================

def split_patients(df):
    """Split dataset by Unique ID to prevent leakage"""
    print("\n" + "="*70)
    print("STEP 2: SPLITTING DATASET (Patient-Level)")
    print("="*70)
    
    unique_ids = df['unique_id'].unique()
    print(f"Total Unique Patients: {len(unique_ids)}")
    
    # Split 70% Train, 30% Temp
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42, shuffle=True)
    # Split Temp into 15% Val, 15% Test
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42, shuffle=True)
    
    print(f"  Train Patients: {len(train_ids)} ({len(train_ids)/len(unique_ids):.1%})")
    print(f"  Val Patients:   {len(val_ids)} ({len(val_ids)/len(unique_ids):.1%})")
    print(f"  Test Patients:  {len(test_ids)} ({len(test_ids)/len(unique_ids):.1%})")
    
    train_df = df[df['unique_id'].isin(train_ids)].copy()
    val_df = df[df['unique_id'].isin(val_ids)].copy()
    test_df = df[df['unique_id'].isin(test_ids)].copy()
    
    return train_df, val_df, test_df

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

def main():
    try:
        # 1. LOAD RAW DATA
        df_raw, vf_cols, rnfl_cols = load_grape_data()
        
        # 2. SPLIT RAW DATA
        train_raw, val_raw, test_raw = split_patients(df_raw)
        
        # 3. PREPROCESS (FIT on Train, TRANSFORM Val/Test)
        print("\n" + "="*70)
        print("STEP 3: PREPROCESSING PIPELINE")
        print("="*70)
        
        processor = ClinicalPreprocessor(vf_cols, rnfl_cols)
        
        # Fit & Process Train
        train_proc = processor.fit_transform(train_raw)
        
        # Process Val & Test (Using Train stats)
        val_proc = processor.transform(val_raw, "VALIDATION")
        test_proc = processor.transform(test_raw, "TEST")
        
        # 4. SAVE
        print("\n" + "="*70)
        print("STEP 4: SAVING DATASETS")
        print("="*70)
        
        # Create directories if needed
        if not os.path.exists(PROCESSED_DIR): 
            os.makedirs(PROCESSED_DIR)
            
        train_proc.to_csv(os.path.join(PROCESSED_DIR, 'grape_train.csv'), index=False)
        val_proc.to_csv(os.path.join(PROCESSED_DIR, 'grape_val.csv'), index=False)
        test_proc.to_csv(os.path.join(PROCESSED_DIR, 'grape_test.csv'), index=False)
        
        # Save Full for reference (Optional)
        full_proc = pd.concat([train_proc, val_proc, test_proc], axis=0)
        full_proc.to_csv(os.path.join(PROCESSED_DIR, 'grape_clinical_full_processed.csv'), index=False)

        print(f"\nSUCCESS! Files saved to {PROCESSED_DIR}.")
        
        # Metadata
        with open(os.path.join(PROCESSED_DIR, 'dataset_stats.txt'), 'w') as f:
            f.write(f"Train Samples: {len(train_proc)}\n")
            f.write(f"Val Samples: {len(val_proc)}\n")
            f.write(f"Test Samples: {len(test_proc)}\n")
            f.write(f"Features: {len(train_proc.columns)}\n")

    except Exception as e:
        print(f"\n [CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
# working code 
import pandas as pd
import numpy as np
import os
import copy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RAW_DATA_PATH = os.path.join(ROOT_DIR, '1_data', 'raw', 'grape')
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
EXCEL_FILE = 'VF_clinical_information.xlsx'

print(f"{'='*70}")
print(f"GRAPE GLAUCOMA FORECASTING - CLINICAL PREPROCESSING (WITH SPLIT)")
print(f"{'='*70}")
print(f"File: {EXCEL_FILE}")

# ==========================================
# 2. DATA LOADER UTILITIES
# ==========================================

def clean_header_name(val):
    """Convert Excel float headers (1.0 -> 1)"""
    s = str(val).strip()
    if s.lower() == 'nan' or s == '': 
        return None
    try:
        f = float(s)
        if f.is_integer(): 
            return str(int(f))
    except: 
        pass
    return s

def load_sheet_robust(file_path, sheet_name):
    """Load Excel sheet with 2-row header structure"""
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
        if h1 is not None: 
            new_columns.append(h1)
        elif h0 is not None: 
            new_columns.append(h0)
        else: 
            new_columns.append("UNKNOWN")

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=2)
    df.columns = new_columns
    df.columns = df.columns.astype(str)
    return df

def get_col(df, keyword):
    """Find column by keyword (case-insensitive)"""
    matches = [c for c in df.columns if keyword.lower() in c.lower()]
    return matches[0] if matches else None

# ==========================================
# 3. MAIN DATA ENGINE (LOADER)
# ==========================================

def load_grape_data():
    """Load, merge, and prepare RAW GRAPE dataset"""
    print("\n--- STEP 1: LOADING & MERGING DATA ---")
    file_path = os.path.join(RAW_DATA_PATH, EXCEL_FILE)
    if not os.path.exists(file_path): 
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load Sheets
    base_df = load_sheet_robust(file_path, 'Baseline')
    follow_df = load_sheet_robust(file_path, 'Follow-up')
    print(f"✓ Loaded Baseline: {base_df.shape[0]} patients")
    print(f"✓ Loaded Follow-up: {follow_df.shape[0]} records")

    # Create Unique ID (Patient_Eye)
    subj_base = get_col(base_df, 'Subject')
    lat_base = get_col(base_df, 'Laterality')
    base_df['unique_id'] = base_df[subj_base].astype(str) + '_' + base_df[lat_base].astype(str)

    subj_fol = get_col(follow_df, 'Subject')
    lat_fol = get_col(follow_df, 'Laterality')
    follow_df['unique_id'] = follow_df[subj_fol].astype(str) + '_' + follow_df[lat_fol].astype(str)

    # Identify Feature Columns
    vf_cols_base = [str(i) for i in range(61) if str(i) in base_df.columns]
    rnfl_cols = [c for c in ['Mean', 'S', 'N', 'I', 'T'] if c in base_df.columns]
    
    # Find Static Features
    age_col = get_col(base_df, 'Age')
    gender_col = get_col(base_df, 'Gender')
    cct_col = get_col(base_df, 'CCT')
    cat_col = get_col(base_df, 'Category')
    dx_col = get_col(base_df, 'Diagnosis')

    extra_static_map = {
        age_col: 'Age', 
        gender_col: 'Gender', 
        cct_col: 'CCT', 
        cat_col: 'Category_of_Glaucoma',
        dx_col: 'Diagnosis'
    }
    extra_static_map = {k: v for k, v in extra_static_map.items() if k}
    
    print(f"  VF Columns Found: {len(vf_cols_base)}")
    print(f"  RNFL Columns: {rnfl_cols}")
    print(f"  Static Features: {list(extra_static_map.values())}")

    # Progression Logic
    potential_labels = ['MD', 'PLR2', 'PLR3']
    found_labels = [c for c in potential_labels if c in base_df.columns]
    if found_labels:
        for col in found_labels:
            base_df[col] = pd.to_numeric(base_df[col], errors='coerce').fillna(0)
        base_df['Progression_Flag'] = base_df[found_labels].max(axis=1).astype(int)
        print(f"  Progression Indicators: {found_labels}")
    else:
        raise ValueError("CRITICAL: No Progression labels (MD, PLR2, PLR3) found!")

    # Prepare Static Data
    base_df['Visit Number'] = 0.0
    base_df['Interval Years'] = 0.0
    base_df.rename(columns=extra_static_map, inplace=True)
    
    static_cols = ['unique_id', 'Progression_Flag', 'Age', 'Gender', 'CCT', 
                   'Category_of_Glaucoma', 'Diagnosis'] + rnfl_cols
    static_cols = [c for c in static_cols if c in base_df.columns]
    
    patient_static = base_df[static_cols].copy()

    # Normalize Dynamic Columns
    col_map = {'Interval': 'Interval Years', 'IOP': 'IOP', 'CFP': 'Corresponding CFP'}
    def normalize_cols(df):
        rename_dict = {}
        for key, target in col_map.items():
            found = get_col(df, key)
            if found: 
                rename_dict[found] = target
        df.rename(columns=rename_dict, inplace=True)
        return df

    base_df = normalize_cols(base_df)
    follow_df = normalize_cols(follow_df)

    # Stack & Merge
    common_cols = ['unique_id', 'Visit Number', 'Interval Years', 'IOP', 'Corresponding CFP']
    vf_cols_follow = [c for c in follow_df.columns if c in vf_cols_base]
    
    stack_base = base_df[[c for c in (common_cols + vf_cols_base) if c in base_df.columns]]
    stack_fol = follow_df[[c for c in (common_cols + vf_cols_follow) if c in follow_df.columns]]

    full_seq = pd.concat([stack_base, stack_fol], ignore_index=True)
    final_df = pd.merge(full_seq, patient_static, on='unique_id', how='left')
    final_df = final_df.sort_values(by=['unique_id', 'Interval Years']).reset_index(drop=True)

    return final_df, vf_cols_base, rnfl_cols

# ==========================================
# 4. PREPROCESSING ENGINE (CLASS-BASED TO PREVENT LEAKAGE)
# ==========================================

class ClinicalPreprocessor:
    def __init__(self, vf_cols, rnfl_cols):
        self.vf_cols = vf_cols
        self.rnfl_cols = rnfl_cols
        
        # Placeholders for statistics learned from Training Set
        self.numeric_means = {}
        self.scalers = {}  # {col_name: StandardScaler}
        self.encoders = {} # {col_name: LabelEncoder}
        self.numeric_feats = []
        
        # Track report features
        self.track_feats = ['IOP', 'Age', 'CCT', 'Interval Years']
        if 'Mean' in rnfl_cols: self.track_feats.append('Mean')

    def print_stats(self, df, label):
        """Print statistics for reporting"""
        print(f"\n[REPORT] {label}")
        print(f"{'Feature':<20} | {'Missing':<12} | {'Mean':<12} | {'Std':<12} | {'Min':<10} | {'Max':<10}")
        print("-" * 90)
        
        for feat in self.track_feats:
            if feat not in df.columns: continue
            
            missing = df[feat].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            
            if pd.api.types.is_numeric_dtype(df[feat]):
                mean_val = f"{df[feat].mean():.2f}"
                std_val = f"{df[feat].std():.2f}"
                min_val = f"{df[feat].min():.2f}"
                max_val = f"{df[feat].max():.2f}"
            else:
                mean_val = "N/A"
                std_val = "N/A"
                min_val = "N/A"
                max_val = "N/A"
                
            print(f"{feat:<20} | {missing:>3} ({missing_pct:>5.1f}%) | {mean_val:<12} | {std_val:<12} | {min_val:<10} | {max_val:<10}")

    def _prepare_raw(self, df):
        """Steps common to both Train and Test (Safe operations)"""
        df = df.copy()
        
        # 1. Save Raw Time
        df['Interval_Years_Raw'] = pd.to_numeric(df['Interval Years'], errors='coerce').fillna(0.0)
        
        # 2. Define Numeric Features
        self.numeric_feats = ['IOP', 'Age', 'CCT'] + self.vf_cols + self.rnfl_cols
        self.numeric_feats = [c for c in self.numeric_feats if c in df.columns]
        
        # 3. Force Numeric
        for col in self.numeric_feats:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df

    def _interpolate_per_patient(self, df):
        """Safe to do per split or globally, as it is within-patient"""
        interp_cols = ['IOP'] + [c for c in self.vf_cols if c in df.columns]
        interp_cols = [c for c in interp_cols if c in df.columns]
        
        df[interp_cols] = df.groupby('unique_id')[interp_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        return df

    def fit_transform(self, df):
        """LEARN stats from TRAIN set and apply them"""
        print("\n--- FITTING PREPROCESSOR ON TRAINING DATA ---")
        self.print_stats(df, "TRAIN SET - BEFORE PREPROCESSING")
        
        df = self._prepare_raw(df)
        
        # 1. Encode Categoricals (Learn Mappings)
        if 'Gender' in df.columns:
            # Simple map doesn't need fitting, but good to be explicit
            gender_map = {'M': 0, 'F': 1, 'MALE': 0, 'FEMALE': 1}
            df['Gender'] = df['Gender'].astype(str).str.upper().map(gender_map).fillna(0).astype(int)
            
        if 'Category_of_Glaucoma' in df.columns:
            le = LabelEncoder()
            # Convert to string, fit
            df['Category_of_Glaucoma'] = df['Category_of_Glaucoma'].astype(str).fillna('Unknown')
            df['Category_of_Glaucoma'] = le.fit_transform(df['Category_of_Glaucoma'])
            self.encoders['Category'] = le
            print(f"  ✓ Learned Categories: {list(le.classes_)}")

        if 'Diagnosis' in df.columns:
            le_diag = LabelEncoder()
            df['Diagnosis'] = df['Diagnosis'].astype(str).fillna('Unknown')
            df['Diagnosis'] = le_diag.fit_transform(df['Diagnosis'])
            self.encoders['Diagnosis'] = le_diag

        # 2. Interpolate
        df = self._interpolate_per_patient(df)
        
        # 3. Learn Global Means for Filling Gaps
        for col in self.numeric_feats:
            mean_val = df[col].mean()
            self.numeric_means[col] = mean_val
            df[col] = df[col].fillna(mean_val)
        print(f"  ✓ Learned global means from Training set")

        # 4. Learn Scaling (Z-Score)
        # Norm feats excludes flags/IDs
        norm_feats = [c for c in self.numeric_feats if c not in ['Progression_Flag']]
        self.scaler = StandardScaler()
        df[norm_feats] = self.scaler.fit_transform(df[norm_feats])
        
        # Learn Time Scaling
        self.time_scaler = StandardScaler()
        df['Interval_Norm'] = self.time_scaler.fit_transform(df[['Interval_Years_Raw']])
        
        print(f"  ✓ Learned scaling parameters from Training set")
        self.print_stats(df, "TRAIN SET - AFTER PREPROCESSING")
        return df

    def transform(self, df, dataset_name="TEST"):
        """APPLY saved stats to VAL/TEST set (No Peeking!)"""
        print(f"\n--- TRANSFORMING {dataset_name} DATA (Using Train Stats) ---")
        self.print_stats(df, f"{dataset_name} - BEFORE")
        
        df = self._prepare_raw(df)
        
        # 1. Apply Categoricals
        if 'Gender' in df.columns:
            gender_map = {'M': 0, 'F': 1, 'MALE': 0, 'FEMALE': 1}
            df['Gender'] = df['Gender'].astype(str).str.upper().map(gender_map).fillna(0).astype(int)
            
        # Helper to safely encode unseen labels
        def safe_transform(encoder, series):
            # Map unknown classes to 'Unknown' or mode, simple hack: use transform, if error fill 0
            # Better: convert series to string, if not in classes_, replace with 'Unknown'
            # Assuming 'Unknown' was in train. If not, map to index 0.
            known_classes = set(encoder.classes_)
            series = series.apply(lambda x: x if x in known_classes else encoder.classes_[0])
            return encoder.transform(series)

        if 'Category_of_Glaucoma' in df.columns and 'Category' in self.encoders:
            df['Category_of_Glaucoma'] = df['Category_of_Glaucoma'].astype(str).fillna('Unknown')
            df['Category_of_Glaucoma'] = safe_transform(self.encoders['Category'], df['Category_of_Glaucoma'])

        if 'Diagnosis' in df.columns and 'Diagnosis' in self.encoders:
            df['Diagnosis'] = df['Diagnosis'].astype(str).fillna('Unknown')
            df['Diagnosis'] = safe_transform(self.encoders['Diagnosis'], df['Diagnosis'])

        # 2. Interpolate
        df = self._interpolate_per_patient(df)
        
        # 3. Fill Missing using TRAIN Means (Prevent Leakage)
        for col in self.numeric_feats:
            if col in self.numeric_means:
                df[col] = df[col].fillna(self.numeric_means[col])
        
        # 4. Apply Scaling using TRAIN Scaler
        norm_feats = [c for c in self.numeric_feats if c not in ['Progression_Flag']]
        df[norm_feats] = self.scaler.transform(df[norm_feats])
        
        # Apply Time Scaling
        df['Interval_Norm'] = self.time_scaler.transform(df[['Interval_Years_Raw']])
        
        self.print_stats(df, f"{dataset_name} - AFTER")
        return df

# ==========================================
# 5. DATASET SPLITTER
# ==========================================

def split_patients(df):
    """Split dataset by Unique ID to prevent leakage"""
    print("\n" + "="*70)
    print("STEP 2: SPLITTING DATASET (Patient-Level)")
    print("="*70)
    
    unique_ids = df['unique_id'].unique()
    print(f"Total Unique Patients: {len(unique_ids)}")
    
    # Split 70% Train, 30% Temp
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42, shuffle=True)
    # Split Temp into 15% Val, 15% Test
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42, shuffle=True)
    
    print(f"  Train Patients: {len(train_ids)} ({len(train_ids)/len(unique_ids):.1%})")
    print(f"  Val Patients:   {len(val_ids)} ({len(val_ids)/len(unique_ids):.1%})")
    print(f"  Test Patients:  {len(test_ids)} ({len(test_ids)/len(unique_ids):.1%})")
    
    train_df = df[df['unique_id'].isin(train_ids)].copy()
    val_df = df[df['unique_id'].isin(val_ids)].copy()
    test_df = df[df['unique_id'].isin(test_ids)].copy()
    
    return train_df, val_df, test_df

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

def main():
    try:
        # 1. LOAD RAW DATA
        df_raw, vf_cols, rnfl_cols = load_grape_data()
        
        # 2. SPLIT RAW DATA
        train_raw, val_raw, test_raw = split_patients(df_raw)
        
        # 3. PREPROCESS (FIT on Train, TRANSFORM Val/Test)
        print("\n" + "="*70)
        print("STEP 3: PREPROCESSING PIPELINE")
        print("="*70)
        
        processor = ClinicalPreprocessor(vf_cols, rnfl_cols)
        
        # Fit & Process Train
        train_proc = processor.fit_transform(train_raw)
        
        # Process Val & Test (Using Train stats)
        val_proc = processor.transform(val_raw, "VALIDATION")
        test_proc = processor.transform(test_raw, "TEST")
        
        # 4. SAVE
        print("\n" + "="*70)
        print("STEP 4: SAVING DATASETS")
        print("="*70)
        
        if not os.path.exists(PROCESSED_DIR): 
            os.makedirs(PROCESSED_DIR)
            
        train_proc.to_csv(os.path.join(PROCESSED_DIR, 'grape_train.csv'), index=False)
        val_proc.to_csv(os.path.join(PROCESSED_DIR, 'grape_val.csv'), index=False)
        test_proc.to_csv(os.path.join(PROCESSED_DIR, 'grape_test.csv'), index=False)
        
        # Save Full for reference (Optional)
        full_proc = pd.concat([train_proc, val_proc, test_proc], axis=0)
        full_proc.to_csv(os.path.join(PROCESSED_DIR, 'grape_clinical_full_processed3.csv'), index=False)

        print(f"\nSUCCESS! Files saved to {PROCESSED_DIR}:")
        print("  - grape_train.csv")
        print("  - grape_val.csv")
        print("  - grape_test.csv")
        print("  - grape_clinical_full_processed3.csv")
        
        # Metadata
        with open(os.path.join(PROCESSED_DIR, 'dataset_stats.txt'), 'w') as f:
            f.write(f"Train Samples: {len(train_proc)}\n")
            f.write(f"Val Samples: {len(val_proc)}\n")
            f.write(f"Test Samples: {len(test_proc)}\n")
            f.write(f"Features: {len(train_proc.columns)}\n")

    except Exception as e:
        print(f"\n [CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

