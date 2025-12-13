import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

# ==========================================
# 1. CONFIGURATION
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RAW_DATA_PATH = os.path.join(ROOT_DIR, '1_data', 'raw', 'grape')
PROCESSED_DIR = os.path.join(ROOT_DIR, '1_data', 'processed')
EXCEL_FILE = 'VF_clinical_information.xlsx'

print(f"{'='*70}")
print(f"GRAPE GLAUCOMA FORECASTING - COMPLETE CLINICAL PREPROCESSING")
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
# 3. MAIN DATA ENGINE
# ==========================================

def load_grape_data():
    """Load, merge, and prepare GRAPE dataset"""
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
    
    # Clean None keys
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
    
    # Rename extra columns
    base_df.rename(columns=extra_static_map, inplace=True)
    
    # Select static columns (drop NaN columns)
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
# 4. QUALITY CONTROL & VALIDATION
# ==========================================

def detect_and_report_outliers(df, vf_cols, rnfl_cols):
    """Detect outliers using IQR method (only on numeric columns)"""
    print("\n--- STEP 2A: OUTLIER DETECTION ---")
    
    outlier_features = vf_cols + rnfl_cols + ['IOP']
    outlier_features = [c for c in outlier_features if c in df.columns]
    
    outlier_counts = {}
    for col in outlier_features:
        # Convert to numeric first (handles string values)
        col_numeric = pd.to_numeric(df[col], errors='coerce')
        
        # Skip if all NaN
        if col_numeric.isna().all():
            continue
        
        Q1 = col_numeric.quantile(0.25)
        Q3 = col_numeric.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = col_numeric[(col_numeric < lower_bound) | (col_numeric > upper_bound)].shape[0]
        outlier_counts[col] = outliers
    
    total_outliers = sum(outlier_counts.values())
    high_outlier_cols = [k for k, v in outlier_counts.items() if v > 0]
    
    print(f"  Total Outlier Instances: {total_outliers}")
    if high_outlier_cols:
        print(f"  Features with Outliers: {high_outlier_cols[:10]}")
    
    return outlier_counts

def missing_value_analysis(df):
    """Detailed missing value report"""
    print("\n--- STEP 2B: MISSING VALUE ANALYSIS ---")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    # Overall
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_rate = (total_missing / total_cells) * 100
    print(f"  Overall Missing Rate: {missing_rate:.2f}%")
    
    # By column
    high_missing = missing[missing > 0].sort_values(ascending=False)
    if len(high_missing) > 0:
        print(f"\n  Columns with Missing Data (Top 10):")
        for col, count in high_missing.head(10).items():
            print(f"    {col}: {count} ({(count/len(df))*100:.1f}%)")
    else:
        print(f"\n  No missing data detected!")
    
    # By patient
    patient_missing = df.groupby('unique_id').apply(lambda x: x.isnull().sum().sum())
    print(f"\n  Missing Data per Patient:")
    print(f"    Mean: {patient_missing.mean():.1f} values")
    print(f"    Max: {patient_missing.max()} values")
    print(f"    Min: {patient_missing.min()} values")
    
    return missing

def validate_clinical_ranges(df, vf_cols):
    """Check if values fall within clinical valid ranges"""
    print("\n--- STEP 2C: CLINICAL RANGE VALIDATION ---")
    
    validations = {}
    
    # IOP: 0-60 mmHg
    if 'IOP' in df.columns:
        iop_numeric = pd.to_numeric(df['IOP'], errors='coerce')
        valid_iop = iop_numeric[(iop_numeric > 0) & (iop_numeric < 60)].shape[0]
        total_iop = iop_numeric.notna().sum()
        if total_iop > 0:
            validations['IOP'] = (valid_iop, total_iop)
            print(f"  IOP (0-60 mmHg): {valid_iop}/{total_iop} valid ({100*valid_iop/total_iop:.1f}%)")
    
    # VF: -35 to 0 dB (typical range)
    if vf_cols:
        vf_valid_total = 0
        vf_count_total = 0
        for col in vf_cols[:5]:  # Show first 5
            col_numeric = pd.to_numeric(df[col], errors='coerce')
            valid_vf = col_numeric[(col_numeric >= -35) & (col_numeric <= 0)].shape[0]
            total_vf = col_numeric.notna().sum()
            if total_vf > 0:
                validations[col] = (valid_vf, total_vf)
                vf_valid_total += valid_vf
                vf_count_total += total_vf
        
        if vf_count_total > 0:
            print(f"  VF Points (-35 to 0 dB): {vf_valid_total}/{vf_count_total} valid ({100*vf_valid_total/vf_count_total:.1f}%)")
    
    # Age: 18-100
    if 'Age' in df.columns:
        age_numeric = pd.to_numeric(df['Age'], errors='coerce')
        valid_age = age_numeric[(age_numeric > 18) & (age_numeric < 100)].shape[0]
        total_age = age_numeric.notna().sum()
        if total_age > 0:
            validations['Age'] = (valid_age, total_age)
            print(f"  Age (18-100): {valid_age}/{total_age} valid ({100*valid_age/total_age:.1f}%)")

# ==========================================
# 5. REPORTING & CLEANING ENGINE
# ==========================================

def print_stats(df, label, features):
    """Print statistics before/after cleaning"""
    print(f"\n[REPORT] {label}")
    print(f"{'Feature':<20} | {'Missing':<12} | {'Mean':<12} | {'Std':<12} | {'Min':<10} | {'Max':<10}")
    print("-" * 90)
    
    for feat in features:
        if feat not in df.columns: 
            continue
        
        missing = df[feat].isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        if pd.api.types.is_numeric_dtype(df[feat]):
            mean_val = f"{df[feat].mean():.2f}"
            std_val = f"{df[feat].std():.2f}"
            min_val = f"{df[feat].min():.2f}"
            max_val = f"{df[feat].max():.2f}"
        else:
            mean_val = str(df[feat].mode()[0] if not df[feat].mode().empty else "N/A")[:10]
            std_val = "N/A"
            min_val = "N/A"
            max_val = "N/A"
            
        print(f"{feat:<20} | {missing:>3} ({missing_pct:>5.1f}%) | {mean_val:<12} | {std_val:<12} | {min_val:<10} | {max_val:<10}")

def clean_and_normalize(df, vf_cols, rnfl_cols):
    """Complete cleaning and normalization pipeline"""
    print("\n" + "="*70)
    print("STEP 2: CLEANING & NORMALIZATION")
    print("="*70)
    
    # Track features for reporting
    track_feats = ['IOP', 'Age', 'CCT', 'Interval Years']
    if 'Mean' in rnfl_cols: 
        track_feats.append('Mean')
    
    # === BEFORE CLEANING REPORT ===
    print_stats(df, "BEFORE PREPROCESSING", track_feats)
    
    # 1. SAVE RAW TIME (Before any modification)
    df['Interval_Years_Raw'] = pd.to_numeric(df['Interval Years'], errors='coerce').fillna(0.0)
    
    # 2. CONVERT TO NUMERIC FIRST (Critical - before any checks!)
    numeric_feats = ['IOP', 'Age', 'CCT'] + vf_cols + rnfl_cols
    numeric_feats = [c for c in numeric_feats if c in df.columns]
    
    for col in numeric_feats:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"  ✓ Converted {len(numeric_feats)} columns to numeric")
    
    # === QUALITY CHECKS (Now on numeric data) ===
    detect_and_report_outliers(df, vf_cols, rnfl_cols)
    missing_value_analysis(df)
    validate_clinical_ranges(df, vf_cols)
    
    print("\n--- STEP 2-D: DATA CLEANING OPERATIONS ---")
    
    # 3. ENCODE CATEGORICAL VARIABLES
    
    # 3. ENCODE CATEGORICAL VARIABLES
    if 'Gender' in df.columns:
        gender_map = {'M': 0, 'F': 1, 'MALE': 0, 'FEMALE': 1}
        df['Gender'] = df['Gender'].astype(str).str.upper().map(gender_map).fillna(0).astype(int)
        print(f"  ✓ Encoded Gender (M=0, F=1)")
    
    if 'Category_of_Glaucoma' in df.columns:
        le = LabelEncoder()
        df['Category_of_Glaucoma'] = df['Category_of_Glaucoma'].astype(str).fillna('Unknown')
        df['Category_of_Glaucoma'] = le.fit_transform(df['Category_of_Glaucoma'])
        print(f"  ✓ Encoded Glaucoma Categories: {list(le.classes_)}")
    
    if 'Diagnosis' in df.columns:
        le_diag = LabelEncoder()
        df['Diagnosis'] = df['Diagnosis'].astype(str).fillna('Unknown')
        df['Diagnosis'] = le_diag.fit_transform(df['Diagnosis'])
        print(f"  ✓ Encoded Diagnosis: {list(le_diag.classes_)}")
    
    # 4. INTERPOLATION (Time-series fill per patient)
    interp_cols = ['IOP'] + [c for c in vf_cols if c in df.columns]
    interp_cols = [c for c in interp_cols if c in df.columns]
    
    df[interp_cols] = df.groupby('unique_id')[interp_cols].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    print(f"  ✓ Applied linear interpolation to {len(interp_cols)} features per patient")
    
    # 5. FILL REMAINING GAPS (Global mean as fallback)
    remaining_missing = df[numeric_feats].isnull().sum().sum()
    df[numeric_feats] = df[numeric_feats].fillna(df[numeric_feats].mean())
    print(f"  ✓ Filled {remaining_missing} remaining missing values with global means")
    
    # 6. Z-SCORE NORMALIZATION (Standardization)
    # Normalize only continuous variables, NOT categorical flags
    norm_feats = [c for c in numeric_feats if c not in ['Progression_Flag']]
    
    scaler = StandardScaler()
    df[norm_feats] = scaler.fit_transform(df[norm_feats])
    print(f"  ✓ Applied Z-score normalization to {len(norm_feats)} continuous features")
    
    # Normalize time separately
    time_scaler = StandardScaler()
    df['Interval_Norm'] = time_scaler.fit_transform(df[['Interval_Years_Raw']])
    print(f"  ✓ Normalized time interval separately")
    print_stats(df, "AFTER PREPROCESSING (Normalized)", track_feats)
    
    return df

# ==========================================
# 6. DATASET STATISTICS & SUMMARY
# ==========================================

def generate_final_report(df, vf_cols, rnfl_cols):
    """Generate comprehensive dataset summary"""
    print("\n" + "="*70)
    print("STEP 3: FINAL DATASET SUMMARY")
    print("="*70)
    
    print(f"\n📊 DATASET DIMENSIONS:")
    print(f"  Total Rows: {df.shape[0]}")
    print(f"  Total Columns: {df.shape[1]}")
    
    unique_patients = df['unique_id'].nunique()
    print(f"\n👥 PATIENT INFORMATION:")
    print(f"  Unique Patients (eyes): {unique_patients}")
    print(f"  Avg Rows per Patient: {df.shape[0] / unique_patients:.1f}")
    
    print(f"\n🔬 FEATURE BREAKDOWN:")
    print(f"  VF Points: {len(vf_cols)}")
    print(f"  RNFL Metrics: {len(rnfl_cols)}")
    print(f"  Static Features: Age, Gender, CCT, Category, Diagnosis")
    print(f"  Dynamic Features: IOP, Interval Years")
    
    if 'Progression_Flag' in df.columns:
        prog_count = df['Progression_Flag'].sum()
        stable_count = (1 - df['Progression_Flag']).sum()
        print(f"\n⚠️  PROGRESSION STATUS (unique patients):")
        prog_patients = df[df['Progression_Flag'] == 1]['unique_id'].nunique()
        stable_patients = df[df['Progression_Flag'] == 0]['unique_id'].nunique()
        print(f"  Progressors: {prog_patients} ({100*prog_patients/unique_patients:.1f}%)")
        print(f"  Stable: {stable_patients} ({100*stable_patients/unique_patients:.1f}%)")
    
    # Data completeness
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    print(f"\n✅ DATA QUALITY:")
    print(f"  Completeness: {completeness:.2f}%")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  Location: {PROCESSED_DIR}")
    print(f"  Filename: grape_clinical_processed2.csv")

# ==========================================
# 7. MAIN EXECUTION
# ==========================================

def main():
    """Main preprocessing pipeline"""
    try:
        # Step 1: Load
        print("\n")
        df, vf_cols, rnfl_cols = load_grape_data()
        print(f"\n✓ Data loaded successfully!")
        
        # Step 2: Clean & Normalize
        print("\n")
        final_df = clean_and_normalize(df, vf_cols, rnfl_cols)
        
        # Step 3: Generate Report
        print("\n")
        generate_final_report(final_df, vf_cols, rnfl_cols)
        
        # Step 4: Save
        print("\n" + "="*70)
        print("STEP 4: SAVING OUTPUT")
        print("="*70)
        
        if not os.path.exists(PROCESSED_DIR): 
            os.makedirs(PROCESSED_DIR)
        
        save_path = os.path.join(PROCESSED_DIR, 'grape_clinical_processed2.csv')
        final_df.to_csv(save_path, index=False)
        
        print(f"\n✅ SUCCESS! File saved to:")
        print(f"   {save_path}")
        
        # Save metadata
        metadata_path = os.path.join(PROCESSED_DIR, 'preprocessing_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write("GRAPE Glaucoma Dataset - Preprocessing Metadata\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Samples: {final_df.shape[0]}\n")
            f.write(f"Total Features: {final_df.shape[1]}\n")
            f.write(f"Unique Patients: {final_df['unique_id'].nunique()}\n")
            f.write(f"VF Points: {len(vf_cols)}\n")
            f.write(f"RNFL Metrics: {len(rnfl_cols)}\n")
            f.write(f"\nColumns: {', '.join(final_df.columns.tolist())}\n")
        
        print(f"   {metadata_path}")
        print(f"\n{'='*70}")
        print("PREPROCESSING COMPLETE! ✓")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n❌ [CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()