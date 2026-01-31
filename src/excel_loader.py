"""
excel_loader.py - Load and Integrate Excel Clinical Data

This module loads additional clinical data from Excel and merges it with
MST predictions for comprehensive patient profiles.
"""

import pandas as pd
import os
from typing import Dict, List, Optional
from config import EXCEL_DATA_PATH, logger


class ExcelDataLoader:
    """Load and merge Excel clinical data with patient JSON data"""
    
    def __init__(self, excel_path: str = EXCEL_DATA_PATH):
        """
        Initialize Excel data loader
        
        Args:
            excel_path: Path to Excel file with clinical data
        """
        self.excel_path = excel_path
        self.excel_data = None
        
        if os.path.exists(excel_path):
            self.load_excel()
        else:
            logger.warning(f"Excel file not found: {excel_path}")
            logger.info("Pipeline will continue without additional clinical data")
    
    def load_excel(self) -> bool:
        """
        Load Excel file into pandas DataFrame
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading Excel data from: {self.excel_path}")
            self.excel_data = pd.read_excel(self.excel_path)
            
            # Validate required columns
            required_columns = ['patient_id', 'visit']
            missing_columns = [col for col in required_columns if col not in self.excel_data.columns]
            
            if missing_columns:
                logger.error(f"Excel missing required columns: {missing_columns}")
                logger.error("Excel must have at least: patient_id, visit")
                self.excel_data = None
                return False
            
            logger.info(f"[OK] Loaded Excel data: {len(self.excel_data)} rows")
            logger.info(f"  Columns: {list(self.excel_data.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            self.excel_data = None
            return False
    
    def get_patient_clinical_data(self, patient_id: str, visit: int) -> Optional[Dict]:
        """
        Get clinical data for a specific patient visit
        
        Args:
            patient_id: Patient identifier
            visit: Visit number
            
        Returns:
            Dictionary of clinical data or None if not found
        """
        if self.excel_data is None:
            return None
        
        # Find matching row
        mask = (self.excel_data['patient_id'] == patient_id) & (self.excel_data['visit'] == visit)
        matches = self.excel_data[mask]
        
        if len(matches) == 0:
            return None
        
        if len(matches) > 1:
            logger.warning(f"Multiple Excel rows found for {patient_id}, visit {visit}. Using first.")
        
        # Convert row to dictionary, excluding patient_id and visit
        row = matches.iloc[0]
        clinical_data = {}
        
        for col in self.excel_data.columns:
            if col not in ['patient_id', 'visit']:
                value = row[col]
                # Handle NaN values
                if pd.notna(value):
                    clinical_data[col] = value
        
        return clinical_data
    
    def enrich_patient_data(self, patient_data: Dict) -> Dict:
        """
        Enrich patient JSON data with Excel clinical data
        
        Args:
            patient_data: Patient data dictionary from JSON
            
        Returns:
            Enhanced patient data with clinical info
        """
        if self.excel_data is None:
            logger.info("No Excel data available - continuing without additional clinical data")
            return patient_data
        
        patient_id = patient_data['patient_id']
        logger.info(f"Enriching patient {patient_id} with Excel data...")
        
        enriched_count = 0
        
        # Handle both 'visits' (list) and 'visit' (single) formats
        visits = patient_data.get('visits', [])
        if not visits and 'visit' in patient_data:
        # If there's no 'visits' list but there is a 'visit' field, treat the whole patient_data as a single visit
            visits = [patient_data]
    
        for visit in visits:
            visit_num = visit.get('visit', 0)
            clinical_data = self.get_patient_clinical_data(patient_id, visit_num)
            
            if clinical_data:
                visit['additional_clinical_data'] = clinical_data
                enriched_count += 1
                
                # Log what was added
                fields = ', '.join(str(key) for key in clinical_data.keys())  # Convert all keys to strings
                logger.info(f"  Visit {visit_num}: Added {len(clinical_data)} fields ({fields})")
        
        if enriched_count > 0:
            logger.info(f"[OK] Enriched {enriched_count}/{len(visits)} visits with Excel data")
        else:
            logger.warning(f"  No Excel data found for patient {patient_id}")
        
        return patient_data
    
    def get_available_fields(self) -> List[str]:
        """
        Get list of available clinical fields in Excel
        
        Returns:
            List of column names (excluding patient_id and visit)
        """
        if self.excel_data is None:
            return []
        
        return [col for col in self.excel_data.columns if col not in ['patient_id', 'visit']]
    
    def has_field(self, field_name: str) -> bool:
        """
        Check if a specific field exists in Excel data
        
        Args:
            field_name: Name of the field to check
            
        Returns:
            True if field exists
        """
        if self.excel_data is None:
            return False
        return field_name in self.excel_data.columns


def format_clinical_data_for_prompt(visit_data: Dict) -> str:
    """
    Format additional clinical data for inclusion in LLM prompt
    
    Args:
        visit_data: Visit dictionary with additional_clinical_data
        
    Returns:
        Formatted string for prompt
    """
    clinical_data = visit_data.get('additional_clinical_data', {})
    
    if not clinical_data:
        return "  Additional Clinical Data: Not available\n"
    
    text = "  Additional Clinical Data:\n"
    
    # Common field mappings with better formatting
    field_names = {
        'age': 'Age',
        'gender': 'Gender',
        'IOP': 'Intraocular Pressure (IOP)',
        'iop': 'Intraocular Pressure (IOP)',
        'RNFL_superior': 'RNFL Thickness - Superior',
        'RNFL_inferior': 'RNFL Thickness - Inferior',
        'RNFL_nasal': 'RNFL Thickness - Nasal',
        'RNFL_temporal': 'RNFL Thickness - Temporal',
        'RNFL_average': 'RNFL Thickness - Average',
        'CCT': 'Central Corneal Thickness (CCT)',
        'cct': 'Central Corneal Thickness (CCT)',
        'cup_disc_ratio': 'Cup-to-Disc Ratio',
        'CDR': 'Cup-to-Disc Ratio',
        'medications': 'Current Medications',
        'baseline_IOP': 'Baseline IOP',
        'family_history': 'Family History of Glaucoma',
        'diabetes': 'Diabetes',
        'hypertension': 'Hypertension'
    }
    
    # Format each field
    for key, value in clinical_data.items():
        display_name = str(key)
        
        # Format value based on type
        if isinstance(value, float):
            key_str = str(key).upper()
            if 'IOP' in key_str:
                formatted_value = f"{value:.1f} mmHg"
            elif 'RNFL' in key_str:
                formatted_value = f"{value:.1f} μm"
            elif 'CCT' in key_str:
                formatted_value = f"{value:.0f} μm"
            elif 'ratio' in str(key).lower() or 'CDR' in key_str:
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)
        
        text += f"    • {display_name}: {formatted_value}\n"
    
    return text


def main():
    """Test Excel loader"""
    print("="*60)
    print("EXCEL DATA LOADER TEST")
    print("="*60)
    
    loader = ExcelDataLoader()
    
    if loader.excel_data is not None:
        print(f"\n[OK] Excel loaded successfully")
        print(f"  Rows: {len(loader.excel_data)}")
        print(f"  Columns: {list(loader.excel_data.columns)}")
        
        # Show available fields
        fields = loader.get_available_fields()
        print(f"\n  Available clinical fields: {', '.join(fields)}")
        
        # Test getting data for a patient
        print(f"\n" + "="*60)
        print("TEST: Getting data for patient 101_OD, visit 0")
        print("="*60)
        
        data = loader.get_patient_clinical_data("101_OD", 0)
        if data:
            print("Clinical data found:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print("No data found for this patient/visit")
        
        # Test enrichment
        print(f"\n" + "="*60)
        print("TEST: Enriching patient data")
        print("="*60)
        
        test_patient = {
            "patient_id": "101_OD",
            "visits": [
                {"visit": 0, "mae_dB": 7.5},
                {"visit": 1, "mae_dB": 6.8}
            ]
        }
        
        enriched = loader.enrich_patient_data(test_patient)
        print(f"Enriched visits: {len([v for v in enriched['visits'] if 'additional_clinical_data' in v])}")
        
    else:
        print(f"\n[ERROR] Excel file not found or could not be loaded")
        print(f"Expected location: {EXCEL_DATA_PATH}")
        print("\nTo use Excel data:")
        print("1. Create Excel file with columns: patient_id, visit, IOP, RNFL_superior, etc.")
        print("2. Save to: data/excel/patient_clinical_data.xlsx")
    
    print("\n[OK] Excel loader test complete!")


if __name__ == "__main__":
    main()