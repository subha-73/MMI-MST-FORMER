"""
clinical_metrics.py - Calculate Clinical Metrics from Visual Field Data
"""

import numpy as np
from typing import Dict, List
from config import VF_POINTS, NORMAL_VF_THRESHOLD, logger


class ClinicalMetricsCalculator:
    """Calculate standard clinical metrics from visual field data"""
    
    def __init__(self, normal_threshold: float = NORMAL_VF_THRESHOLD):
        self.normal_threshold = normal_threshold
    
    def calculate_mean_deviation(self, vf_values: List[float]) -> float:
        """
        Calculate Mean Deviation (MD)
        
        MD = average sensitivity - age-matched normal
        Negative values indicate visual field loss
        
        Args:
            vf_values: List of 61 VF sensitivity values (dB)
            
        Returns:
            MD in dB
        """
        vf_array = np.array(vf_values)
        mean_sensitivity = np.mean(vf_array)
        MD = mean_sensitivity - self.normal_threshold
        return round(float(MD), 2)
    
    def calculate_visual_field_index(self, vf_values: List[float]) -> float:
        """
        Calculate Visual Field Index (VFI)
        
        VFI = percentage of normal visual field remaining
        Range: 0-100%
        
        Args:
            vf_values: List of 61 VF sensitivity values (dB)
            
        Returns:
            VFI as percentage
        """
        vf_array = np.array(vf_values)
        mean_sensitivity = np.mean(vf_array)
        VFI = (mean_sensitivity / self.normal_threshold) * 100
        VFI = max(0, min(100, VFI))  # Clamp to 0-100
        return round(float(VFI), 1)
    
    def calculate_pattern_standard_deviation(self, vf_values: List[float]) -> float:
        """
        Calculate Pattern Standard Deviation (PSD)
        
        PSD measures irregularity/variability of visual field defects
        Higher values indicate more localized/irregular defects
        
        Args:
            vf_values: List of 61 VF sensitivity values (dB)
            
        Returns:
            PSD in dB
        """
        vf_array = np.array(vf_values)
        PSD = np.std(vf_array)
        return round(float(PSD), 2)
    
    def analyze_quadrants(self, vf_values: List[float]) -> Dict:
        """
        Analyze visual field by quadrants
        
        Standard 24-2 or 30-2 pattern:
        - Points 0-30: Superior field
        - Points 31-60: Inferior field
        
        Args:
            vf_values: List of 61 VF sensitivity values (dB)
            
        Returns:
            Dictionary with quadrant analysis
        """
        vf_array = np.array(vf_values)
        
        # Split superior/inferior
        superior = vf_array[0:31]
        inferior = vf_array[31:61]
        
        superior_mean = np.mean(superior)
        inferior_mean = np.mean(inferior)
        
        # Determine defect pattern
        threshold_diff = 5.0  # dB difference to classify as defect
        
        if superior_mean < inferior_mean - threshold_diff:
            pattern = "superior defect"
        elif inferior_mean < superior_mean - threshold_diff:
            pattern = "inferior defect"
        else:
            pattern = "diffuse defect"
        
        return {
            "superior_mean": round(float(superior_mean), 2),
            "inferior_mean": round(float(inferior_mean), 2),
            "pattern": pattern,
            "superior_inferior_diff": round(float(abs(superior_mean - inferior_mean)), 2)
        }
    
    def calculate_all_metrics(self, vf_values: List[float]) -> Dict:
        """
        Calculate all clinical metrics at once
        
        Args:
            vf_values: List of 61 VF sensitivity values (dB)
            
        Returns:
            Dictionary with all metrics
        """
        if len(vf_values) != VF_POINTS:
            logger.warning(f"Expected {VF_POINTS} VF points, got {len(vf_values)}")
        
        metrics = {
            "MD": self.calculate_mean_deviation(vf_values),
            "VFI": self.calculate_visual_field_index(vf_values),
            "PSD": self.calculate_pattern_standard_deviation(vf_values),
            "mean_vf": round(float(np.mean(vf_values)), 2),
            "min_vf": round(float(np.min(vf_values)), 2),
            "max_vf": round(float(np.max(vf_values)), 2)
        }
        
        # Add quadrant analysis
        quadrant_data = self.analyze_quadrants(vf_values)
        metrics.update(quadrant_data)
        
        return metrics
    
    def classify_severity(self, MD: float) -> str:
        """
        Classify glaucoma severity based on Mean Deviation
        
        Based on AAO guidelines:
        - Normal/Mild: MD > -6 dB
        - Moderate: MD -6 to -12 dB
        - Severe: MD -12 to -20 dB
        - Advanced: MD < -20 dB
        
        Args:
            MD: Mean Deviation in dB
            
        Returns:
            Severity classification string
        """
        if MD >= -6:
            return "Mild"
        elif MD >= -12:
            return "Moderate"
        elif MD >= -20:
            return "Severe"
        else:
            return "Advanced"


def process_visit_metrics(visit_data: Dict) -> Dict:
    """
    Process a single visit and calculate all metrics
    
    Args:
        visit_data: Dictionary with visit information including true_vf_dB
        
    Returns:
        Visit data with added clinical_metrics
    """
    calculator = ClinicalMetricsCalculator()
    
    # Calculate metrics from true VF values
    true_vf = visit_data["true_vf_dB"]
    metrics = calculator.calculate_all_metrics(true_vf)
    
    # Add severity classification
    metrics["severity"] = calculator.classify_severity(metrics["MD"])
    
    # Add to visit data
    visit_data["clinical_metrics"] = metrics
    
    logger.info(f"Visit {visit_data['visit']}: MD={metrics['MD']} dB, "
                f"VFI={metrics['VFI']}%, Severity={metrics['severity']}")
    
    return visit_data


def process_all_visits(patient_data: Dict) -> Dict:
    """
    Process all visits for a patient
    
    Args:
        patient_data: Dictionary with patient_id and visits list
        
    Returns:
        Patient data with clinical metrics added to each visit
    """
    logger.info(f"Processing metrics for patient: {patient_data['patient_id']}")
    
    for visit in patient_data["visits"]:
        process_visit_metrics(visit)
    
    return patient_data


def main():
    """Test clinical metrics calculation"""
    # Test data (example from your MST output)
    test_vf = [
        21.17, 30.91, 8.38, 12.80, 35.00, 26.13, 4.50, 20.32, 26.39, 23.05,
        12.84, 18.89, 13.07, 14.12, 10.07, 14.47, 13.09, 14.76, 26.51, 16.14,
        24.89, 25.00, 18.03, 18.02, 24.82, 20.77, 12.10, 9.00, 8.54, 15.47,
        10.82, 10.81, 25.00, 30.72, 21.54, 23.86, 22.56, 27.73, 29.51, 24.21,
        26.68, 19.67, 10.85, 10.03, 9.68, 10.39, 10.37, 14.90, 16.30, 21.18,
        28.14, 23.50, 29.15, 26.57, 27.86, 28.00, 10.36, 10.36, 17.66, 23.37,
        31.29
    ]
    
    calculator = ClinicalMetricsCalculator()
    metrics = calculator.calculate_all_metrics(test_vf)
    
    print("\n" + "="*60)
    print("CLINICAL METRICS CALCULATION TEST")
    print("="*60)
    print(f"\nMean Deviation (MD): {metrics['MD']} dB")
    print(f"Visual Field Index (VFI): {metrics['VFI']}%")
    print(f"Pattern Std Deviation (PSD): {metrics['PSD']} dB")
    print(f"Mean VF Sensitivity: {metrics['mean_vf']} dB")
    print(f"\nQuadrant Analysis:")
    print(f"  Superior: {metrics['superior_mean']} dB")
    print(f"  Inferior: {metrics['inferior_mean']} dB")
    print(f"  Pattern: {metrics['pattern']}")
    print(f"\nSeverity: {calculator.classify_severity(metrics['MD'])}")
    print("\n[OK] Metrics calculation test complete!")


if __name__ == "__main__":
    main()