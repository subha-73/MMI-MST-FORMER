"""
progression_analysis.py - Multi-Visit Progression Analysis
"""

from typing import Dict, List
import numpy as np
from config import (
    DEFAULT_VISIT_INTERVAL_MONTHS, 
    PROGRESSION_THRESHOLDS, 
    logger
)


class ProgressionAnalyzer:
    """Analyze disease progression across multiple visits"""
    
    def __init__(self, visit_interval_months: float = DEFAULT_VISIT_INTERVAL_MONTHS):
        self.visit_interval_months = visit_interval_months
    
    def analyze_progression(self, visits: List[Dict]) -> Dict:
        """
        Analyze progression across multiple visits
        
        Args:
            visits: List of visit dictionaries with clinical_metrics
            
        Returns:
            Comprehensive progression analysis
        """
        if len(visits) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 visits for progression analysis",
                "num_visits": len(visits)
            }
        
        # Extract metrics from each visit
        visit_metrics = []
        for v in visits:
            if "clinical_metrics" not in v:
                logger.error(f"Visit {v['visit']} missing clinical_metrics")
                continue
            
            visit_metrics.append({
                "visit": v["visit"],
                "MD": v["clinical_metrics"]["MD"],
                "VFI": v["clinical_metrics"]["VFI"],
                "PSD": v["clinical_metrics"]["PSD"],
                "mae": v.get("mae_dB", None),
                "severity": v["clinical_metrics"]["severity"]
            })
        
        if len(visit_metrics) < 2:
            return {
                "status": "insufficient_data",
                "message": "Not enough visits with clinical metrics"
            }
        
        # Calculate overall changes
        first_visit = visit_metrics[0]
        last_visit = visit_metrics[-1]
        
        delta_MD = last_visit["MD"] - first_visit["MD"]
        delta_VFI = last_visit["VFI"] - first_visit["VFI"]
        
        # Calculate progression rate
        num_intervals = last_visit["visit"] - first_visit["visit"]
        time_interval_years = (num_intervals * self.visit_interval_months) / 12.0
        
        if time_interval_years > 0:
            progression_rate_MD = delta_MD / time_interval_years
            progression_rate_VFI = delta_VFI / time_interval_years
        else:
            progression_rate_MD = 0
            progression_rate_VFI = 0
        
        # Classify progression risk
        risk_level = self._classify_progression_risk(abs(progression_rate_MD))
        
        # Determine trend
        if delta_MD > 0.5:
            trend = "Improving"
        elif delta_MD < -0.5:
            trend = "Worsening"
        else:
            trend = "Stable"
        
        # Check for acceleration (if 3+ visits)
        acceleration = self._analyze_acceleration(visit_metrics, time_interval_years)
        
        # Detect concerning patterns
        warnings = self._detect_warning_signs(visit_metrics, progression_rate_MD)
        
        return {
            "status": "analyzed",
            "num_visits": len(visits),
            "time_span_years": round(time_interval_years, 1),
            "time_span_months": round(time_interval_years * 12, 0),
            
            # Overall changes
            "delta_MD": round(delta_MD, 2),
            "delta_VFI": round(delta_VFI, 1),
            "delta_PSD": round(last_visit["PSD"] - first_visit["PSD"], 2),
            
            # Rates
            "progression_rate_MD": round(progression_rate_MD, 2),
            "progression_rate_VFI": round(progression_rate_VFI, 1),
            
            # Classification
            "risk_level": risk_level,
            "trend": trend,
            "acceleration": acceleration,
            
            # Severity changes
            "initial_severity": first_visit["severity"],
            "current_severity": last_visit["severity"],
            "severity_changed": first_visit["severity"] != last_visit["severity"],
            
            # Visit history
            "visit_history": visit_metrics,
            
            # Warnings
            "warnings": warnings,
            
            # Statistical measures
            "md_std_dev": round(float(np.std([v["MD"] for v in visit_metrics])), 2),
            "vfi_std_dev": round(float(np.std([v["VFI"] for v in visit_metrics])), 2)
        }
    
    def _classify_progression_risk(self, rate_magnitude: float) -> str:
        """
        Classify progression risk based on rate
        
        Args:
            rate_magnitude: Absolute value of progression rate (dB/year)
            
        Returns:
            Risk level: Slow, Moderate, Rapid, or Very Rapid
        """
        if rate_magnitude < PROGRESSION_THRESHOLDS["Slow"]:
            return "Slow"
        elif rate_magnitude < PROGRESSION_THRESHOLDS["Moderate"]:
            return "Moderate"
        elif rate_magnitude < PROGRESSION_THRESHOLDS["Rapid"]:
            return "Rapid"
        else:
            return "Very Rapid"
    
    def _analyze_acceleration(self, visit_metrics: List[Dict], total_time: float) -> str:
        """
        Detect if progression is accelerating or decelerating
        
        Args:
            visit_metrics: List of visit metrics
            total_time: Total time span in years
            
        Returns:
            Acceleration status
        """
        if len(visit_metrics) < 3:
            return "Insufficient data"
        
        # Split into first half and second half
        mid_point = len(visit_metrics) // 2
        
        first_half = visit_metrics[:mid_point+1]
        second_half = visit_metrics[mid_point:]
        
        # Calculate rates for each half
        first_half_delta = first_half[-1]["MD"] - first_half[0]["MD"]
        second_half_delta = second_half[-1]["MD"] - second_half[0]["MD"]
        
        first_half_time = len(first_half) - 1
        second_half_time = len(second_half) - 1
        
        if first_half_time > 0 and second_half_time > 0:
            first_half_rate = abs(first_half_delta / first_half_time)
            second_half_rate = abs(second_half_delta / second_half_time)
            
            # Compare rates
            if second_half_rate > first_half_rate * 1.5:
                return "Accelerating"
            elif second_half_rate < first_half_rate * 0.5:
                return "Decelerating"
            else:
                return "Constant"
        
        return "Insufficient data"
    
    def _detect_warning_signs(self, visit_metrics: List[Dict], progression_rate: float) -> List[str]:
        """
        Detect concerning patterns in progression
        
        Args:
            visit_metrics: List of visit metrics
            progression_rate: Progression rate in dB/year
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check for rapid progression
        if abs(progression_rate) > PROGRESSION_THRESHOLDS["Rapid"]:
            warnings.append("Rapid progression detected (>2 dB/year)")
        
        # Check for severity escalation
        if visit_metrics[0]["severity"] != visit_metrics[-1]["severity"]:
            warnings.append(f"Severity worsened from {visit_metrics[0]['severity']} to {visit_metrics[-1]['severity']}")
        
        # Check for large single-visit drops
        for i in range(1, len(visit_metrics)):
            md_drop = visit_metrics[i]["MD"] - visit_metrics[i-1]["MD"]
            if md_drop < -3.0:  # More than 3 dB drop in one visit
                warnings.append(f"Significant MD drop at visit {visit_metrics[i]['visit']}: {round(md_drop, 2)} dB")
        
        # Check for consistently worsening trend
        md_values = [v["MD"] for v in visit_metrics]
        if all(md_values[i] <= md_values[i-1] for i in range(1, len(md_values))):
            warnings.append("Consistently worsening MD across all visits")
        
        # Check current severity
        current_md = visit_metrics[-1]["MD"]
        if current_md < -12:
            warnings.append(f"Current MD indicates severe/advanced disease (MD={current_md} dB)")
        
        return warnings
    
    def calculate_slope(self, visit_metrics: List[Dict]) -> Dict:
        """
        Calculate linear regression slope for MD over time
        
        Args:
            visit_metrics: List of visit metrics
            
        Returns:
            Slope analysis
        """
        if len(visit_metrics) < 2:
            return {"error": "Insufficient data for slope calculation"}
        
        visits = np.array([v["visit"] for v in visit_metrics])
        md_values = np.array([v["MD"] for v in visit_metrics])
        
        # Linear regression
        slope, intercept = np.polyfit(visits, md_values, 1)
        
        # R-squared
        y_pred = slope * visits + intercept
        ss_res = np.sum((md_values - y_pred) ** 2)
        ss_tot = np.sum((md_values - np.mean(md_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "slope_per_visit": round(float(slope), 3),
            "slope_per_year": round(float(slope / (self.visit_interval_months / 12)), 3),
            "intercept": round(float(intercept), 2),
            "r_squared": round(float(r_squared), 3),
            "linearity": "Good" if r_squared > 0.8 else "Moderate" if r_squared > 0.5 else "Poor"
        }


def main():
    """Test progression analysis"""
    # Test data (simulated 3 visits)
    test_visits = [
        {
            "visit": 0,
            "mae_dB": 7.54,
            "clinical_metrics": {
                "MD": -9.7,
                "VFI": 67.7,
                "PSD": 8.5,
                "severity": "Moderate"
            }
        },
        {
            "visit": 1,
            "mae_dB": 6.87,
            "clinical_metrics": {
                "MD": -9.5,
                "VFI": 68.1,
                "PSD": 8.3,
                "severity": "Moderate"
            }
        },
        {
            "visit": 2,
            "mae_dB": 7.08,
            "clinical_metrics": {
                "MD": -10.2,
                "VFI": 65.3,
                "PSD": 9.1,
                "severity": "Moderate"
            }
        }
    ]
    
    analyzer = ProgressionAnalyzer()
    analysis = analyzer.analyze_progression(test_visits)
    
    print("\n" + "="*60)
    print("PROGRESSION ANALYSIS TEST")
    print("="*60)
    print(f"\nStatus: {analysis['status']}")
    print(f"Visits: {analysis['num_visits']}")
    print(f"Time span: {analysis['time_span_years']} years")
    print(f"\nProgression Rate: {analysis['progression_rate_MD']} dB/year")
    print(f"Risk Level: {analysis['risk_level']}")
    print(f"Trend: {analysis['trend']}")
    print(f"Acceleration: {analysis['acceleration']}")
    
    if analysis['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in analysis['warnings']:
            print(f"  - {warning}")
    
    # Test slope calculation
    slope_analysis = analyzer.calculate_slope(test_visits)
    print(f"\nSlope Analysis:")
    print(f"  Slope: {slope_analysis['slope_per_year']} dB/year")
    print(f"  R²: {slope_analysis['r_squared']}")
    print(f"  Linearity: {slope_analysis['linearity']}")
    
    print("\n[OK] Progression analysis test complete!")


if __name__ == "__main__":
    main()