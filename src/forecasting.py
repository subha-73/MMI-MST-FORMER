"""
forecasting.py - Future Disease Progression Forecasting
"""

from typing import Dict, List
import numpy as np
from config import SEVERITY_THRESHOLDS, FORECAST_HORIZONS, logger


class ProgressionForecaster:
    """Forecast future visual field progression"""
    
    def __init__(self):
        self.severity_thresholds = SEVERITY_THRESHOLDS
    
    def forecast_linear(
        self, 
        current_metrics: Dict, 
        progression_rate: float,
        time_horizons: List[float] = None
    ) -> Dict:
        """
        Forecast future progression using linear extrapolation
        
        Args:
            current_metrics: Latest visit clinical metrics
            progression_rate: Progression rate in dB/year
            time_horizons: List of forecast times in years (default from config)
            
        Returns:
            Forecast predictions
        """
        if time_horizons is None:
            time_horizons = FORECAST_HORIZONS
        
        current_MD = current_metrics["MD"]
        current_VFI = current_metrics["VFI"]
        
        logger.info(f"Forecasting from MD={current_MD} dB at rate={progression_rate} dB/year")
        
        forecasts = []
        
        for time_years in time_horizons:
            # Linear extrapolation for MD
            predicted_MD = current_MD + (progression_rate * time_years)
            
            # Extrapolate VFI (proportional to MD change)
            if current_MD != 0:
                vfi_rate = (progression_rate / abs(current_MD)) * current_VFI
                predicted_VFI = current_VFI + (vfi_rate * time_years)
            else:
                predicted_VFI = current_VFI
            
            # Clamp VFI to valid range
            predicted_VFI = max(0, min(100, predicted_VFI))
            
            # Determine severity at this forecast point
            predicted_severity = self._classify_severity(predicted_MD)
            
            # Calculate confidence (decreases with time)
            confidence = self._calculate_confidence(time_years, len(time_horizons))
            
            forecasts.append({
                "time_horizon_years": time_years,
                "time_horizon_months": int(time_years * 12),
                "predicted_MD": round(predicted_MD, 2),
                "predicted_VFI": round(predicted_VFI, 1),
                "predicted_severity": predicted_severity,
                "confidence": confidence
            })
        
        # Calculate time to reach next severity stage
        time_to_next_stage = self._calculate_time_to_next_stage(
            current_MD, 
            current_metrics["severity"],
            progression_rate
        )
        
        # Risk assessment
        risk_assessment = self._assess_progression_risk(
            current_MD,
            progression_rate,
            forecasts
        )
        
        return {
            "method": "linear_extrapolation",
            "current_MD": current_MD,
            "current_VFI": current_VFI,
            "current_severity": current_metrics["severity"],
            "progression_rate": progression_rate,
            "forecasts": forecasts,
            "time_to_next_severity_stage": time_to_next_stage,
            "risk_assessment": risk_assessment
        }
    
    def _classify_severity(self, MD: float) -> str:
        """Classify severity based on MD value"""
        if MD >= self.severity_thresholds["Mild"]:
            return "Mild"
        elif MD >= self.severity_thresholds["Moderate"]:
            return "Moderate"
        elif MD >= self.severity_thresholds["Severe"]:
            return "Severe"
        else:
            return "Advanced"
    
    def _calculate_confidence(self, time_years: float, total_horizons: int) -> str:
        """
        Calculate confidence level for forecast
        
        Confidence decreases with longer time horizons
        """
        if time_years <= 0.5:
            return "High"
        elif time_years <= 1.0:
            return "Moderate"
        else:
            return "Low"
    
    def _calculate_time_to_next_stage(
        self, 
        current_MD: float, 
        current_severity: str,
        progression_rate: float
    ) -> Dict:
        """
        Calculate time until patient reaches next severity stage
        
        Args:
            current_MD: Current mean deviation
            current_severity: Current severity classification
            progression_rate: Rate of progression (dB/year)
            
        Returns:
            Time to next stage information
        """
        if progression_rate >= 0:
            # Improving or stable - no progression to next stage
            return {
                "status": "Not progressing",
                "message": "MD is stable or improving",
                "years": None,
                "months": None
            }
        
        # Find next worse severity threshold
        severity_order = ["Mild", "Moderate", "Severe", "Advanced"]
        
        try:
            current_idx = severity_order.index(current_severity)
        except ValueError:
            return {
                "status": "Unknown severity",
                "message": f"Cannot calculate for severity: {current_severity}",
                "years": None,
                "months": None
            }
        
        if current_idx >= len(severity_order) - 1:
            # Already at worst stage
            return {
                "status": "At final stage",
                "message": "Already at advanced stage",
                "years": None,
                "months": None
            }
        
        # Get next stage threshold
        next_severity = severity_order[current_idx + 1]
        next_threshold = self.severity_thresholds[next_severity]
        
        # Calculate time to reach threshold
        md_difference = next_threshold - current_MD
        time_years = md_difference / progression_rate
        
        if time_years < 0:
            time_years = abs(time_years)
        
        return {
            "status": "Calculated",
            "next_severity": next_severity,
            "next_threshold_MD": next_threshold,
            "md_difference": round(md_difference, 2),
            "years": round(time_years, 1),
            "months": round(time_years * 12, 0),
            "message": f"Estimated {round(time_years, 1)} years to {next_severity} stage"
        }
    
    def _assess_progression_risk(
        self,
        current_MD: float,
        progression_rate: float,
        forecasts: List[Dict]
    ) -> Dict:
        """
        Assess overall risk based on current status and forecast
        
        Args:
            current_MD: Current mean deviation
            progression_rate: Rate of progression
            forecasts: List of forecast predictions
            
        Returns:
            Risk assessment
        """
        risk_factors = []
        
        # Check current severity
        if current_MD < -12:
            risk_factors.append("Already in severe/advanced stage")
        
        # Check progression rate
        if abs(progression_rate) > 2.0:
            risk_factors.append(f"Very rapid progression ({abs(progression_rate)} dB/year)")
        elif abs(progression_rate) > 1.0:
            risk_factors.append(f"Rapid progression ({abs(progression_rate)} dB/year)")
        
        # Check 1-year forecast
        one_year_forecast = next((f for f in forecasts if f["time_horizon_years"] == 1.0), None)
        if one_year_forecast:
            if one_year_forecast["predicted_MD"] < -20:
                risk_factors.append("Predicted to reach advanced stage within 1 year")
            elif one_year_forecast["predicted_MD"] < -12:
                risk_factors.append("Predicted to reach severe stage within 1 year")
        
        # Overall risk level
        if len(risk_factors) >= 3:
            risk_level = "Very High"
        elif len(risk_factors) >= 2:
            risk_level = "High"
        elif len(risk_factors) >= 1:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "num_risk_factors": len(risk_factors)
        }
    
    def forecast_with_confidence_intervals(
        self,
        current_metrics: Dict,
        progression_rate: float,
        rate_std_dev: float,
        time_horizons: List[float] = None
    ) -> Dict:
        """
        Forecast with confidence intervals (best/worst case)
        
        Args:
            current_metrics: Latest visit metrics
            progression_rate: Mean progression rate
            rate_std_dev: Standard deviation of rate
            time_horizons: Forecast times
            
        Returns:
            Forecast with confidence intervals
        """
        if time_horizons is None:
            time_horizons = FORECAST_HORIZONS
        
        current_MD = current_metrics["MD"]
        
        forecasts_with_ci = []
        
        for time_years in time_horizons:
            # Best case (rate - 1 std dev)
            best_case_rate = progression_rate + rate_std_dev  # Less negative = better
            best_case_MD = current_MD + (best_case_rate * time_years)
            
            # Expected case
            expected_MD = current_MD + (progression_rate * time_years)
            
            # Worst case (rate - 1 std dev)
            worst_case_rate = progression_rate - rate_std_dev  # More negative = worse
            worst_case_MD = current_MD + (worst_case_rate * time_years)
            
            forecasts_with_ci.append({
                "time_horizon_years": time_years,
                "time_horizon_months": int(time_years * 12),
                "best_case_MD": round(best_case_MD, 2),
                "expected_MD": round(expected_MD, 2),
                "worst_case_MD": round(worst_case_MD, 2),
                "uncertainty_range": round(abs(best_case_MD - worst_case_MD), 2)
            })
        
        return {
            "method": "linear_with_confidence_intervals",
            "forecasts": forecasts_with_ci,
            "rate_std_dev": round(rate_std_dev, 2)
        }


def main():
    """Test forecasting"""
    # Test data
    test_metrics = {
        "MD": -10.2,
        "VFI": 65.3,
        "severity": "Moderate"
    }
    
    progression_rate = -1.0  # dB/year
    
    forecaster = ProgressionForecaster()
    forecast = forecaster.forecast_linear(test_metrics, progression_rate)
    
    print("\n" + "="*60)
    print("PROGRESSION FORECASTING TEST")
    print("="*60)
    print(f"\nCurrent Status:")
    print(f"  MD: {forecast['current_MD']} dB")
    print(f"  VFI: {forecast['current_VFI']}%")
    print(f"  Severity: {forecast['current_severity']}")
    print(f"\nProgression Rate: {forecast['progression_rate']} dB/year")
    
    print(f"\nForecasts:")
    for f in forecast['forecasts']:
        print(f"\n  At {f['time_horizon_months']} months:")
        print(f"    MD: {f['predicted_MD']} dB")
        print(f"    VFI: {f['predicted_VFI']}%")
        print(f"    Severity: {f['predicted_severity']}")
        print(f"    Confidence: {f['confidence']}")
    
    time_to_next = forecast['time_to_next_severity_stage']
    if time_to_next['years']:
        print(f"\nTime to Next Stage:")
        print(f"  {time_to_next['message']}")
        print(f"  ({time_to_next['months']} months)")
    
    risk = forecast['risk_assessment']
    print(f"\nRisk Assessment:")
    print(f"  Level: {risk['risk_level']}")
    if risk['risk_factors']:
        print(f"  Factors:")
        for factor in risk['risk_factors']:
            print(f"    - {factor}")
    
    print("\n[OK] Forecasting test complete!")


if __name__ == "__main__":
    main()