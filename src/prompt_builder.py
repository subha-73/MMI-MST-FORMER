"""
prompt_builder.py - LLM Prompt Construction
"""

from typing import Dict, List
from datetime import datetime
from config import logger


class PromptBuilder:
    """Build comprehensive prompts for clinical report generation"""
    
    def __init__(self):
        self.system_instruction = self._get_system_instruction()
    
    def _get_system_instruction(self) -> str:
        """Get system instruction for clinical AI"""
        return """You are a clinical decision support AI assistant specializing in ophthalmology and glaucoma management.

Your role is to:
1. Interpret clinical visual field data accurately
2. Apply evidence-based guidelines (AAO, WHO, Indian)
3. Provide clear, actionable recommendations
4. Maintain medical precision and clarity

You must:
- Only use information from provided clinical guidelines
- Cite sources for all recommendations (e.g., "According to AAO 2024...")
- Acknowledge uncertainty when appropriate
- Never provide definitive diagnoses or treatment prescriptions
- Frame outputs as clinical decision support, not final medical advice
- Use precise medical terminology
- Structure output with clear section headers"""
    
    def build_comprehensive_prompt(
        self,
        patient_data: Dict,
        progression_analysis: Dict,
        forecast: Dict,
        retrieved_chunks: List[Dict],
        retrieved_chunks_formatted: str
    ) -> str:
        """
        Build complete prompt with all patient data and guidelines
        
        Args:
            patient_data: Patient information with all visits
            progression_analysis: Multi-visit progression analysis
            forecast: Future progression forecast
            retrieved_chunks: List of retrieved guideline chunks
            retrieved_chunks_formatted: Pre-formatted guideline text
            
        Returns:
            Complete prompt string
        """
        logger.info("Building comprehensive prompt...")
        
        # Format visit history
        visit_history = self._format_visit_history(patient_data["visits"])
        
        # Format progression metrics
        progression_text = self._format_progression_analysis(progression_analysis)
        
        # Format forecast
        forecast_text = self._format_forecast(forecast)
        
        # Build complete prompt
        prompt = f"""
{'='*70}
CLINICAL GUIDELINES CONTEXT
{'='*70}

{retrieved_chunks_formatted}

{'='*70}
PATIENT DATA - LONGITUDINAL ANALYSIS
{'='*70}

Patient ID: {patient_data['patient_id']}
Number of Visits Analyzed: {len(patient_data['visits'])}
Analysis Period: {progression_analysis.get('time_span_months', 'N/A')} months ({progression_analysis.get('time_span_years', 'N/A')} years)
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{visit_history}

{'='*70}
PROGRESSION ANALYSIS
{'='*70}

{progression_text}

{'='*70}
FORECAST (BASED ON CURRENT PROGRESSION RATE)
{'='*70}

{forecast_text}

{'='*70}
TASK: GENERATE COMPREHENSIVE CLINICAL REPORT
{'='*70}

Generate a detailed clinical report with the following sections:

1. EXECUTIVE SUMMARY
   - 2-3 sentence overview of patient's current status
   - Key findings and concerns
   - Urgency level (Routine / Prompt / Urgent)

2. CURRENT DISEASE STATUS (Latest Visit)
   - Severity classification with specific guideline reference
   - Mean Deviation (MD) and Visual Field Index (VFI) interpretation
   - Visual field defect pattern description
   - Affected anatomical regions (superior/inferior fields)
   - Current versus normal visual function

3. PROGRESSION ANALYSIS ⭐
   - Interpret the calculated progression rate ({progression_analysis.get('progression_rate_MD', 'N/A')} dB/year)
   - Compare rate against guideline criteria for slow/moderate/rapid progression
   - Explain the trend (improving/stable/worsening) with clinical significance
   - Discuss any acceleration or deceleration patterns
   - Identify modifiable and non-modifiable risk factors contributing to progression

4. FORECAST AND PROGNOSIS ⭐
   - Interpret forecasted MD values at 6 months, 1 year, and 2 years
   - Predict when patient will likely reach next severity stage
   - Assess risk of vision-threatening progression
   - Explain confidence level in predictions
   - Discuss factors that could alter the forecast trajectory

5. AI MODEL PERFORMANCE EVALUATION
   - Assess MST model prediction accuracy trends (MAE/RMSE across visits)
   - Comment on reliability for clinical decision support
   - Note any concerning prediction errors or patterns
   - Discuss clinical utility of AI predictions

6. RISK STRATIFICATION
   - Overall risk level (Low / Moderate / High / Very High)
   - List key modifiable risk factors
   - List non-modifiable risk factors
   - Assess likelihood of progression to vision-threatening stages
   - Consider patient-specific risk factors

7. CLINICAL RECOMMENDATIONS
   - Assessment of current treatment adequacy based on progression
   - Recommendations for therapy intensification (if needed)
   - Target IOP recommendations with guideline citation
   - Additional interventions to consider
   - Lifestyle modifications
   - **CRITICAL: Cite specific guidelines** for each recommendation

8. MONITORING AND FOLLOW-UP PLAN
   - Next appointment timing based on progression risk level
   - Visual field testing frequency
   - Additional tests required (OCT RNFL, optic nerve imaging, IOP)
   - Red flags requiring urgent re-evaluation
   - Long-term monitoring strategy (1-2 year plan)
   - Criteria for treatment modification

CRITICAL INSTRUCTIONS:
- Base ALL recommendations strictly on the retrieved guidelines above
- Cite specific guideline sources and sections when making recommendations
- Do NOT recommend treatments outside guideline scope
- Explain medical reasoning clearly for each recommendation
- Use precise medical terminology appropriate for healthcare professionals
- Acknowledge any limitations in the data or analysis
- Maintain professional, evidence-based tone throughout
- Structure output with clear markdown section headers
- Be thorough but concise - aim for clarity over verbosity
"""
        
        logger.info(f"Prompt built: {len(prompt)} characters")
        return prompt
    
    def _format_visit_history(self, visits: List[Dict]) -> str:
        """Format visit history section"""
        text = ""
        
        for visit in visits:
            metrics = visit.get("clinical_metrics", {})
            
            text += f"\n{'─'*60}\n"
            text += f"VISIT {visit['visit']}\n"
            text += f"{'─'*60}\n"
            text += f"Clinical Metrics:\n"
            text += f"  • Mean Deviation (MD): {metrics.get('MD', 'N/A')} dB\n"
            text += f"  • Visual Field Index (VFI): {metrics.get('VFI', 'N/A')}%\n"
            text += f"  • Pattern Standard Deviation (PSD): {metrics.get('PSD', 'N/A')} dB\n"
            text += f"  • Mean VF Sensitivity: {metrics.get('mean_vf', 'N/A')} dB\n"
            text += f"  • Severity Classification: {metrics.get('severity', 'N/A')}\n"
            text += f"\nSpatial Analysis:\n"
            text += f"  • Defect Pattern: {metrics.get('pattern', 'N/A')}\n"
            text += f"  • Superior Field: {metrics.get('superior_mean', 'N/A')} dB\n"
            text += f"  • Inferior Field: {metrics.get('inferior_mean', 'N/A')} dB\n"
            
            # Add Excel clinical data if available
            if 'additional_clinical_data' in visit:
                from excel_loader import format_clinical_data_for_prompt
                text += f"\n{format_clinical_data_for_prompt(visit)}"
            
            text += f"\nAI Model Performance:\n"
            text += f"  • Mean Absolute Error (MAE): {visit.get('mae_dB', 'N/A')} dB\n"
            text += f"  • Root Mean Square Error (RMSE): {visit.get('rmse_dB', 'N/A')} dB\n"
            text += "\n"
        
        return text
    
    def _format_progression_analysis(self, analysis: Dict) -> str:
        """Format progression analysis section"""
        if analysis.get("status") != "analyzed":
            return "Insufficient data for progression analysis (need 2+ visits)\n"
        
        text = f"""
Overall Changes Over {analysis.get('time_span_months', 'N/A')} Months:
  • Mean Deviation change: {analysis.get('delta_MD', 'N/A')} dB
  • Visual Field Index change: {analysis.get('delta_VFI', 'N/A')}%
  • Pattern Std Dev change: {analysis.get('delta_PSD', 'N/A')} dB

Progression Metrics:
  • Rate: {analysis.get('progression_rate_MD', 'N/A')} dB/year
  • VFI Rate: {analysis.get('progression_rate_VFI', 'N/A')}% per year
  • Risk Classification: {analysis.get('risk_level', 'N/A')}
  • Overall Trend: {analysis.get('trend', 'N/A')}
  • Pattern: {analysis.get('acceleration', 'N/A')}

Severity Evolution:
  • Initial: {analysis.get('initial_severity', 'N/A')}
  • Current: {analysis.get('current_severity', 'N/A')}
  • Changed: {'Yes' if analysis.get('severity_changed') else 'No'}
"""
        
        # Add warnings if present
        warnings = analysis.get('warnings', [])
        if warnings:
            text += f"\n⚠️  Clinical Warnings:\n"
            for warning in warnings:
                text += f"  • {warning}\n"
        
        return text
    
    def _format_forecast(self, forecast: Dict) -> str:
        """Format forecast section"""
        text = f"""
Current Status:
  • Mean Deviation: {forecast.get('current_MD', 'N/A')} dB
  • Visual Field Index: {forecast.get('current_VFI', 'N/A')}%
  • Severity: {forecast.get('current_severity', 'N/A')}

Progression Rate: {forecast.get('progression_rate', 'N/A')} dB/year
Forecast Method: {forecast.get('method', 'N/A')}

Predicted Future Values:
"""
        
        for f in forecast.get('forecasts', []):
            text += f"""
  At {f['time_horizon_months']} Months ({f['time_horizon_years']} years):
    • Predicted MD: {f['predicted_MD']} dB
    • Predicted VFI: {f['predicted_VFI']}%
    • Predicted Severity: {f['predicted_severity']}
    • Confidence: {f['confidence']}
"""
        
        # Add time to next stage
        time_to_next = forecast.get('time_to_next_severity_stage', {})
        if time_to_next.get('years'):
            text += f"\nTime to Next Severity Stage:\n"
            text += f"  • {time_to_next.get('message', 'N/A')}\n"
            text += f"  • Approximately {time_to_next['months']} months\n"
        
        # Add risk assessment
        risk = forecast.get('risk_assessment', {})
        if risk:
            text += f"\nRisk Assessment:\n"
            text += f"  • Overall Risk Level: {risk.get('risk_level', 'N/A')}\n"
            if risk.get('risk_factors'):
                text += f"  • Risk Factors:\n"
                for factor in risk['risk_factors']:
                    text += f"    - {factor}\n"
        
        return text


def main():
    """Test prompt builder"""
    # Mock data
    test_patient = {
        "patient_id": "101_OD",
        "visits": [
            {
                "visit": 0,
                "mae_dB": 7.54,
                "rmse_dB": 10.24,
                "clinical_metrics": {
                    "MD": -9.7,
                    "VFI": 67.7,
                    "PSD": 8.5,
                    "mean_vf": 20.3,
                    "severity": "Moderate",
                    "pattern": "superior defect",
                    "superior_mean": 12.5,
                    "inferior_mean": 18.3
                }
            }
        ]
    }
    
    test_progression = {
        "status": "analyzed",
        "time_span_months": 12,
        "time_span_years": 1.0,
        "progression_rate_MD": -1.0,
        "risk_level": "Moderate",
        "trend": "Worsening",
        "delta_MD": -0.5,
        "delta_VFI": -2.4,
        "warnings": ["Progression detected"]
    }
    
    test_forecast = {
        "current_MD": -10.2,
        "current_VFI": 65.3,
        "current_severity": "Moderate",
        "progression_rate": -1.0,
        "forecasts": [
            {"time_horizon_months": 6, "time_horizon_years": 0.5,
             "predicted_MD": -10.7, "predicted_VFI": 64.1,
             "predicted_severity": "Moderate", "confidence": "High"}
        ]
    }
    
    test_chunks_formatted = "[AAO 2024]\nModerate glaucoma requires target IOP <15 mmHg..."
    
    builder = PromptBuilder()
    prompt = builder.build_comprehensive_prompt(
        test_patient,
        test_progression,
        test_forecast,
        [],
        test_chunks_formatted
    )
    
    print("\n" + "="*60)
    print("PROMPT BUILDER TEST")
    print("="*60)
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Estimated tokens: ~{len(prompt) // 4}")
    print("\nFirst 500 characters of prompt:")
    print(prompt[:500] + "...")
    print("\n[OK] Prompt builder test complete!")


if __name__ == "__main__":
    main()