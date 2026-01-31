"""
report_generator.py - LLM Report Generation and Formatting
"""

import json
import os
from typing import Dict
from datetime import datetime
import google.generativeai as genai
from config import (
    GEMINI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_OUTPUT_TOKENS,
    JSON_REPORTS_DIR, TEXT_REPORTS_DIR, MD_REPORTS_DIR, logger
)


class ReportGenerator:
    """Generate clinical reports using Gemini LLM"""
    
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(LLM_MODEL)
        logger.info(f"ReportGenerator initialized with model: {LLM_MODEL}")
    
    def generate_report(
        self,
        prompt: str,
        system_instruction: str = None,
        temperature: float = LLM_TEMPERATURE
    ) -> str:
        """
        Generate clinical report using Gemini
        
        Args:
            prompt: Complete prompt with patient data and guidelines
            system_instruction: Optional system instruction
            temperature: Sampling temperature (lower = more factual)
            
        Returns:
            Generated report text
        """
        logger.info("Generating clinical report with Gemini...")
        logger.info(f"Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
        
        try:
            # Generate content
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    candidate_count=1
                )
            )
            
            report_text = response.text
            
            logger.info(f"Report generated: {len(report_text)} characters")
            return report_text
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def create_structured_output(
        self,
        narrative_report: str,
        patient_data: Dict,
        progression_analysis: Dict,
        forecast: Dict,
        retrieved_chunks: list
    ) -> Dict:
        """
        Create structured JSON output combining narrative and data
        
        Args:
            narrative_report: Generated narrative report
            patient_data: Patient information
            progression_analysis: Progression analysis results
            forecast: Forecast results
            retrieved_chunks: Retrieved guideline chunks
            
        Returns:
            Structured report dictionary
        """
        latest_visit = patient_data["visits"][-1]
        
        # Extract citations
        sources_used = list(set([chunk["source"] for chunk in retrieved_chunks]))
        
        structured_output = {
            # Metadata
            "report_metadata": {
                "patient_id": patient_data["patient_id"],
                "report_generated_date": datetime.now().isoformat(),
                "report_type": "Glaucoma Progression Analysis",
                "visits_analyzed": [v["visit"] for v in patient_data["visits"]],
                "analysis_timespan_months": progression_analysis.get("time_span_months", 0),
                "analysis_timespan_years": progression_analysis.get("time_span_years", 0),
                "guidelines_used": sources_used
            },
            
            # Current Status
            "current_status": {
                "visit_number": latest_visit["visit"],
                "MD": latest_visit["clinical_metrics"]["MD"],
                "VFI": latest_visit["clinical_metrics"]["VFI"],
                "PSD": latest_visit["clinical_metrics"]["PSD"],
                "mean_vf": latest_visit["clinical_metrics"]["mean_vf"],
                "severity": latest_visit["clinical_metrics"]["severity"],
                "pattern": latest_visit["clinical_metrics"]["pattern"],
                "superior_mean": latest_visit["clinical_metrics"]["superior_mean"],
                "inferior_mean": latest_visit["clinical_metrics"]["inferior_mean"]
            },
            
            # Progression Analysis
            "progression_analysis": {
                "status": progression_analysis.get("status", "unknown"),
                "rate_dB_per_year": progression_analysis.get("progression_rate_MD", None),
                "rate_VFI_per_year": progression_analysis.get("progression_rate_VFI", None),
                "risk_level": progression_analysis.get("risk_level", "Unknown"),
                "trend": progression_analysis.get("trend", "Unknown"),
                "acceleration": progression_analysis.get("acceleration", "Unknown"),
                "delta_MD": progression_analysis.get("delta_MD", None),
                "delta_VFI": progression_analysis.get("delta_VFI", None),
                "warnings": progression_analysis.get("warnings", [])
            },
            
            # Forecast
            "forecast": {
                "method": forecast.get("method", "Unknown"),
                "current_severity": forecast.get("current_severity", "Unknown"),
                "progression_rate": forecast.get("progression_rate", None),
                "predictions": forecast.get("forecasts", []),
                "time_to_next_stage": forecast.get("time_to_next_severity_stage", {}),
                "risk_assessment": forecast.get("risk_assessment", {})
            },
            
            # AI Model Performance
            "model_performance": {
                "visits": [
                    {
                        "visit": v["visit"],
                        "mae_dB": v.get("mae_dB", None),
                        "rmse_dB": v.get("rmse_dB", None)
                    }
                    for v in patient_data["visits"]
                ],
                "average_mae": sum([v.get("mae_dB", 0) for v in patient_data["visits"]]) / len(patient_data["visits"]),
                "average_rmse": sum([v.get("rmse_dB", 0) for v in patient_data["visits"]]) / len(patient_data["visits"])
            },
            
            # Visit History
            "visit_history": [
                {
                    "visit": v["visit"],
                    "MD": v["clinical_metrics"]["MD"],
                    "VFI": v["clinical_metrics"]["VFI"],
                    "severity": v["clinical_metrics"]["severity"]
                }
                for v in patient_data["visits"]
            ],
            
            # Narrative Report
            "narrative_report": narrative_report,
            
            # Guidelines Citations
            "guidelines_citations": {
                source: len([c for c in retrieved_chunks if c["source"] == source])
                for source in sources_used
            }
        }
        
        return structured_output
    
    def save_report(
        self,
        structured_output: Dict,
        patient_id: str,
        output_formats: list = ["json", "txt", "md"]
    ) -> Dict[str, str]:
        """
        Save report in multiple formats
        
        Args:
            structured_output: Structured report data
            patient_id: Patient identifier
            output_formats: List of formats to save (json, txt, md)
            
        Returns:
            Dictionary of saved file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{patient_id}_{timestamp}"
        
        saved_files = {}
        
        # JSON format (structured data)
        if "json" in output_formats:
            json_path = os.path.join(JSON_REPORTS_DIR, f"{base_filename}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(structured_output, f, indent=2, ensure_ascii=False)
            saved_files["json"] = json_path
            logger.info(f"[OK] Saved JSON: {json_path}")
        
        # Text format (custom 10 lines + 5-line clinical summary)
        if "txt" in output_formats:
            txt_path = os.path.join(TEXT_REPORTS_DIR, f"{base_filename}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                # Header
                f.write(f"CLINICAL REPORT: {patient_id}\n")
                f.write(f"Generated: {structured_output['report_metadata']['report_generated_date']}\n")
                f.write(f"{'='*70}\n\n")
                
                # 10 Key Metrics
                cs = structured_output["current_status"]
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"Current MD: {cs.get('MD', 'N/A')} dB\n")
                f.write(f"Progression Rate: {structured_output['progression_analysis'].get('rate_dB_per_year', 'N/A')} dB/year\n")
                f.write(f"VFI: {cs.get('VFI', 'N/A')}%\n")
                f.write(f"PSD: {cs.get('PSD', 'N/A')}\n")
                f.write(f"Mean VF: {cs.get('mean_vf', 'N/A')}\n")
                f.write(f"Severity: {cs.get('severity', 'N/A')}\n")
                
                # 5-line clinical summary from narrative_report
                summary_lines = structured_output["narrative_report"].split("\n")
                summary_lines = [line.strip() for line in summary_lines if line.strip()]
                f.write("Clinical Summary:\n")
                for line in summary_lines[:5]:  # first 5 lines
                    f.write(f"- {line}\n")
            
            saved_files["txt"] = txt_path
            logger.info(f"[OK] Saved TXT: {txt_path}")

        
        # Markdown format (formatted narrative)
        if "md" in output_formats:
            md_path = os.path.join(MD_REPORTS_DIR, f"{base_filename}.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# Clinical Glaucoma Progression Report\n\n")
                f.write(f"**Patient ID:** {patient_id}  \n")
                f.write(f"**Report Generated:** {structured_output['report_metadata']['report_generated_date']}  \n")
                f.write(f"**Visits Analyzed:** {len(structured_output['visit_history'])}  \n")
                f.write(f"**Analysis Period:** {structured_output['report_metadata']['analysis_timespan_months']} months  \n\n")
                f.write(f"---\n\n")
                f.write(structured_output["narrative_report"])
                f.write(f"\n\n---\n\n")
                f.write(f"## Data Summary\n\n")
                f.write(f"**Current Status:**\n")
                f.write(f"- MD: {structured_output['current_status']['MD']} dB\n")
                f.write(f"- VFI: {structured_output['current_status']['VFI']}%\n")
                f.write(f"- Severity: {structured_output['current_status']['severity']}\n\n")
                f.write(f"**Progression:**\n")
                f.write(f"- Rate: {structured_output['progression_analysis']['rate_dB_per_year']} dB/year\n")
                f.write(f"- Risk: {structured_output['progression_analysis']['risk_level']}\n\n")
                f.write(f"**Guidelines Used:** {', '.join(structured_output['report_metadata']['guidelines_used'])}\n")
            saved_files["md"] = md_path
            logger.info(f"[OK] Saved MD: {md_path}")
        
        logger.info(f"Report saved in {len(saved_files)} formats")
        return saved_files


def main():
    """Test report generator"""
    # This would normally use real data from the full pipeline
    test_prompt = """
Based on the following guidelines:
[AAO 2024] Moderate glaucoma requires target IOP <15 mmHg...

Patient: 101_OD
Current MD: -10.2 dB
Progression: -1.0 dB/year

Generate a brief clinical summary.
"""
    
    generator = ReportGenerator()
    
    print("\n" + "="*60)
    print("REPORT GENERATOR TEST")
    print("="*60)
    print("\nGenerating test report...")
    
    try:
        report = generator.generate_report(test_prompt, temperature=0.3)
        print(f"\n[OK] Report generated ({len(report)} chars)")
        print(f"\nFirst 300 characters:")
        print(report[:300] + "...")
        
        # Test structured output creation
        test_structured = {
            "report_metadata": {
                "patient_id": "101_OD",
                "report_generated_date": datetime.now().isoformat(),
                "guidelines_used": ["AAO", "WHO"]
            },
            "current_status": {"MD": -10.2, "VFI": 65.3, "severity": "Moderate"},
            "progression_analysis": {"rate_dB_per_year": -1.0, "risk_level": "Moderate"},
            "forecast": {"predictions": []},
            "narrative_report": report
        }
        
        # Test saving
        print("\nSaving test report...")
        saved = generator.save_report(test_structured, "TEST_101_OD", ["json", "txt", "md"])
        print(f"\n[OK] Saved {len(saved)} files")
        for fmt, path in saved.items():
            print(f"  {fmt.upper()}: {path}")
        
        print("\n[OK] Report generator test complete!")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("Make sure GEMINI_API_KEY is set in .env file")


if __name__ == "__main__":
    main()