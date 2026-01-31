import anthropic
from typing import Dict, List, Any
import json
from datetime import datetime
import base64

class MultiAgentLLMSystem:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        
        # Storage for documents and results
        self.guideline_documents = []
        self.phase1_results = None
        self.phase2_results = None
        self.phase3_results = None
        
        print("[OK] Multi-Agent LLM System Initialized")
    
    # ==========================================
    # DOCUMENT MANAGEMENT
    # ==========================================
    
    def add_guideline_document(self, file_path: str, guideline_name: str):
        """
        Add a guideline document (PDF/TXT)
        
        Args:
            file_path: Path to the PDF/TXT file
            guideline_name: Name/description of the guideline
        """
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Convert to base64
            base64_data = base64.b64encode(file_data).decode('utf-8')
            
            # Determine media type
            if file_path.endswith('.pdf'):
                media_type = "application/pdf"
            elif file_path.endswith('.txt'):
                media_type = "text/plain"
            else:
                print(f"⚠️  Warning: Unknown file type for {file_path}, treating as PDF")
                media_type = "application/pdf"
            
            # Store document
            self.guideline_documents.append({
                "name": guideline_name,
                "file_path": file_path,
                "content": {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data
                    }
                }
            })
            
            print(f"[OK] Added guideline: {guideline_name} ({file_path})")
            
        except FileNotFoundError:
            print(f"[ERROR] Error: File not found - {file_path}")
        except Exception as e:
            print(f"[ERROR] Error loading document: {str(e)}")
    
    def list_guidelines(self):
        """List all loaded guideline documents"""
        if not self.guideline_documents:
            print("No guidelines loaded yet.")
            return
        
        print("\n" + "="*60)
        print("LOADED GUIDELINES")
        print("="*60)
        for i, doc in enumerate(self.guideline_documents, 1):
            print(f"{i}. {doc['name']}")
            print(f"   File: {doc['file_path']}")
        print("="*60 + "\n")
    
    def clear_guidelines(self):
        """Clear all loaded guidelines"""
        self.guideline_documents = []
        print("[OK] All guidelines cleared")
    
    # ==========================================
    # PHASE 1: PLANNING PHASE
    # ==========================================
    
    def execute_phase1(self, predicted_vector: Dict, clinical_data: Dict) -> Dict:
        """
        Execute Phase 1: Planning Phase
        Returns planning results that can be inspected before proceeding
        """
        print("\n" + "="*60)
        print("🚀 EXECUTING PHASE 1: PLANNING PHASE")
        print("="*60)
        
        # Step 1: Pipeline Trigger
        print("\n[1/4] Pipeline Trigger Agent...")
        trigger_response = self._pipeline_trigger_agent(predicted_vector, clinical_data)
        
        # Step 2: Retrieval Agent
        print("\n[2/4] Retrieval Agent (Fetching Guidelines)...")
        knowledge = self._retrieval_agent(clinical_data)
        
        # Step 3: Context Agent
        print("\n[3/4] Context Agent...")
        context = self._context_agent(clinical_data, knowledge)
        
        # Step 4: Planner Agent
        print("\n[4/4] Planner Agent...")
        diagnostic_plan = self._planner_agent(predicted_vector, context, knowledge)
        
        self.phase1_results = {
            "trigger_status": trigger_response,
            "knowledge_base": knowledge,
            "context": context,
            "diagnostic_plan": diagnostic_plan,
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n" + "="*60)
        print("✅ PHASE 1 COMPLETE")
        print("="*60)
        
        return self.phase1_results
    
    def _pipeline_trigger_agent(self, predicted_vector: Dict, clinical_data: Dict) -> str:
        prompt = f"""You are a Pipeline Trigger Agent. Analyze the input and determine what type of clinical analysis is needed.

INPUT DATA:
- Predicted Visual Field Vector: {len(predicted_vector.get('vf_values', []))} points
- Patient ID: {clinical_data.get('patient_id')}
- Number of historical visits: {len(clinical_data.get('visit_history', []))}
- Latest IOP: {clinical_data['visit_history'][-1]['iop'] if clinical_data.get('visit_history') else 'N/A'}

TASK: Determine:
1. Is this a glaucoma progression analysis? (Yes/No)
2. What level of urgency? (Routine/Urgent/Emergency)
3. What type of analysis is needed? (Forecasting/Structural/Both)

Respond in JSON format:
{{
    "analysis_type": "glaucoma_progression",
    "urgency": "routine|urgent|emergency",
    "required_analyses": ["forecasting", "structural"],
    "reasoning": "brief explanation"
}}"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text
        print(f"   Result: {result[:100]}...")
        return result
    
    def _retrieval_agent(self, clinical_data: Dict) -> Dict:
        """Retrieval Agent - Uses uploaded guideline documents"""
        
        # Build message content
        message_content = []
        
        # Add all guideline documents first
        guideline_names = []
        for doc in self.guideline_documents:
            message_content.append(doc['content'])
            guideline_names.append(doc['name'])
        
        if guideline_names:
            guidelines_text = f"You have access to these guideline documents: {', '.join(guideline_names)}\n\n"
        else:
            guidelines_text = "No external guidelines provided. Use your medical knowledge.\n\n"
        
        # Add the prompt
        prompt = f"""You are a Medical Knowledge Retrieval Agent specializing in glaucoma.

{guidelines_text}
PATIENT CONTEXT:
- Latest IOP: {clinical_data['visit_history'][-1]['iop']} mmHg
- Latest RNFL: {clinical_data['visit_history'][-1]['rnfl']} μm
- Patient Age: {clinical_data.get('age', 'N/A')}

TASK: Extract relevant clinical guidelines and knowledge from the provided documents:
1. What is the normal range for IOP according to guidelines?
2. What RNFL thickness indicates glaucoma risk?
3. What are the glaucoma staging criteria?
4. What are standard follow-up protocols?

Provide in JSON format with source citations:
{{
    "iop_guidelines": {{
        "normal_range": "value from guidelines",
        "treatment_threshold": "value from guidelines",
        "source": "which guideline document"
    }},
    "rnfl_guidelines": {{
        "normal_range": "value",
        "borderline": "value",
        "abnormal": "value",
        "source": "guideline reference"
    }},
    "staging_criteria": {{
        "early": "criteria",
        "moderate": "criteria",
        "advanced": "criteria",
        "source": "guideline reference"
    }},
    "follow_up_protocols": {{
        "stable": "interval",
        "suspect": "interval",
        "progressing": "interval",
        "source": "guideline reference"
    }}
}}"""
        
        message_content.append({"type": "text", "text": prompt})
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": message_content}]
        )
        
        knowledge_text = response.content[0].text
        print(f"   Retrieved from {len(self.guideline_documents)} guideline(s)")
        
        try:
            clean_text = knowledge_text.replace('```json', '').replace('```', '').strip()
            knowledge = json.loads(clean_text)
        except:
            knowledge = {"raw_text": knowledge_text}
        
        return knowledge
    
    def _context_agent(self, clinical_data: Dict, knowledge: Dict) -> Dict:
        prompt = f"""You are a Clinical Context Agent. Organize the patient's clinical context.

PATIENT DATA:
{json.dumps(clinical_data.get('visit_history', []), indent=2)}

MEDICAL KNOWLEDGE:
{json.dumps(knowledge, indent=2)}

TASK: Analyze and structure the clinical context:
1. What is the patient's current disease stage?
2. What are the key risk factors present?
3. What is the progression pattern (stable/slow/rapid)?
4. Any concerning trends?

Respond in JSON:
{{
    "current_stage": "normal|suspect|early|moderate|advanced",
    "risk_factors": ["factor1", "factor2"],
    "progression_pattern": "stable|slow_progression|rapid_progression",
    "concerning_trends": ["trend1", "trend2"],
    "time_since_baseline_years": X.X
}}"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        context_text = response.content[0].text
        print(f"   Context analyzed")
        
        try:
            clean_text = context_text.replace('```json', '').replace('```', '').strip()
            context = json.loads(clean_text)
        except:
            context = {"raw_text": context_text}
        
        return context
    
    def _planner_agent(self, predicted_vector: Dict, context: Dict, knowledge: Dict) -> Dict:
        prompt = f"""You are a Diagnostic Planning Agent for glaucoma analysis.

CONTEXT:
{json.dumps(context, indent=2)}

PREDICTED VF DATA:
- Overall Mean Deviation: {predicted_vector.get('overall_mean_deviation', 'N/A')} dB
- Pattern: {predicted_vector.get('interpretation', {}).get('pattern', 'N/A')}
- Severity: {predicted_vector.get('interpretation', {}).get('severity', 'N/A')}

TASK: Create a structured diagnostic plan:
1. What analyses should be prioritized?
2. What specific aspects need detailed examination?
3. What comparisons are most relevant?

Respond in JSON:
{{
    "priority_analyses": ["forecasting", "structural"],
    "focus_areas": ["temporal_progression", "vf_changes", "structural_damage"],
    "key_questions": ["question1", "question2"],
    "comparison_timepoints": ["baseline_vs_current", "recent_acceleration"]
}}"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        plan_text = response.content[0].text
        print(f"   Diagnostic plan created")
        
        try:
            clean_text = plan_text.replace('```json', '').replace('```', '').strip()
            plan = json.loads(clean_text)
        except:
            plan = {"raw_text": plan_text}
        
        return plan
    
    # ==========================================
    # PHASE 2: ANALYSIS PHASE
    # ==========================================
    
    def execute_phase2(self, predicted_vector: Dict, clinical_data: Dict,
                      temporal_trends: Dict, spatial_insights: Dict) -> Dict:
        """
        Execute Phase 2: Analysis Phase
        Must run Phase 1 first
        """
        if self.phase1_results is None:
            print("[ERROR] Error: Phase 1 must be executed first!")
            return None
        
        print("\n" + "="*60)
        print("🔍 EXECUTING PHASE 2: ANALYSIS PHASE")
        print("="*60)
        
        # Step 1: Forecasting Tool Agent
        print("\n[1/4] Forecasting Tool Agent...")
        forecasting_results = self._forecasting_tool_agent(
            temporal_trends, predicted_vector, clinical_data
        )
        
        # Step 2: Structural Tool Agent
        print("\n[2/4] Structural Tool Agent...")
        structural_results = self._structural_tool_agent(
            spatial_insights, predicted_vector, clinical_data
        )
        
        # Step 3: Coding Agent
        print("\n[3/4] Coding Agent...")
        coding_results = self._coding_agent(predicted_vector, clinical_data)
        
        # Step 4: Orchestrator Agent
        print("\n[4/4] Orchestrator Agent...")
        orchestrated_results = self._orchestrator_agent(
            forecasting_results, structural_results, coding_results
        )
        
        self.phase2_results = {
            "forecasting": forecasting_results,
            "structural": structural_results,
            "coding": coding_results,
            "orchestrated": orchestrated_results,
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n" + "="*60)
        print("✅ PHASE 2 COMPLETE")
        print("="*60)
        
        return self.phase2_results
    
    def _forecasting_tool_agent(self, temporal_trends: Dict, 
                                predicted_vector: Dict, 
                                clinical_data: Dict) -> Dict:
        prompt = f"""You are a Forecasting Analysis Agent specializing in glaucoma progression.

TEMPORAL TRENDS:
{json.dumps(temporal_trends, indent=2)}

PREDICTED VISUAL FIELD:
{json.dumps(predicted_vector.get('interpretation', {}), indent=2)}

CRITICAL TIMEPOINTS (from model attention):
{temporal_trends.get('critical_timepoints', [])}

TASK: Perform detailed forecasting analysis:
1. Analyze the rate of progression (IOP, RNFL, VF)
2. Identify acceleration or deceleration patterns
3. Predict expected status at next visit
4. Assess reliability of forecast

Respond in JSON:
{{
    "progression_rate": "stable|slow|moderate|rapid",
    "progression_metrics": {{
        "iop_rate_per_year": X.X,
        "rnfl_rate_per_year": X.X,
        "acceleration_detected": true/false,
        "acceleration_period": "timeframe"
    }},
    "next_visit_forecast": {{
        "expected_vf_md": X.X,
        "expected_pattern": "pattern description",
        "confidence": 0.XX
    }},
    "temporal_insights": ["insight 1", "insight 2"]
}}"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        print(f"   Forecasting complete")
        
        try:
            clean_text = result_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(clean_text)
        except:
            result = {"raw_text": result_text}
        
        return result
    
    def _structural_tool_agent(self, spatial_insights: Dict,
                               predicted_vector: Dict,
                               clinical_data: Dict) -> Dict:
        prompt = f"""You are a Structural Analysis Agent for ophthalmic imaging.

SPATIAL ATTENTION INSIGHTS (what the model focused on):
{json.dumps(spatial_insights, indent=2)}

VISUAL FIELD QUADRANT DATA:
{json.dumps(predicted_vector.get('quadrant_values', {}), indent=2)}

LATEST RNFL: {clinical_data['visit_history'][-1]['rnfl']} μm

TASK: Analyze structural indicators:
1. Which anatomical regions show damage?
2. How do attention patterns correlate with VF defects?
3. What is the structure-function relationship?
4. Are there discordant findings?

Respond in JSON:
{{
    "damaged_regions": ["region1", "region2"],
    "structure_function_correlation": "concordant|discordant",
    "attention_vf_alignment": {{
        "high_attention_region": "region",
        "corresponding_vf_defect": "defect pattern",
        "correlation_strength": "strong|moderate|weak"
    }},
    "structural_insights": ["insight 1", "insight 2"]
}}"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        print(f"   Structural analysis complete")
        
        try:
            clean_text = result_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(clean_text)
        except:
            result = {"raw_text": result_text}
        
        return result
    
    def _coding_agent(self, predicted_vector: Dict, clinical_data: Dict) -> Dict:
        prompt = f"""You are a Medical Coding and Computation Agent.

VF DATA:
- Overall Mean Deviation: {predicted_vector.get('overall_mean_deviation', 0):.2f} dB
- Pattern Standard Deviation: {predicted_vector.get('pattern_standard_deviation', 0):.2f} dB

QUADRANT MEANS:
- Superior Nasal: {predicted_vector.get('quadrant_values', {}).get('superior_nasal', {}).get('mean', 0):.2f} dB
- Superior Temporal: {predicted_vector.get('quadrant_values', {}).get('superior_temporal', {}).get('mean', 0):.2f} dB
- Inferior Nasal: {predicted_vector.get('quadrant_values', {}).get('inferior_nasal', {}).get('mean', 0):.2f} dB
- Inferior Temporal: {predicted_vector.get('quadrant_values', {}).get('inferior_temporal', {}).get('mean', 0):.2f} dB

TASK: Perform clinical computations and coding:
1. Calculate Mean Deviation (MD) interpretation
2. Calculate Pattern Standard Deviation (PSD) interpretation
3. Determine Glaucoma Hemifield Test (GHT) equivalent
4. Assign severity code

Respond in JSON:
{{
    "md_interpretation": {{"value": X.X, "percentile": "normal|borderline|abnormal", "severity": "normal|early|moderate|advanced"}},
    "psd_interpretation": {{"value": X.X, "interpretation": "localized loss detected/not detected"}},
    "ght_equivalent": "within normal limits|borderline|outside normal limits",
    "severity_code": "0|1|2|3|4",
    "icd10_codes": ["H40.1111"]
}}"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        print(f"   Coding complete")
        
        try:
            clean_text = result_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(clean_text)
        except:
            result = {"raw_text": result_text}
        
        return result
    
    def _orchestrator_agent(self, forecasting: Dict, structural: Dict, 
                           coding: Dict) -> Dict:
        prompt = f"""You are an Orchestrator Agent coordinating multiple analysis streams.

FORECASTING RESULTS:
{json.dumps(forecasting, indent=2)}

STRUCTURAL RESULTS:
{json.dumps(structural, indent=2)}

CODING RESULTS:
{json.dumps(coding, indent=2)}

TASK: Integrate all analyses and identify:
1. Consistent findings across all analyses
2. Conflicting or discordant findings
3. Most significant findings
4. Key takeaways for final report

Respond in JSON:
{{
    "consistent_findings": ["finding 1", "finding 2"],
    "discordant_findings": ["discordance 1"],
    "priority_findings": [
        {{"finding": "description", "evidence": ["forecasting", "structural"], "clinical_significance": "high|medium|low"}}
    ],
    "integrated_assessment": "brief summary"
}}"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text
        print(f"   Orchestration complete")
        
        try:
            clean_text = result_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(clean_text)
        except:
            result = {"raw_text": result_text}
        
        return result
    
    # ==========================================
    # PHASE 3: SYNTHESIS PHASE
    # ==========================================
    
    def execute_phase3(self, clinical_data: Dict) -> str:
        """
        Execute Phase 3: Synthesis and Report Generation
        Must run Phase 1 and Phase 2 first
        """
        if self.phase1_results is None:
            print("[ERROR] Error: Phase 1 must be executed first!")
            return None
        
        if self.phase2_results is None:
            print("[ERROR] Error: Phase 2 must be executed first!")
            return None
        
        print("\n" + "="*60)
        print("📝 EXECUTING PHASE 3: SYNTHESIS AND REPORTING")
        print("="*60)
        
        print("\n[1/1] Decider Agent (Generating Final Report)...")
        final_report = self._decider_agent(
            self.phase1_results,
            self.phase2_results,
            clinical_data
        )
        
        self.phase3_results = {
            "final_report": final_report,
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n" + "="*60)
        print("✅ PHASE 3 COMPLETE")
        print("="*60)
        
        return final_report
    
    def _decider_agent(self, phase1: Dict, phase2: Dict, 
                      clinical_data: Dict) -> str:
        prompt = f"""You are the Decider Agent, responsible for final clinical decision-making and report generation.

PLANNING PHASE RESULTS:
Context: {json.dumps(phase1.get('context', {}), indent=2)}
Diagnostic Plan: {json.dumps(phase1.get('diagnostic_plan', {}), indent=2)}
Knowledge Base: {json.dumps(phase1.get('knowledge_base', {}), indent=2)}

ANALYSIS PHASE RESULTS:
Forecasting: {json.dumps(phase2.get('forecasting', {}), indent=2)}
Structural: {json.dumps(phase2.get('structural', {}), indent=2)}
Coding: {json.dumps(phase2.get('coding', {}), indent=2)}
Orchestrated: {json.dumps(phase2.get('orchestrated', {}), indent=2)}

PATIENT INFORMATION:
ID: {clinical_data.get('patient_id')}
Age: {clinical_data.get('age')}
Number of visits: {len(clinical_data.get('visit_history', []))}

TASK: Generate a comprehensive clinical report with these sections:

1. EXECUTIVE SUMMARY (2-3 sentences)
2. CLINICAL FINDINGS (Disease Stage, VF Analysis, Structural Assessment)
3. AI MODEL INSIGHTS (Temporal Patterns, Spatial Attention, Prediction Confidence)
4. RISK STRATIFICATION (Risk Level, Key Factors, Progression Rate)
5. RECOMMENDATIONS (Follow-up, Treatment, Additional Testing, Monitoring)
6. TECHNICAL NOTES (Model Attention, Critical Timepoints, Uncertainty)

Format as a professional medical report suitable for clinical use."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        final_report = response.content[0].text
        print(f"   Final report generated ({len(final_report)} characters)")
        
        return final_report
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def get_phase1_results(self):
        """Get Phase 1 results"""
        return self.phase1_results
    
    def get_phase2_results(self):
        """Get Phase 2 results"""
        return self.phase2_results
    
    def get_phase3_results(self):
        """Get Phase 3 results"""
        return self.phase3_results
    
    def save_all_results(self, output_file: str = "llm_results.json"):
        """Save all phase results to JSON file"""
        all_results = {
            "phase1_planning": self.phase1_results,
            "phase2_analysis": self.phase2_results,
            "phase3_synthesis": self.phase3_results,
            "guidelines_used": [doc['name'] for doc in self.guideline_documents]
        }
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"[OK] All results saved to {output_file}")
    
    def reset_phases(self):
        """Reset all phase results (keeps guidelines loaded)"""
        self.phase1_results = None
        self.phase2_results = None
        self.phase3_results = None
        print("[OK] All phases reset (guidelines retained)")


# ==========================================
# USAGE EXAMPLE
# ==========================================

if __name__ == "__main__":
    
    # Initialize system
    agent_system = MultiAgentLLMSystem(api_key="your-anthropic-api-key")
    
    # ==========================================
    # STEP 1: UPLOAD GUIDELINE DOCUMENTS
    # ==========================================
    
    print("\n📄 UPLOADING GUIDELINE DOCUMENTS...")
    agent_system.add_guideline_document(
        "WHO_Glaucoma_Guidelines_2024.pdf", 
        "WHO Glaucoma Guidelines 2024"
    )
    agent_system.add_guideline_document(
        "AAO_Preferred_Practice_Pattern.pdf", 
        "AAO Preferred Practice Pattern"
    )
    agent_system.add_guideline_document(
        "Indian_Guidelines.pdf", 
        "Indian Glaucoma Society Guidelines"
    )
    
    # List loaded guidelines
    agent_system.list_guidelines()
    
    # ==========================================
    # PREPARE INPUT DATA
    # ==========================================
    
    predicted_vector = {
        "vf_values": [0.23] * 61,
        "overall_mean_deviation": -8.5,
        "pattern_standard_deviation": 4.2,
        "interpretation": {
            "severity": "Moderate defect",
            "pattern": "Superior arcuate defect"
        },
        "quadrant_values": {
            "superior_nasal": {"mean": 22.3, "values": []},
            "superior_temporal": {"mean": 18.7, "values": []},
            "inferior_nasal": {"mean": 26.1, "values": []},
            "inferior_temporal": {"mean": 25.8, "values": []}
        }
    }
    
    clinical_data = {
        "patient_id": "12345",
        "age": 63,
        "gender": "Male",
        "visit_history": [
            {"date": "2015-03-10", "iop": 15, "rnfl": 92, "age": 55, "status": "Normal"},
            {"date": "2017-05-22", "iop": 17, "rnfl": 88, "age": 57, "status": "Normal"},
            {"date": "2019-08-15", "iop": 19, "rnfl": 84, "age": 59, "status": "Suspect"},
            {"date": "2021-11-03", "iop": 20, "rnfl": 80, "age": 61, "status": "Suspect"},
            {"date": "2023-02-18", "iop": 21, "rnfl": 78, "age": 63, "status": "Suspect"}
        ]
    }
    
    temporal_trends = {
        "iop": {"trend": "increasing", "change": "+6.0 mmHg"},
        "rnfl": {"trend": "decreasing", "change": "-14.0 μm"},
        "critical_timepoints": [
            "2021-11-03 (weight: 0.345)",
            "2023-02-18 (weight: 0.412)"
        ],
        "total_follow_up_years": 8.0
    }
    
    spatial_insights = {
        "high_attention_regions": ["Optic disc", "Superior temporal"],
        "region_importance": {
            "Optic disc": "8/10",
            "Superior temporal": "6/10"
        }
    }
    
    # ==========================================
    # STEP 2: EXECUTE PHASE 1 (PLANNING)
    # ==========================================
    
    phase1_results = agent_system.execute_phase1(predicted_vector, clinical_data)
    
    # Inspect Phase 1 results
    print("\n📊 PHASE 1 RESULTS:")
    print(json.dumps(phase1_results, indent=2))
    
    # Wait for user confirmation (optional)
    input("\nPress Enter to continue to Phase 2...")
    
    # ==========================================
    # STEP 3: EXECUTE PHASE 2 (ANALYSIS)
    # ==========================================
    
    phase2_results = agent_system.execute_phase2(
        predicted_vector,
        clinical_data,
        temporal_trends,
        spatial_insights
    )
    
    # Inspect Phase 2 results
    print("\n📊 PHASE 2 RESULTS:")
    print(json.dumps(phase2_results, indent=2))
    
    # Wait for user confirmation (optional)
    input("\nPress Enter to continue to Phase 3...")
    
    # ==========================================
    # STEP 4: EXECUTE PHASE 3 (SYNTHESIS)
    # ==========================================
    
    final_report = agent_system.execute_phase3(clinical_data)
    
    # Display final report
    print("\n" + "="*60)
    print("📄 FINAL CLINICAL REPORT")
    print("="*60)
    print(final_report)
    print("="*60)
    
    # ==========================================
    # STEP 5: SAVE RESULTS
    # ==========================================
    
    agent_system.save_all_results("patient_12345_report.json")