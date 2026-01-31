"""
main.py - Main Orchestration Script

This script coordinates the entire RAG pipeline:
1. Load patient data
2. Calculate clinical metrics
3. Analyze progression
4. Generate forecasts
5. Retrieve relevant guidelines
6. Generate clinical reports
"""

import json
import time
from typing import Dict, List

# Import all modules
from config import PATIENT_DATA_PATH, logger, create_directories
from pdf_processor import PDFProcessor
from vector_store import FAISSVectorStore
from clinical_metrics import process_all_visits
from progression_analysis import ProgressionAnalyzer
from forecasting import ProgressionForecaster
from rag_retriever import RAGRetriever
from prompt_builder import PromptBuilder
from report_generator import ReportGenerator
from excel_loader import ExcelDataLoader  # Excel integration


class GlaucomaRAGPipeline:
    """Complete RAG pipeline for glaucoma progression reporting"""
    
    def __init__(self, rebuild_database: bool = False):
        """
        Initialize pipeline
        
        Args:
            rebuild_database: If True, rebuild vector database from PDFs
        """
        logger.info("="*70)
        logger.info("INITIALIZING GLAUCOMA RAG PIPELINE")
        logger.info("="*70)
        
        # Create directories
        create_directories()
        
        # Initialize components
        self.vector_store = FAISSVectorStore()
        self.progression_analyzer = ProgressionAnalyzer()
        self.forecaster = ProgressionForecaster()
        self.prompt_builder = PromptBuilder()
        self.report_generator = ReportGenerator()
        self.excel_loader = ExcelDataLoader()  # Excel data loader
        
        # Build or load vector database
        if rebuild_database:
            self._build_vector_database()
        else:
            # Just connect to existing database
            stats = self.vector_store.get_stats()
            if stats["total_documents"] == 0:
                logger.warning("Vector database is empty. Building from PDFs...")
                self._build_vector_database()
            else:
                logger.info(f"Vector database loaded: {stats['total_documents']} documents")
        
        self.rag_retriever = RAGRetriever(self.vector_store)
        
        logger.info("[OK] Pipeline initialized successfully")
    
    def _build_vector_database(self):
        """Build vector database from PDF guidelines"""
        logger.info("Building vector database from PDFs...")
        
        # Process PDFs
        pdf_processor = PDFProcessor()
        chunks = pdf_processor.process_all_pdfs()
        
        if not chunks:
            raise ValueError("No PDF files found or processing failed. "
                           "Please add PDF guidelines to data/guidelines/")
        
        # Build database
        self.vector_store = FAISSVectorStore()

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk.get("metadata", {}) for chunk in chunks]

        self.vector_store.add_documents(texts, metadatas)
        self.vector_store.save()
        logger.info("[OK] Vector database built successfully")
    
    def load_patient_data(self, json_path: str = PATIENT_DATA_PATH) -> List[Dict]:
        """Load patient data from JSON file"""
        logger.info(f"Loading patient data from: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Group visits by patient_id
            patients_dict = {}
            
            if isinstance(data, list):
                for entry in data:
                    patient_id = entry.get('patient_id')
                    if not patient_id:
                        continue
                    
                    # Initialize patient if not exists
                    if patient_id not in patients_dict:
                        patients_dict[patient_id] = {
                            'patient_id': patient_id,
                            'visits': []
                        }
                    
                    # Add this visit to the patient's visits list
                    patients_dict[patient_id]['visits'].append(entry)
            
            elif isinstance(data, dict):
                # Single patient with visits already grouped
                if 'visits' in data:
                    patients = [data]
                else:
                    # Single visit
                    patients = [{
                        'patient_id': data.get('patient_id', 'UNKNOWN'),
                        'visits': [data]
                    }]
                logger.info(f"[OK] Loaded {len(patients)} patients")
                return patients
            else:
                raise ValueError("Invalid JSON format")
            
            # Convert dict to list
            patients = list(patients_dict.values())
            
            # Sort visits within each patient
            for patient in patients:
                patient['visits'].sort(key=lambda x: x.get('visit', 0))
            
            logger.info(f"[OK] Loaded {len(patients)} patients")
            for patient in patients:
                logger.info(f"  {patient['patient_id']}: {len(patient['visits'])} visits")
            
            return patients
            
        except FileNotFoundError:
            logger.error(f"Patient data file not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
    
    def process_single_patient(self, patient_data: Dict) -> Dict:
        """
        Process a single patient through the complete pipeline
        
        Args:
            patient_data: Patient dictionary with visits
            
        Returns:
            Complete analysis with generated report
        """
        patient_id = patient_data["patient_id"]
        logger.info("="*70)
        logger.info(f"PROCESSING PATIENT: {patient_id}")
        logger.info("="*70)
        
        # Step 0: Enrich with Excel data (if available)
        logger.info("Step 0: Loading additional clinical data from Excel...")
        patient_data = self.excel_loader.enrich_patient_data(patient_data)
        
        # Step 1: Calculate clinical metrics for all visits
        logger.info("Step 1: Calculating clinical metrics...")
        patient_data = process_all_visits(patient_data)
        
        # Step 2: Analyze progression across visits
        logger.info("Step 2: Analyzing progression...")
        progression_analysis = self.progression_analyzer.analyze_progression(
            patient_data["visits"]
        )
        
        if progression_analysis["status"] != "analyzed":
            logger.warning(f"Cannot analyze progression: {progression_analysis.get('message', 'Unknown')}")
            return None
        
        # Step 3: Generate forecast
        logger.info("Step 3: Generating forecast...")
        latest_visit = patient_data["visits"][-1]
        forecast = self.forecaster.forecast_linear(
            current_metrics=latest_visit["clinical_metrics"],
            progression_rate=progression_analysis["progression_rate_MD"]
        )
        
        # Step 4: Generate RAG queries and retrieve guidelines
        logger.info("Step 4: Retrieving relevant guidelines...")
        queries = self.rag_retriever.generate_queries(
            current_metrics=latest_visit["clinical_metrics"],
            progression_analysis=progression_analysis,
            forecast=forecast
        )
        
        retrieved_chunks = self.rag_retriever.retrieve_for_queries(queries)
        retrieved_chunks_formatted = self.rag_retriever.format_for_prompt(retrieved_chunks)
        
        # Step 5: Build comprehensive prompt
        logger.info("Step 5: Building prompt...")
        prompt = self.prompt_builder.build_comprehensive_prompt(
            patient_data=patient_data,
            progression_analysis=progression_analysis,
            forecast=forecast,
            retrieved_chunks=retrieved_chunks,
            retrieved_chunks_formatted=retrieved_chunks_formatted
        )
        
        # Step 6: Generate clinical report
        logger.info("Step 6: Generating clinical report...")
        narrative_report = self.report_generator.generate_report(
            prompt=prompt,
            system_instruction=self.prompt_builder.system_instruction
        )
        
        # Step 7: Create structured output
        logger.info("Step 7: Creating structured output...")
        structured_output = self.report_generator.create_structured_output(
            narrative_report=narrative_report,
            patient_data=patient_data,
            progression_analysis=progression_analysis,
            forecast=forecast,
            retrieved_chunks=retrieved_chunks
        )
        
        # Step 8: Save reports
        logger.info("Step 8: Saving reports...")
        saved_files = self.report_generator.save_report(
            structured_output=structured_output,
            patient_id=patient_id,
            output_formats=["txt"]
            # output_formats=["json","txt","md"]
        )
        
        logger.info("="*70)
        logger.info(f"[OK] PATIENT {patient_id} PROCESSING COMPLETE")
        logger.info(f"Report files:")
        for fmt, path in saved_files.items():
            logger.info(f"  {fmt.upper()}: {path}")
        logger.info("="*70)
        
        return {
            "patient_id": patient_id,
            "status": "success",
            "files": saved_files,
            "structured_output": structured_output
        }
    
    def process_all_patients(
        self,
        patients: List[Dict],
        delay_between_patients: int = 5
    ) -> List[Dict]:
        """
        Process multiple patients with rate limiting
        
        Args:
            patients: List of patient dictionaries
            delay_between_patients: Seconds to wait between patients
            
        Returns:
            List of processing results
        """
        results = []
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH PROCESSING: {len(patients)} PATIENTS")
        logger.info(f"{'='*70}\n")
        
        for i, patient_data in enumerate(patients, 1):
            patient_id = patient_data.get("patient_id", f"UNKNOWN_{i}")
            
            logger.info(f"\n[{i}/{len(patients)}] Processing {patient_id}...")
            
            try:
                result = self.process_single_patient(patient_data)
                results.append(result)
                
                # Rate limiting between patients
                if i < len(patients):
                    logger.info(f"Waiting {delay_between_patients}s before next patient...")
                    time.sleep(delay_between_patients)
                    
            except Exception as e:
                logger.error(f"Error processing {patient_id}: {e}")
                results.append({
                    "patient_id": patient_id,
                    "status": "failed",
                    "error": str(e)
                })
                continue
        
        # Summary
        successful = sum(1 for r in results if r and r["status"] == "success")
        failed = len(results) - successful
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"{'='*70}\n")
        
        return results


def main():
    """Main entry point"""
    
    # Configuration
    REBUILD_DATABASE = False  # Set to True to rebuild from PDFs
    PATIENT_DATA_JSON = PATIENT_DATA_PATH
    
    try:
        # Initialize pipeline
        pipeline = GlaucomaRAGPipeline(rebuild_database=REBUILD_DATABASE)
        
        # Load patient data
        patients = pipeline.load_patient_data(PATIENT_DATA_JSON)
        
        # Process patients
        if len(patients) == 1:
            # Single patient
            result = pipeline.process_single_patient(patients[0])
        else:
            # Multiple patients
            results = pipeline.process_all_patients(patients, delay_between_patients=5)
        
        logger.info("\n Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()