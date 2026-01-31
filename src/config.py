"""
config.py - Configuration and Settings for Glaucoma RAG System

Updated for:
- FAISS vector database (instead of ChromaDB)
- Gemini API models: gemini-2.5-flash and gemini-embedding-001
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================================================
# API CONFIGURATION
# ==================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file. "
        "Please create a .env file with: GEMINI_API_KEY=your_key_here"
    )

# Gemini Models (CORRECTED to match your available models)
EMBEDDING_MODEL = "models/gemini-embedding-001"  # 768 dimensions
LLM_MODEL = "models/gemini-2.5-flash"  # Fast and efficient
# Alternative: "models/gemini-2.5-pro" for better quality but slower

# ==================================================
# FILE PATHS
# ==================================================

# Data directories
GUIDELINES_DIR = "./data/guidelines"
PATIENT_DATA_PATH = "./data/patient_data/mst_predictions.json"
EXCEL_DATA_PATH = "./data/excel/patient_clinical_data.xlsx"

# Database (CHANGED from ChromaDB to FAISS)
VECTOR_DB_PATH = "./database/faiss_db"  # FAISS database path
EMBEDDING_DIMENSION = 768  # gemini-embedding-001 dimension

# Output directories
REPORTS_DIR = "./reports"
JSON_REPORTS_DIR = "./reports/json"
TEXT_REPORTS_DIR = "./reports/text"
MD_REPORTS_DIR = "./reports/markdown"

# ==================================================
# PROCESSING PARAMETERS
# ==================================================

# Text Chunking
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 80  # characters

# RAG Retrieval
TOP_K_CHUNKS = 5  # Number of chunks to retrieve per query
MAX_RETRIEVAL_CHUNKS = 20  # Maximum total chunks in prompt

# LLM Generation
LLM_TEMPERATURE = 0.3  # Lower = more factual (0.0 - 1.0)
MAX_OUTPUT_TOKENS = 4000

# Rate Limiting (for Gemini free tier)
REQUESTS_PER_MINUTE = 15  # Gemini free tier limit
DELAY_BETWEEN_REQUESTS = 4  # seconds (60/15 = 4)

# ==================================================
# CLINICAL PARAMETERS
# ==================================================

# Visual Field
VF_POINTS = 61  # Number of VF test points (standard 24-2 pattern)
NORMAL_VF_THRESHOLD = 30.0  # dB (age-matched normal)

# Severity Thresholds (AAO Guidelines - based on MD)
SEVERITY_THRESHOLDS = {
    "Normal": 0,
    "Mild": -6,
    "Moderate": -12,
    "Severe": -20,
    "Advanced": -30
}

# Progression Risk Thresholds (dB/year)
PROGRESSION_THRESHOLDS = {
    "Slow": 0.5,
    "Moderate": 1.0,
    "Rapid": 2.0
}

# Forecast Time Horizons (years)
FORECAST_HORIZONS = [0.5, 1.0, 2.0]  # 6 months, 1 year, 2 years

# Visit interval assumption (if dates not provided)
DEFAULT_VISIT_INTERVAL_MONTHS = 6

# ==================================================
# LOGGING
# ==================================================

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('glaucoma_rag.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ==================================================
# DIRECTORY CREATION
# ==================================================

def create_directories():
    """Create necessary directories if they don't exist"""
    import os
    
    directories = [
        GUIDELINES_DIR,
        os.path.dirname(PATIENT_DATA_PATH),
        os.path.dirname(EXCEL_DATA_PATH),
        VECTOR_DB_PATH,  # CHANGED from CHROMA_DB_PATH
        REPORTS_DIR,
        JSON_REPORTS_DIR,
        TEXT_REPORTS_DIR,
        MD_REPORTS_DIR
    ]
    
    for directory in directories:
        if directory:  # Skip empty strings
            os.makedirs(directory, exist_ok=True)
    
    logger.info("All directories created/verified")

# ==================================================
# VALIDATION
# ==================================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check API key
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is not set")
    
    # Check embedding model
    if EMBEDDING_MODEL != "models/gemini-embedding-001":
        errors.append(f"Warning: Embedding model is {EMBEDDING_MODEL}, expected 'models/gemini-embedding-001'")
    
    # Check LLM model
    if not LLM_MODEL.startswith("models/gemini"):
        errors.append(f"Warning: LLM model is {LLM_MODEL}, should start with 'models/gemini'")
    
    # Check embedding dimension
    if EMBEDDING_DIMENSION != 768:
        errors.append(f"Warning: Embedding dimension is {EMBEDDING_DIMENSION}, gemini-embedding-001 uses 768")
    
    if errors:
        for error in errors:
            logger.warning(error)
        return False
    
    logger.info("Configuration validated successfully")
    return True

# ==================================================
# TESTING
# ==================================================

if __name__ == "__main__":
    # Test configuration
    print("="*60)
    print("GLAUCOMA RAG SYSTEM - CONFIGURATION")
    print("="*60)
    
    print("\n📊 API Configuration:")
    print(f"  API Key present: {bool(GEMINI_API_KEY)}")
    print(f"  API Key (masked): {GEMINI_API_KEY[:10]}...{GEMINI_API_KEY[-4:] if GEMINI_API_KEY else 'N/A'}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  Embedding Dimension: {EMBEDDING_DIMENSION}")
    
    print("\n📁 File Paths:")
    print(f"  Guidelines Directory: {GUIDELINES_DIR}")
    print(f"  Patient Data: {PATIENT_DATA_PATH}")
    print(f"  Vector Database: {VECTOR_DB_PATH}")
    print(f"  Reports Directory: {REPORTS_DIR}")
    
    print("\n⚙️  Processing Parameters:")
    print(f"  Chunk Size: {CHUNK_SIZE} characters")
    print(f"  Chunk Overlap: {CHUNK_OVERLAP} characters")
    print(f"  Top K Retrieval: {TOP_K_CHUNKS}")
    print(f"  LLM Temperature: {LLM_TEMPERATURE}")
    print(f"  Max Output Tokens: {MAX_OUTPUT_TOKENS}")
    
    print("\n🏥 Clinical Parameters:")
    print(f"  VF Points: {VF_POINTS}")
    print(f"  Severity Thresholds: {list(SEVERITY_THRESHOLDS.keys())}")
    print(f"  Progression Thresholds: {list(PROGRESSION_THRESHOLDS.keys())}")
    print(f"  Forecast Horizons: {FORECAST_HORIZONS} years")
    
    print("\n" + "="*60)
    print("Creating directories...")
    print("="*60)
    create_directories()
    
    print("\n" + "="*60)
    print("Validating configuration...")
    print("="*60)
    is_valid = validate_config()
    
    if is_valid:
        print("\n Configuration loaded and validated successfully!")
    else:
        print("\n Configuration has warnings (see above)")
    
    print("="*60)