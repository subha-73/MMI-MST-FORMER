# test_setup.py
import sys
import os

def test_imports():
    """Test all required libraries"""
    try:
        import google.generativeai as genai
        print("[OK] google-generativeai installed")
        
        import faiss
        print("[OK] faiss installed")
        
        import pypdf
        print("[OK] pypdf installed")
        
        import langchain
        print("[OK] langchain installed")
        
        import numpy
        print("[OK] numpy installed")
        
        import pandas
        print("[OK] pandas installed")
        
        from dotenv import load_dotenv
        print("[OK] python-dotenv installed")
        
        print("\n✅ All dependencies installed successfully!")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        return False

def test_api_key():
    """Test Gemini API key"""
    from dotenv import load_dotenv
    import os
    import google.generativeai as genai

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in .env file")
        return False

    try:
        genai.configure(api_key=api_key)
        # Using the ACTUAL available model from your API
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content("Say hello")
        print(f"[OK] Gemini API working! Response: {response.text[:50]}...")
        return True

    except Exception as e:
        print(f"[ERROR] Gemini API error: {e}")
        return False


def test_faiss():
    """Test FAISS"""
    import faiss
    import numpy as np
    import os
    
    try:
        # Create temporary test index
        test_path = "./database/test_faiss"
        os.makedirs(test_path, exist_ok=True)
        
        # Create a simple index
        dimension = 128
        index = faiss.IndexFlatL2(dimension)
        
        # Add some vectors
        vectors = np.random.random((10, dimension)).astype('float32')
        index.add(vectors)
        
        # Save index
        faiss.write_index(index, f"{test_path}/test.faiss")
        
        # Load and search
        loaded_index = faiss.read_index(f"{test_path}/test.faiss")
        query = np.random.random((1, dimension)).astype('float32')
        distances, indices = loaded_index.search(query, k=3)
        
        # Cleanup
        os.remove(f"{test_path}/test.faiss")
        os.rmdir(test_path)
        
        print("[OK] FAISS working! Index created, saved, loaded, and searched successfully.")
        print(f"  - Found {len(indices[0])} nearest neighbors")
        print(f"  - Distances: {distances[0][:3]}")
        return True
        
    except Exception as e:
        print(f"[ERROR] FAISS error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_gemini_embeddings():
    """Test Gemini embedding API"""
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        import os

        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        # Using the ACTUAL available embedding model from your API
        res = genai.embed_content(
            model="models/gemini-embedding-001",
            content="This is a test sentence.",
            task_type="retrieval_document"
        )

        emb = res["embedding"]
        print(f"[OK] Gemini embeddings working! Vector length: {len(emb)}")
        print(f"  - Using Gemini API (NOT sentence-transformers)")
        return True

    except Exception as e:
        print(f"[ERROR] Gemini embedding error: {e}")
        return False


def check_directory_structure():
    """Check if required directories exist"""
    required_dirs = [
        "data/guidelines",
        "data/patient_data",
        "database/faiss_db",
        "reports/json",
        "reports/text",
        "reports/markdown",
        "src"
    ]
    
    print("\nChecking directory structure...")
    all_exist = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"[OK] {dir_path} exists")
        else:
            print(f"⚠ {dir_path} does not exist - creating it...")
            os.makedirs(dir_path, exist_ok=True)
            all_exist = False
    
    if all_exist:
        print("[OK] All directories exist")
    else:
        print("[OK] Created missing directories")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("TESTING PROJECT SETUP (FAISS VERSION)")
    print("="*60 + "\n")
    
    all_passed = True
    
    print("0. Checking directory structure...")
    if not check_directory_structure():
        all_passed = False
    
    print("\n1. Testing Python dependencies...")
    if not test_imports():
        all_passed = False
    
    print("\n2. Testing Gemini API...")
    if not test_api_key():
        all_passed = False
    
    print("\n3. Testing FAISS...")
    if not test_faiss():
        all_passed = False
    
    print("\n4. Testing Gemini Embeddings (NOT sentence-transformers)...")
    if not test_gemini_embeddings():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED! You're ready to start coding.")
        print("\nNext steps:")
        print("1. Place your PDF files in data/guidelines/")
        print("2. Place your MST predictions in data/patient_data/")
        print("3. Run the implementation scripts")
    else:
        print("[ERROR] Some tests failed. Please fix errors above.")
    print("="*60)