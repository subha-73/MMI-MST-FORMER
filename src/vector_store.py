"""
vector_store.py - FAISS Vector Database with Gemini Embeddings

This module handles:
1. Loading and initializing FAISS index
2. Generating embeddings using Gemini API (models/gemini-embedding-001)
3. Adding documents to the vector store
4. Searching for similar documents
5. Saving/loading the index from disk

Gemini Models Used:
- Embedding: models/gemini-embedding-001 (768 dimensions)
- Text Generation: models/gemini-2.5-flash (for report generation elsewhere)
"""

import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from typing import List, Dict, Optional
from tqdm import tqdm
import time


class FAISSVectorStore:
    """
    FAISS-based vector store with Gemini embeddings for semantic search
    
    Key Features:
    - Uses Gemini API (models/gemini-embedding-001) for 768-dim embeddings
    - FAISS IndexFlatL2 for exact nearest neighbor search
    - Persistent storage with save/load functionality
    - Separate task types for documents vs queries
    - Rate limiting to respect API quotas (15 req/min)
    """
    
    def __init__(
        self,
        db_path: str = "./database/faiss_db",
        api_key: str = None,
        embedding_model: str = "models/gemini-embedding-001",  # CONFIRMED from your API
        dimension: int = 3072  # gemini-embedding-001 produces 3072-dimensional vectors
    ):
        """
        Initialize FAISS vector store with Gemini embeddings
        
        Args:
            db_path: Path to save FAISS database files
            api_key: Gemini API key (if None, must be configured externally)
            embedding_model: Gemini embedding model (default: models/gemini-embedding-001)
            dimension: Dimension of embeddings (768 for gemini-embedding-001)
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.dimension = dimension
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # File paths
        self.index_path = os.path.join(db_path, "index.faiss")
        self.documents_path = os.path.join(db_path, "documents.pkl")
        
        # Configure Gemini API
        if api_key:
            genai.configure(api_key=api_key)
        
        print(f"Initializing FAISS Vector Store")
        print(f"  Embedding model: {embedding_model}")
        print(f"  Dimension: {dimension}")
        print(f"  Database path: {db_path}")
        
        # Initialize or load FAISS index
        self.index = None
        self.documents = []  # List of document dictionaries
        
        # Load existing index if available
        if self._index_exists():
            self.load()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        print("Creating new FAISS index...")

        # Auto-detect dimension using a sample embedding
        sample = genai.embed_content(
            model=self.embedding_model_name,
            content="dimension check",
            task_type="retrieval_document"
        )
        self.dimension = len(sample["embedding"])

        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []

        print(f"[OK] New index created with dimension {self.dimension}")

    
    def _index_exists(self) -> bool:
        """Check if saved index exists"""
        return (
            os.path.exists(self.index_path) and
            os.path.exists(self.documents_path)
        )
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 10,
        task_type: str = "retrieval_document",
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings using Gemini API
        
        Args:
            texts: List of text strings
            batch_size: Number of texts per batch
            task_type: Type of embedding task
                - "retrieval_document" for indexing documents
                - "retrieval_query" for search queries
                - "semantic_similarity" for comparing texts
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (shape: [len(texts), dimension])
        """
        if not texts:
            return np.array([])
        
        print(f"Generating embeddings for {len(texts)} texts...")
        print(f"  Task type: {task_type}")
        
        embeddings = []
        
        # Process texts one by one to respect API rate limits
        # Gemini free tier: 15 requests/minute = 1 request every 4 seconds safe
        iterator = range(len(texts))
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
        
        for i in iterator:
            text = texts[i]
            
            try:
                # Generate embedding using Gemini API
                result = genai.embed_content(
                    model=self.embedding_model_name,
                    content=text,
                    task_type=task_type
                )
                embeddings.append(result["embedding"])
                
                # Rate limiting: 4 seconds between requests = safe for 15/min limit
                if i < len(texts) - 1:  # Don't sleep after last request
                    time.sleep(2.0)
                
            except Exception as e:
                print(f"\n⚠ Error embedding text (length {len(text)}): {str(e)[:100]}")
                # Add zero vector as placeholder for failed embeddings
                embeddings.append([0.0] * self.dimension)
        
        embeddings_array = np.array(embeddings, dtype='float32')
        print(f"[OK] Generated embeddings with shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict] = None,
        batch_size: int = 10
    ):
        """
        Add documents to the vector store
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dictionaries (optional)
            batch_size: Not used with Gemini API (kept for compatibility)
        """
        if not texts:
            print("No documents to add.")
            return
        
        print(f"\nAdding {len(texts)} documents to vector store...")
        
        # Generate embeddings with document task type
        embeddings = self.generate_embeddings(
            texts, 
            task_type="retrieval_document"
        )
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents with metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        for text, metadata in zip(texts, metadatas):
            doc = {
                "text": text,
                "metadata": metadata,
                "id": len(self.documents)
            }
            self.documents.append(doc)
        
        print(f"[OK] Added {len(texts)} documents to vector store")
        print(f"  Total documents in store: {len(self.documents)}")
        print(f"  Total vectors in index: {self.index.ntotal}")
    
    def search(
        self,
        query: str,
        k: int = 5,
        return_distances: bool = True
    ) -> List[Dict]:
        """
        Search for similar documents using semantic similarity
        
        Args:
            query: Query text
            k: Number of results to return
            return_distances: Whether to include distances in results
            
        Returns:
            List of dictionaries with 'text', 'metadata', 'distance', 'similarity'
        """
        if self.index.ntotal == 0:
            print("⚠ Warning: Index is empty. No documents to search.")
            return []
        
        # Generate query embedding (use retrieval_query task type)
        query_embedding = self.generate_embeddings(
            [query], 
            task_type="retrieval_query",
            show_progress=False
        )
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.documents):  # Valid index
                result = {
                    "text": self.documents[idx]["text"],
                    "metadata": self.documents[idx]["metadata"],
                }
                
                if return_distances:
                    result["distance"] = float(distance)
                    # Convert L2 distance to similarity score (0-1 range)
                    result["similarity"] = 1.0 / (1.0 + distance)
                
                results.append(result)
        
        return results
    
    def save(self):
        """Save FAISS index and documents to disk"""
        print(f"\nSaving FAISS vector store to {self.db_path}...")
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        print(f"  [OK] Saved index: {self.index_path}")
        
        # Save documents metadata
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"  [OK] Saved documents: {self.documents_path}")
        
        print(f"[OK] Saved {self.index.ntotal} vectors and {len(self.documents)} documents")
    
    def load(self):
        """Load FAISS index and documents from disk"""
        print(f"\nLoading FAISS vector store from {self.db_path}...")
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            print(f"  [OK] Loaded index: {self.index_path}")
            
            # Load documents metadata
            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"  [OK] Loaded documents: {self.documents_path}")
            
            print(f"[OK] Loaded {self.index.ntotal} vectors and {len(self.documents)} documents")
            
        except Exception as e:
            print(f"⚠ Error loading vector store: {e}")
            print("  Creating new index instead...")
            self._create_new_index()
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "embedding_model": self.embedding_model_name,
            "index_path": self.index_path,
            "index_type": "IndexFlatL2 (exact search)"
        }
    
    def clear(self):
        """Clear the vector store (removes all documents and vectors)"""
        self._create_new_index()
        print("[OK] Vector store cleared")


# Example usage and testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    print("="*60)
    print("FAISS VECTOR STORE TEST")
    print("="*60)
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("[ERROR] Error: GEMINI_API_KEY not found in .env file")
        exit(1)
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Initialize vector store
    print("\n" + "="*60)
    print("Initializing Vector Store")
    print("="*60)
    vector_store = FAISSVectorStore(api_key=api_key)
    
    # Sample medical texts
    sample_docs = [
        "Glaucoma is characterized by progressive optic nerve damage and visual field loss.",
        "Intraocular pressure (IOP) is the primary modifiable risk factor for glaucoma.",
        "Visual field testing is essential for monitoring glaucoma progression.",
        "RNFL thickness measurement helps assess structural damage in glaucoma.",
        "Medical therapy for glaucoma typically involves topical medications to lower IOP."
    ]
    
    sample_metadata = [
        {"source": "AAO Guidelines", "section": "Definition"},
        {"source": "AAO Guidelines", "section": "Risk Factors"},
        {"source": "WHO Guidelines", "section": "Diagnosis"},
        {"source": "WHO Guidelines", "section": "Structural Assessment"},
        {"source": "Indian Guidelines", "section": "Treatment"}
    ]
    
    # Add documents
    print("\n" + "="*60)
    print("Adding Documents")
    print("="*60)
    vector_store.add_documents(sample_docs, sample_metadata)
    
    # Save to disk
    vector_store.save()
    
    # Test search
    print("\n" + "="*60)
    print("Testing Search")
    print("="*60)
    
    queries = [
        "How to measure glaucoma progression?",
        "What causes high eye pressure?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = vector_store.search(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"  Text: {result['text'][:80]}...")
            print(f"  Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  Similarity: {result['similarity']:.4f}")
    
    # Display statistics
    print("\n" + "="*60)
    print("Vector Store Statistics")
    print("="*60)
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ TEST COMPLETED")