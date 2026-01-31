"""
rag_retriever.py - RAG Query Generation and Retrieval

Updated for FAISS vector store with Gemini embeddings
"""

import time
from typing import List, Dict
from config import TOP_K_CHUNKS, MAX_RETRIEVAL_CHUNKS, DELAY_BETWEEN_REQUESTS, logger


class RAGRetriever:
    """Generate queries and retrieve relevant guideline chunks using FAISS"""
    
    def __init__(self, vector_store):
        """
        Initialize RAG retriever
        
        Args:
            vector_store: FAISSVectorStore instance
        """
        self.vector_store = vector_store
    
    def generate_queries(
        self,
        current_metrics: Dict,
        progression_analysis: Dict,
        forecast: Dict
    ) -> List[str]:
        """
        Generate targeted RAG queries based on patient data
        
        Args:
            current_metrics: Latest visit clinical metrics
            progression_analysis: Multi-visit progression analysis
            forecast: Future progression forecast
            
        Returns:
            List of query strings
        """
        MD = current_metrics["MD"]
        severity = current_metrics["severity"]
        rate = progression_analysis.get("progression_rate_MD", 0)
        
        # REDUCED to 2 queries to save API quota
        queries = [
            f"Treatment guidelines {severity} glaucoma progression rate {abs(rate)} dB per year",
            f"Target intraocular pressure monitoring {severity} glaucoma management"
        ]
        
        logger.info(f"Generated {len(queries)} RAG queries (reduced for quota)")
        return queries
    
    def retrieve_for_queries(
        self,
        queries: List[str],
        top_k_per_query: int = TOP_K_CHUNKS
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for multiple queries using FAISS
        
        Args:
            queries: List of query strings
            top_k_per_query: Number of chunks to retrieve per query
            
        Returns:
            List of retrieved chunk dictionaries
        """
        logger.info(f"Retrieving guidelines for {len(queries)} queries...")
        
        all_retrieved = []
        
        for i, query in enumerate(queries):
            try:
                # Query FAISS vector database
                # Returns list of dicts with 'text', 'metadata', 'distance', 'similarity'
                results = self.vector_store.search(query, k=top_k_per_query)
                
                # Process results
                for j, result in enumerate(results):
                    all_retrieved.append({
                        "query": query,
                        "text": result["text"],
                        "source": result["metadata"].get("source", "Unknown"),
                        "chunk_id": result["metadata"].get("chunk_id", f"chunk_{j}"),
                        "distance": result.get("distance", 0),
                        "similarity": result.get("similarity", 0),
                        "relevance_rank": j + 1,
                        "query_index": i
                    })
                
                # Progress logging
                if (i + 1) % 5 == 0:
                    logger.info(f"Progress: {i + 1}/{len(queries)} queries processed")
                
                # Rate limiting (Gemini API has limits on embeddings)
                if i < len(queries) - 1:
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    
            except Exception as e:
                logger.error(f"Error retrieving for query '{query}': {e}")
                continue
        
        logger.info(f"Retrieved {len(all_retrieved)} total chunks")
        
        # Deduplicate
        unique_retrieved = self._deduplicate_chunks(all_retrieved)
        
        # Limit to max chunks
        if len(unique_retrieved) > MAX_RETRIEVAL_CHUNKS:
            unique_retrieved = unique_retrieved[:MAX_RETRIEVAL_CHUNKS]
            logger.info(f"Limited to {MAX_RETRIEVAL_CHUNKS} chunks")
        
        return unique_retrieved
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Remove duplicate chunks, keeping highest relevance (lowest distance)
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Deduplicated list
        """
        seen_texts = {}
        
        for chunk in chunks:
            text = chunk["text"]
            if text not in seen_texts:
                seen_texts[text] = chunk
            else:
                # Keep chunk with better relevance (lower distance = more similar)
                if chunk["distance"] < seen_texts[text]["distance"]:
                    seen_texts[text] = chunk
        
        unique = list(seen_texts.values())
        logger.info(f"Deduplicated: {len(chunks)} -> {len(unique)} chunks")
        
        return unique
    
    def organize_by_source(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Organize retrieved chunks by source guideline
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Dictionary mapping source -> chunks
        """
        by_source = {}
        
        for chunk in chunks:
            source = chunk["source"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk)
        
        return by_source
    
    def format_for_prompt(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks for inclusion in LLM prompt
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted string for prompt
        """
        # Organize by source
        by_source = self.organize_by_source(chunks)
        
        formatted_text = ""
        
        for source, source_chunks in by_source.items():
            formatted_text += f"\n{'='*60}\n"
            formatted_text += f"SOURCE: {source}\n"
            formatted_text += f"{'='*60}\n\n"
            
            for i, chunk in enumerate(source_chunks, 1):
                formatted_text += f"[{source} - Excerpt {i}]\n"
                formatted_text += chunk["text"]
                formatted_text += "\n\n"
        
        return formatted_text
    
    def get_citation_map(self, chunks: List[Dict]) -> Dict[str, int]:
        """
        Create citation map for tracking sources
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Dictionary mapping source -> count
        """
        citation_map = {}
        
        for chunk in chunks:
            source = chunk["source"]
            citation_map[source] = citation_map.get(source, 0) + 1
        
        return citation_map
    
    def get_retrieval_summary(self, chunks: List[Dict]) -> Dict:
        """
        Generate summary statistics about retrieved chunks
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Summary statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "unique_sources": 0,
                "avg_similarity": 0,
                "sources": {}
            }
        
        citation_map = self.get_citation_map(chunks)
        similarities = [c.get("similarity", 0) for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "unique_sources": len(citation_map),
            "avg_similarity": round(sum(similarities) / len(similarities), 3) if similarities else 0,
            "sources": citation_map,
            "queries_used": len(set(c["query"] for c in chunks))
        }


def main():
    """Test RAG retrieval with FAISS"""
    from vector_store import FAISSVectorStore
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Initialize vector store
    print("\n" + "="*60)
    print("RAG RETRIEVAL TEST (FAISS)")
    print("="*60)
    
    vector_store = FAISSVectorStore(api_key=api_key)
    
    # Check if database has documents
    stats = vector_store.get_stats()
    if stats["total_documents"] == 0:
        print("\n[ERROR] No documents in database.")
        print("Please run the following first:")
        print("  1. Add PDFs to data/guidelines/")
        print("  2. Run: python build_vector_db.py")
        return
    
    print(f"\n[OK] Vector store loaded: {stats['total_documents']} documents")
    
    # Test data
    test_metrics = {
        "MD": -10.2,
        "VFI": 65.3,
        "severity": "Moderate",
        "pattern": "superior defect"
    }
    
    test_progression = {
        "progression_rate_MD": -1.0,
        "risk_level": "Rapid"
    }
    
    test_forecast = {
        "forecasts": [
            {"time_horizon_months": 12, "predicted_MD": -11.2}
        ]
    }
    
    # Initialize retriever
    retriever = RAGRetriever(vector_store)
    
    # Generate queries
    print("\n" + "="*60)
    print("GENERATING QUERIES")
    print("="*60)
    
    queries = retriever.generate_queries(test_metrics, test_progression, test_forecast)
    print(f"\nGenerated {len(queries)} queries")
    print("\nExample queries:")
    for i, q in enumerate(queries[:5], 1):
        print(f"  {i}. {q}")
    
    # Retrieve chunks
    print("\n" + "="*60)
    print("RETRIEVING RELEVANT GUIDELINES")
    print("="*60)
    
    print("\nSearching vector database...")
    chunks = retriever.retrieve_for_queries(queries[:5], top_k_per_query=3)  # Limit for test
    
    # Show results
    summary = retriever.get_retrieval_summary(chunks)
    print(f"\n[OK] Retrieved {summary['total_chunks']} unique chunks")
    print(f"  Unique sources: {summary['unique_sources']}")
    print(f"  Average similarity: {summary['avg_similarity']}")
    
    # Show citation map
    print(f"\nChunks by source:")
    for source, count in summary['sources'].items():
        print(f"  {source}: {count} chunks")
    
    # Show example chunk
    if chunks:
        print(f"\n" + "="*60)
        print("EXAMPLE RETRIEVED CHUNK")
        print("="*60)
        example = chunks[0]
        print(f"Source: {example['source']}")
        print(f"Similarity: {example['similarity']:.3f}")
        print(f"Text preview: {example['text'][:200]}...")
    
    # Show formatted output
    formatted = retriever.format_for_prompt(chunks[:3])
    print(f"\n" + "="*60)
    print("FORMATTED FOR PROMPT (first 3 chunks)")
    print("="*60)
    print(formatted[:500] + "...")
    
    print("\n[OK] RAG retrieval test complete!")


if __name__ == "__main__":
    main()