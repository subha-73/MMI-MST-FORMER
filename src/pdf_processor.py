"""
pdf_processor.py - PDF Extraction and Chunking

This module is compatible with both ChromaDB and FAISS vector stores.
It processes PDF guideline documents and creates text chunks for embedding.
"""

import os
import re
from typing import List, Dict, Tuple
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import GUIDELINES_DIR, CHUNK_SIZE, CHUNK_OVERLAP, logger


class PDFProcessor:
    """Process PDF guideline documents into text chunks"""
    
    def __init__(self, guidelines_dir: str = GUIDELINES_DIR):
        """
        Initialize PDF processor
        
        Args:
            guidelines_dir: Directory containing PDF files
        """
        self.guidelines_dir = guidelines_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        logger.info(f"PDFProcessor initialized with directory: {guidelines_dir}")
        logger.info(f"Chunk settings: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract all text from a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        logger.info(f"Extracting text from: {os.path.basename(pdf_path)}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:  # Only add if text was extracted
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
            
            # Clean extracted text
            text = self.clean_text(text)
            
            metadata = {
                "source": os.path.basename(pdf_path).replace('.pdf', ''),
                "num_pages": len(reader.pages),
                "file_path": pdf_path,
                "file_name": os.path.basename(pdf_path)
            }
            
            logger.info(f"[OK] Extracted {len(text)} characters from {metadata['num_pages']} pages")
            return text, metadata
            
        except Exception as e:
            logger.error(f"[ERROR] Error extracting PDF {pdf_path}: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers (optional - comment out if you want to keep them)
        # text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Remove common PDF artifacts
        text = text.replace('\x00', '')  # Null characters
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def chunk_text(self, text: str, source_name: str) -> List[Dict]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Full text to chunk
            source_name: Name of source document
            
        Returns:
            List of chunk dictionaries
        """
        logger.info(f"Chunking text from {source_name}")
        
        chunks = self.text_splitter.split_text(text)
        
        chunked_data = []
        for idx, chunk in enumerate(chunks):
            chunked_data.append({
                "chunk_id": f"{source_name}_chunk_{idx}",
                "text": chunk,
                "source": source_name,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            })
        
        logger.info(f"[OK] Created {len(chunks)} chunks from {source_name}")
        return chunked_data
    
    def process_all_pdfs(self) -> List[Dict]:
        """
        Process all PDF files in guidelines directory
        
        Returns:
            List of all chunks from all PDFs
        """
        all_chunks = []
        
        # Check if directory exists
        if not os.path.exists(self.guidelines_dir):
            logger.error(f"[ERROR] Guidelines directory not found: {self.guidelines_dir}")
            return []
        
        # Find all PDF files
        pdf_files = [f for f in os.listdir(self.guidelines_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"⚠️  No PDF files found in {self.guidelines_dir}")
            logger.info("Please add PDF files to the guidelines directory")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        logger.info(f"Files: {', '.join(pdf_files)}")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.guidelines_dir, pdf_file)
            
            try:
                # Extract text
                text, metadata = self.extract_text_from_pdf(pdf_path)
                
                # Chunk text
                chunks = self.chunk_text(text, metadata["source"])
                
                # Add file-level metadata to each chunk
                for chunk in chunks:
                    chunk["metadata"] = metadata
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to process {pdf_file}: {e}")
                continue
        
        logger.info(f"[OK] Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def get_texts_and_metadatas(self, chunks: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """
        Extract texts and metadatas for vector store
        
        This format works with both ChromaDB and FAISS vector stores
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (texts, metadatas) for vector store
        """
        texts = []
        metadatas = []
        
        for chunk in chunks:
            texts.append(chunk["text"])
            
            # Create metadata for vector store
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "chunk_size": chunk["chunk_size"],
                "file_name": chunk["metadata"]["file_name"]
            }
            metadatas.append(metadata)
        
        return texts, metadatas


def main():
    """Test PDF processing"""
    print("="*60)
    print("PDF PROCESSOR TEST")
    print("="*60)
    
    processor = PDFProcessor()
    chunks = processor.process_all_pdfs()
    
    if chunks:
        print(f"\n[OK] Processed {len(chunks)} total chunks")
        
        # Show statistics
        sources = {}
        for chunk in chunks:
            source = chunk["source"]
            sources[source] = sources.get(source, 0) + 1
        
        print(f"\nChunks per source:")
        for source, count in sources.items():
            print(f"  {source}: {count} chunks")
        
        # Show example chunk
        print(f"\n" + "="*60)
        print("EXAMPLE CHUNK")
        print("="*60)
        print(f"ID: {chunks[0]['chunk_id']}")
        print(f"Source: {chunks[0]['source']}")
        print(f"Index: {chunks[0]['chunk_index']} / {chunks[0]['total_chunks']}")
        print(f"Size: {chunks[0]['chunk_size']} characters")
        print(f"\nText preview:")
        print(chunks[0]['text'][:300] + "...")
        
        # Test extraction for vector store
        print(f"\n" + "="*60)
        print("VECTOR STORE FORMAT TEST")
        print("="*60)
        texts, metadatas = processor.get_texts_and_metadatas(chunks)
        print(f"Texts extracted: {len(texts)}")
        print(f"Metadatas extracted: {len(metadatas)}")
        print(f"\nExample metadata:")
        print(metadatas[0])
        
    else:
        print("\n[ERROR] No PDFs found or processing failed")
        print(f"Please add PDF files to: {processor.guidelines_dir}")


if __name__ == "__main__":
    main()