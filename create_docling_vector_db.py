#!/usr/bin/env python3
"""
Create vector database from parsed Docling texts

This script finds all docling.txt files in PDF folders and creates a vector database
using the existing RAG pipeline components (TextChunker, EmbeddingManager, VectorStore).

Usage:
    python create_docling_vector_db.py
    python create_docling_vector_db.py --pdfs-dir custom_pdfs_directory
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.text_chunker import TextChunker, TextChunk
from app.services.embedding_manager import EmbeddingManager
from app.services.vector_store import VectorStore, DocumentInfo
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DoclingVectorDBCreator:
    """Creates vector database from parsed Docling texts"""
    
    def __init__(self, pdfs_dir: str = "pdfs"):
        """
        Initialize the vector DB creator
        
        Args:
            pdfs_dir: Directory containing PDF folders with docling.txt files
        """
        self.pdfs_dir = Path(pdfs_dir)
        self.vector_store_dir = Path(settings.vector_store_dir)
        
        # Initialize pipeline components
        self.text_chunker = TextChunker()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        
        logger.info("Initialized Docling Vector DB Creator")
        logger.info(f"PDFs directory: {self.pdfs_dir}")
        logger.info(f"Vector store directory: {self.vector_store_dir}")
    
    def find_docling_files(self) -> List[Dict[str, Any]]:
        """
        Find all docling.txt files in PDF folders
        
        Returns:
            List of dictionaries with file info
        """
        docling_files = []
        
        if not self.pdfs_dir.exists():
            logger.error(f"PDFs directory not found: {self.pdfs_dir}")
            return []
        
        # Look for folders containing docling.txt files
        for item in self.pdfs_dir.iterdir():
            if item.is_dir():
                docling_file = item / "docling.txt"
                if docling_file.exists():
                    # Generate document ID from folder name
                    doc_id = hashlib.md5(item.name.encode()).hexdigest()[:16]
                    
                    # Try to determine original PDF name
                    pdf_name = item.name
                    if pdf_name.endswith('.pdf'):
                        pdf_name = pdf_name[:-4]  # Remove .pdf extension
                    
                    file_info = {
                        'doc_id': doc_id,
                        'folder_name': item.name,
                        'pdf_name': pdf_name,
                        'docling_file': docling_file,
                        'file_size': docling_file.stat().st_size
                    }
                    docling_files.append(file_info)
                    logger.debug(f"Found: {docling_file} ({file_info['file_size']:,} bytes)")
        
        logger.info(f"Found {len(docling_files)} docling.txt files")
        return docling_files
    
    def read_and_chunk_document(self, file_info: Dict[str, Any]) -> List[TextChunk]:
        """
        Read a docling.txt file and create chunks
        
        Args:
            file_info: Dictionary with file information
            
        Returns:
            List of TextChunk objects
        """
        try:
            # Read the docling.txt file
            with open(file_info['docling_file'], 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"Empty file: {file_info['docling_file']}")
                return []
            
            logger.info(f"Processing {file_info['pdf_name']} ({len(text):,} characters)")
            
            # Use the existing text chunker
            chunks = self.text_chunker.chunk_text(
                text=text,
                document_metadata={
                    'doc_id': file_info['doc_id'],
                    'source_url': f"docling://{file_info['pdf_name']}",
                    'title': file_info['pdf_name']
                }
            )
            
            logger.info(f"Created {len(chunks)} chunks for {file_info['pdf_name']}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_info['docling_file']}: {e}")
            return []
    
    def create_vector_database(self, overwrite: bool = False) -> bool:
        """
        Create vector database from all docling.txt files
        
        Args:
            overwrite: Whether to overwrite existing vector store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if vector store already exists
            has_documents = len(self.vector_store.documents) > 0
            if has_documents and not overwrite:
                logger.warning("Vector store already exists. Use --overwrite to replace it.")
                response = input("Do you want to overwrite the existing vector store? (y/N): ")
                if response.lower() != 'y':
                    logger.info("Aborted by user")
                    return False
            
            # Find all docling.txt files
            docling_files = self.find_docling_files()
            if not docling_files:
                logger.error("No docling.txt files found")
                return False
            
            # Clear existing vector store if overwriting
            if overwrite or has_documents:
                logger.info("Clearing existing vector store...")
                # Manual clear since clear() method may not exist
                self.vector_store.index = None
                self.vector_store.chunk_texts = []
                self.vector_store.chunk_metadata = []
                self.vector_store.documents = {}
            
            # Process each file
            all_chunks = []
            all_documents = []
            doc_chunks_map = {}  # Map doc_id to chunks
            total_chunks = 0
            
            print(f"\nProcessing {len(docling_files)} documents...")
            
            # Create main progress bar
            main_progress = tqdm(
                docling_files, 
                desc="Processing documents", 
                unit="doc",
                ncols=100
            )
            
            for file_info in main_progress:
                main_progress.set_postfix_str(f"Processing {file_info['pdf_name'][:30]}...")
                
                # Chunk the document
                chunks = self.read_and_chunk_document(file_info)
                if chunks:
                    all_chunks.extend(chunks)
                    doc_chunks_map[file_info['doc_id']] = chunks
                    total_chunks += len(chunks)
                    
                    # Create document info
                    doc_info = DocumentInfo(
                        doc_id=file_info['doc_id'],
                        source_url=f"docling://{file_info['pdf_name']}",
                        title=file_info['pdf_name'],
                        pages=1,  # Docling doesn't provide page info, so default to 1
                        chunk_count=len(chunks),
                        indexed_at=time.time(),
                        total_tokens=sum(chunk.token_count for chunk in chunks)
                    )
                    all_documents.append(doc_info)
                
                main_progress.set_postfix_str(f"Total chunks: {total_chunks}")
            
            main_progress.close()
            
            if not all_chunks:
                logger.error("No chunks created from any documents")
                return False
            
            logger.info(f"Total chunks created: {len(all_chunks)}")
            
            # Generate embeddings
            print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
            
            embedding_result = self.embedding_manager.encode_chunks(all_chunks)
            if embedding_result is None:
                logger.error("Failed to generate embeddings")
                return False
            
            logger.info(f"Generated {embedding_result.embeddings.shape[0]} embeddings")
            logger.info(f"Embedding dimensions: {embedding_result.embeddings.shape[1]}")
            logger.info(f"Embedding generation time: {embedding_result.processing_time:.2f}s")
            
            # Store in vector database
            print("\nSaving to vector store...")
            success = True
            
            # Add documents one by one
            chunk_start_idx = 0
            for doc_info in all_documents:
                # Get chunks for this document
                doc_chunks = doc_chunks_map[doc_info.doc_id]
                
                # Get corresponding embeddings
                doc_embeddings = embedding_result.embeddings[chunk_start_idx:chunk_start_idx + len(doc_chunks)]
                
                # Create document metadata
                doc_metadata = {
                    'source_url': doc_info.source_url,
                    'title': doc_info.title,
                    'pages': doc_info.pages,
                    'total_tokens': doc_info.total_tokens
                }
                
                # Add document to vector store
                try:
                    result = self.vector_store.add_document(
                        doc_id=doc_info.doc_id,
                        chunks=doc_chunks,
                        embeddings=doc_embeddings,
                        document_metadata=doc_metadata
                    )
                    logger.info(f"Added document {doc_info.title} with {len(doc_chunks)} chunks")
                except Exception as e:
                    logger.error(f"Failed to add document {doc_info.title}: {e}")
                    success = False
                
                chunk_start_idx += len(doc_chunks)
            
            if success:
                logger.info("Successfully created vector database!")
                logger.info(f"Documents: {len(all_documents)}")
                logger.info(f"Total chunks: {len(all_chunks)}")
                logger.info(f"Vector store location: {self.vector_store_dir}")
                
                # Show statistics
                total_chars = sum(len(chunk.text) for chunk in all_chunks)
                avg_chunk_size = total_chars / len(all_chunks) if all_chunks else 0
                
                print("\n=== Vector Database Statistics ===")
                print(f"Documents processed: {len(all_documents)}")
                print(f"Total chunks: {len(all_chunks):,}")
                print(f"Total characters: {total_chars:,}")
                print(f"Average chunk size: {avg_chunk_size:.0f} characters")
                print(f"Embedding dimensions: {embedding_result.embeddings.shape[1]}")
                print(f"Vector store directory: {self.vector_store_dir}")
                
                return True
            else:
                logger.error("Failed to save to vector store")
                return False
                
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            return False
    
    def show_summary(self):
        """Show summary of available docling.txt files"""
        docling_files = self.find_docling_files()
        
        if not docling_files:
            print("No docling.txt files found")
            return
        
        print(f"\nFound {len(docling_files)} parsed documents:")
        print("=" * 80)
        
        total_size = 0
        for i, file_info in enumerate(docling_files, 1):
            size_kb = file_info['file_size'] / 1024
            total_size += file_info['file_size']
            print(f"{i:2d}. {file_info['pdf_name'][:50]:<50} ({size_kb:6.1f} KB)")
        
        print("=" * 80)
        print(f"Total: {len(docling_files)} documents, {total_size/1024:.1f} KB")


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create vector database from parsed Docling texts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create vector DB from default pdfs/ directory
  python create_docling_vector_db.py
  
  # Show summary of available files
  python create_docling_vector_db.py --summary
  
  # Use custom PDFs directory
  python create_docling_vector_db.py --pdfs-dir /path/to/pdfs
  
  # Overwrite existing vector store
  python create_docling_vector_db.py --overwrite
        """
    )
    
    parser.add_argument('--pdfs-dir', default='pdfs',
                       help='Directory containing PDF folders with docling.txt files')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing vector store without confirmation')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary of available docling.txt files and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize creator
    try:
        creator = DoclingVectorDBCreator(pdfs_dir=args.pdfs_dir)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1
    
    # Show summary if requested
    if args.summary:
        creator.show_summary()
        return 0
    
    # Create vector database
    try:
        success = creator.create_vector_database(overwrite=args.overwrite)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())