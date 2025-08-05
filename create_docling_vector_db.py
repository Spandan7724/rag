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
from typing import List, Dict, Any
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
        self.pdfs_dir = Path(pdfs_dir)
        self.vector_store_dir = Path(settings.vector_store_dir)

        self.text_chunker = TextChunker()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()

        logger.info(f"Initialized Docling Vector DB Creator")
        logger.info(f"PDFs directory: {self.pdfs_dir}")
        logger.info(f"Vector store directory: {self.vector_store_dir}")

    def find_docling_files(self) -> List[Dict[str, Any]]:
        docling_files: List[Dict[str, Any]] = []
        if not self.pdfs_dir.exists():
            logger.error(f"PDFs directory not found: {self.pdfs_dir}")
            return []

        for item in self.pdfs_dir.iterdir():
            if item.is_dir():
                docling_file = item / "docling.txt"
                if docling_file.exists():
                    doc_id = hashlib.md5(item.name.encode()).hexdigest()[:16]
                    pdf_name = item.name
                    if pdf_name.endswith('.pdf'):
                        pdf_name = pdf_name[:-4]
                    docling_files.append({
                        'doc_id': doc_id,
                        'folder_name': item.name,
                        'pdf_name': pdf_name,
                        'docling_file': docling_file,
                        'file_size': docling_file.stat().st_size
                    })
        logger.info(f"Found {len(docling_files)} docling.txt files")
        return docling_files

    def read_and_chunk_document(self, file_info: Dict[str, Any]) -> List[TextChunk]:
        try:
            text = file_info['docling_file'].read_text(encoding='utf-8')
            if not text.strip():
                logger.warning(f"Empty file: {file_info['docling_file']}")
                return []

            logger.info(f"Processing {file_info['pdf_name']} ({len(text):,} chars)")
            document_metadata = {
                'doc_id': file_info['doc_id'],
                'source_url': f"docling://{file_info['pdf_name']}"
            }

            # chunk_text signature: (text: str, document_metadata: dict)
            chunks = self.text_chunker.chunk_text(text, document_metadata)

            # ensure metadata dict exists and update
            for c in chunks:
                if not hasattr(c, 'metadata') or c.metadata is None:
                    c.metadata = {}
                c.metadata.update(document_metadata)

            logger.info(f"Created {len(chunks)} chunks for {file_info['pdf_name']}")
            return chunks

        except Exception as e:
            logger.error(f"Error processing {file_info['docling_file']}: {e}")
            return []

    def create_vector_database(self, overwrite: bool = False) -> bool:
        try:
            if len(self.vector_store.documents) > 0 and not overwrite:
                logger.warning("Vector store exists; use --overwrite to replace")
                if input("Overwrite? (y/N): ").lower() != 'y':
                    return False

            files = self.find_docling_files()
            if not files:
                return False

            if overwrite:
                logger.info("Clearing existing vector store...")
                self.vector_store.index = None
                self.vector_store.chunk_texts = []
                self.vector_store.chunk_metadata = []
                self.vector_store.documents = {}

            all_chunks: List[TextChunk] = []
            all_docs: List[DocumentInfo] = []
            total_chunks = 0

            print(f"\nProcessing {len(files)} documents…")
            progress = tqdm(files, desc="Documents", unit="doc", ncols=100)
            for info in progress:
                progress.set_postfix_str(info['pdf_name'][:30])
                chunks = self.read_and_chunk_document(info)
                if not chunks:
                    continue

                all_chunks.extend(chunks)
                total_chunks += len(chunks)

                # compute total tokens for this document
                total_tokens = sum(
                    c.metadata.get('token_count', len(c.text.split()))
                    for c in chunks
                )

                doc_info = DocumentInfo(
                    doc_id=info['doc_id'],
                    source_url=f"docling://{info['pdf_name']}",
                    title=info['pdf_name'],
                    pages=1,
                    chunk_count=len(chunks),
                    indexed_at=time.time(),
                    total_tokens=total_tokens
                )
                all_docs.append(doc_info)
                progress.set_postfix_str(f"Chunks: {total_chunks}")
            progress.close()

            if not all_chunks:
                logger.error("No chunks at all")
                return False

            logger.info(f"Total chunks: {len(all_chunks)}")
            print(f"\nGenerating embeddings…")
            texts = [c.text for c in all_chunks]
            emb_result = self.embedding_manager.generate_embeddings(texts)
            if emb_result is None:
                return False

            print(f"\nSaving to vector store…")
            success = self.vector_store.add_documents(
                chunks=all_chunks,
                embeddings=emb_result.embeddings,
                documents=all_docs
            )

            if success:
                print("\n=== Stats ===")
                total_chars = sum(len(c.text) for c in all_chunks)
                avg_size = total_chars / len(all_chunks)
                print(f"Documents: {len(all_docs)}")
                print(f"Chunks: {len(all_chunks):,}")
                print(f"Characters: {total_chars:,}")
                print(f"Avg chunk size: {avg_size:.0f}")
                return True

            logger.error("Failed to save")
            return False

        except Exception as e:
            logger.error(f"Error creating vector DB: {e}")
            return False

    def show_summary(self):
        files = self.find_docling_files()
        if not files:
            print("No parsed docs found")
            return
        print(f"\nFound {len(files)} documents:")
        for i, f in enumerate(files, 1):
            kb = f['file_size'] / 1024
            print(f"{i}. {f['pdf_name']} ({kb:.1f} KB)")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Create vector DB from parsed Docling texts"
    )
    parser.add_argument('--pdfs-dir', default='pdfs')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    creator = DoclingVectorDBCreator(pdfs_dir=args.pdfs_dir)

    if args.summary:
        creator.show_summary()
        sys.exit(0)

    success = creator.create_vector_database(overwrite=args.overwrite)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
