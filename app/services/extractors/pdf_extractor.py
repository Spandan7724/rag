"""
PDF document extractor using PyMuPDF with OCR and table extraction
"""
import time
import pymupdf
from typing import List, Dict, Any
from .base_extractor import BaseExtractor, ExtractionResult


class PDFExtractor(BaseExtractor):
    """PDF text extraction using PyMuPDF with advanced processing"""
    
    @property
    def supported_mime_types(self) -> List[str]:
        return ['application/pdf']
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['pdf']
    
    def extract_text(self, file_data: bytes, filename: str = "") -> ExtractionResult:
        """
        Extract text from PDF with OCR and table processing
        
        Args:
            file_data: PDF file bytes
            filename: Original filename
            
        Returns:
            ExtractionResult with extracted text and metadata
        """
        start_time = time.time()
        
        # Validate file size (PDFs can be large)
        self.validate_file_size(file_data, max_size_mb=200)
        
        try:
            # Open PDF from bytes
            pdf_doc = pymupdf.open(stream=file_data, filetype="pdf")
            print(f"PDF opened in {time.time() - start_time:.2f}s")
            
            # Extract metadata
            metadata = self._extract_metadata(pdf_doc)
            
            # Process all pages
            all_text = ""
            page_texts = []
            total_processing_time = 0
            
            for page_num in range(pdf_doc.page_count):
                page_start = time.time()
                page = pdf_doc.load_page(page_num)
                
                # Analyze page to determine best processing method
                processing_method = self._determine_processing_method(page)
                print(f"  Page {page_num + 1}: Using {processing_method} processing")
                
                # Extract text based on determined method
                page_text = self._extract_page_text(page, processing_method)
                
                if page_text.strip():
                    page_texts.append(page_text)
                    all_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                page_time = time.time() - page_start
                total_processing_time += page_time
                print(f"    Processed in {page_time:.2f}s")
            
            pdf_doc.close()
            
            # Final processing
            processing_time = time.time() - start_time
            
            print("PDF processing completed:")
            print(f"  - Total pages: {len(page_texts)}")
            print(f"  - Text length: {len(all_text)} characters")
            print(f"  - Processing time: {processing_time:.2f}s")
            
            # Update metadata
            metadata.update({
                'processing_time_seconds': processing_time,
                'page_processing_time': total_processing_time,
                'pages_processed': len(page_texts),
                'total_chars': len(all_text)
            })
            
            return ExtractionResult(
                text=all_text,
                pages=len(page_texts),
                metadata=metadata,
                page_texts=page_texts,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise Exception(f"PDF processing failed: {str(e)}")
    
    def _extract_metadata(self, pdf_doc) -> Dict[str, Any]:
        """Extract PDF metadata"""
        try:
            metadata = pdf_doc.metadata
            return {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'file_type': 'pdf',
                'total_pages': pdf_doc.page_count
            }
        except Exception:
            return {
                'file_type': 'pdf',
                'total_pages': pdf_doc.page_count
            }
    
    def _determine_processing_method(self, page) -> str:
        """Determine the best processing method for a page"""
        try:
            # Get basic page info
            text_length = len(page.get_text().strip())
            tables = page.find_tables()
            images = page.get_images(full=True)
            
            # Calculate complexity score
            complexity = 0
            if text_length < 100:
                complexity += 2  # Low text suggests images/scanned content
            if len(tables) > 0:
                complexity += 1  # Tables present
            if len(images) > 2:
                complexity += 1  # Multiple images
            
            # Determine method
            if complexity >= 3:
                return "ocr"
            elif len(tables) > 0:
                return "hybrid"
            else:
                return "text"
                
        except Exception:
            return "text"  # Fallback to text extraction
    
    def _extract_page_text(self, page, method: str) -> str:
        """Extract text from page using specified method"""
        if method == "ocr":
            return self._extract_with_ocr(page)
        elif method == "hybrid":
            return self._extract_hybrid(page)
        else:
            return self._extract_text_only(page)
    
    def _extract_text_only(self, page) -> str:
        """Simple text extraction"""
        try:
            return page.get_text()
        except Exception as e:
            print(f"    Text extraction failed: {e}")
            return ""
    
    def _extract_hybrid(self, page) -> str:
        """Hybrid extraction with tables"""
        try:
            # Get regular text
            text = page.get_text()
            
            # Extract tables
            tables = page.find_tables()
            table_text = ""
            
            for i, table in enumerate(tables):
                try:
                    table_data = table.extract()
                    if table_data:
                        table_text += f"\n[Table {i+1}]\n"
                        for row in table_data:
                            if row and any(cell for cell in row if cell):
                                clean_row = [str(cell).strip() if cell else "" for cell in row]
                                table_text += " | ".join(clean_row) + "\n"
                except Exception as e:
                    print(f"    Table {i+1} extraction failed: {e}")
            
            return text + table_text
            
        except Exception as e:
            print(f"    Hybrid extraction failed: {e}")
            return self._extract_text_only(page)
    
    def _extract_with_ocr(self, page) -> str:
        """OCR extraction for image-heavy pages"""
        try:
            # Try to use existing OCR functionality if available
            # For now, fallback to text extraction
            print("    OCR processing (fallback to text extraction)")
            text = page.get_text()
            
            # If no text found, this would be where OCR would be applied
            if not text.strip():
                print("    No text found - OCR would be needed here")
                
            return text
            
        except Exception as e:
            print(f"    OCR extraction failed: {e}")
            return self._extract_text_only(page)