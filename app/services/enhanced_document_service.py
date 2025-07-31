"""
Enhanced document processing service with PyMuPDF4LLM integration
"""
import aiohttp
import pymupdf
import pymupdf4llm as pm4
import tempfile
import os
import io
import time
import json
import hashlib
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from fastapi import HTTPException

# OCR imports (optional)
try:
    import pytesseract
    import PIL.Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARNING] OCR dependencies not available. Install pillow and pytesseract for OCR support.")

from app.models.requests import DocumentContent
from app.core.config import settings
from app.services.document_storage import DocumentStorageService


class EnhancedDocumentService:
    """Enhanced service for document processing with PyMuPDF4LLM"""
    
    def __init__(self, storage_service: Optional[DocumentStorageService] = None):
        self.storage_service = storage_service or DocumentStorageService()
    
    async def process_document_from_url(self, url: str, document_id: Optional[str] = None) -> DocumentContent:
        """Process document from URL with enhanced extraction"""
        try:
            # Download document
            pdf_data = await self.download_document(url)
            
            # Generate document ID if not provided
            if not document_id:
                document_id = self.storage_service.generate_document_id(pdf_data, url)
            
            # Store raw document
            filename = url.split('/')[-1] or "document.pdf"
            await self.storage_service.store_raw_document(
                document_id=document_id,
                content=pdf_data,
                filename=filename,
                source_type="url",
                source_reference=url
            )
            
            # Process document
            return await self.process_document_content(document_id, pdf_data)
            
        except Exception as e:
            if document_id:
                await self.storage_service.update_processing_status(
                    document_id, "failed", error_message=str(e)
                )
            raise
    
    async def process_document_from_storage(self, document_id: str) -> DocumentContent:
        """Process document from storage"""
        try:
            # Load document from storage
            doc_path = self.storage_service.get_document_path(document_id)
            raw_pdf_path = doc_path / "raw.pdf"
            
            if not raw_pdf_path.exists():
                raise HTTPException(404, f"Document {document_id} not found")
            
            # Read PDF data
            with open(raw_pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            return await self.process_document_content(document_id, pdf_data)
            
        except Exception as e:
            await self.storage_service.update_processing_status(
                document_id, "failed", error_message=str(e)
            )
            raise
    
    async def process_document_content(self, document_id: str, pdf_data: bytes) -> DocumentContent:
        """Enhanced document processing with PyMuPDF4LLM"""
        overall_start_time = time.time()
        temp_file_path = None
        
        try:
            await self.storage_service.update_processing_status(document_id, "processing")
            
            # Create temporary file
            temp_file_start = time.time()
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_data)
                temp_file_path = temp_file.name
            temp_file_time = time.time() - temp_file_start
            
            await self.storage_service.log_processing_event(document_id, "temp_file_created", {
                "file_size": len(pdf_data),
                "temp_path": temp_file_path,
                "temp_file_creation_time_ms": int(temp_file_time * 1000)
            })
            
            # Always use enhanced processing with PyMuPDF4LLM + OCR + span mapping
            enhanced_start = time.time()
            result = await self._enhanced_processing(document_id, temp_file_path, pdf_data)
            enhanced_time = time.time() - enhanced_start
            
            # Calculate processing times
            overall_time_ms = int((time.time() - overall_start_time) * 1000)
            enhanced_time_ms = int(enhanced_time * 1000)
            
            # Update status with detailed timing
            await self.storage_service.update_processing_status(
                document_id, "completed", processing_time_ms=overall_time_ms
            )
            
            # Log detailed timing breakdown
            await self.storage_service.log_processing_event(document_id, "timing_breakdown", {
                "overall_processing_time_ms": overall_time_ms,
                "enhanced_processing_time_ms": enhanced_time_ms,
                "temp_file_creation_time_ms": int(temp_file_time * 1000),
                "file_size_bytes": len(pdf_data),
                "processing_speed_chars_per_sec": len(result.text) / enhanced_time if enhanced_time > 0 else 0
            })
            
            return result
            
        except Exception as e:
            overall_time_ms = int((time.time() - overall_start_time) * 1000)
            await self.storage_service.update_processing_status(
                document_id, "failed", 
                processing_time_ms=overall_time_ms,
                error_message=str(e)
            )
            raise
        finally:
            # Cleanup temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
    
    
    async def _enhanced_processing(self, document_id: str, temp_file_path: str, 
                                 pdf_data: bytes) -> DocumentContent:
        """Enhanced processing with PyMuPDF4LLM and integrity checking"""
        processing_start = time.time()
        print(f"[INFO] Enhanced processing for {document_id}")
        
        # Step 1: Extract with PyMuPDF4LLM
        pymupdf4llm_start = time.time()
        markdown_result = await self._extract_with_pymupdf4llm(document_id, temp_file_path)
        pymupdf4llm_time = time.time() - pymupdf4llm_start
        
        # Step 2: Integrity checking and OCR fallback
        ocr_start = time.time()
        if settings.enable_ocr_fallback:
            await self._check_text_density_and_ocr(document_id, temp_file_path, markdown_result)
        ocr_time = time.time() - ocr_start
        
        # Step 3: Get basic metadata
        metadata_start = time.time()
        metadata = await self._extract_basic_metadata(temp_file_path)
        metadata_time = time.time() - metadata_start
        
        # Step 4: Store results (markdown only - text can be generated on-demand)
        storage_start = time.time()
        await self.storage_service.store_markdown(
            document_id, 
            markdown_result["markdown"], 
            markdown_result.get("span_mapping")
        )
        storage_time = time.time() - storage_start
        
        # Determine OCR pages from processing logs
        ocr_pages = []
        if settings.enable_ocr_fallback:
            # This would be populated during OCR processing
            # For now, we'll check if any OCR was performed based on density check
            doc = pymupdf.open(temp_file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if len(text.strip()) < settings.text_density_threshold:
                    ocr_pages.append(page_num + 1)
            doc.close()
        
        # Log detailed processing timing
        total_processing_time = time.time() - processing_start
        await self.storage_service.log_processing_event(document_id, "enhanced_processing_timing", {
            "total_enhanced_processing_time_ms": int(total_processing_time * 1000),
            "pymupdf4llm_extraction_time_ms": int(pymupdf4llm_time * 1000),
            "ocr_processing_time_ms": int(ocr_time * 1000),
            "metadata_extraction_time_ms": int(metadata_time * 1000),
            "storage_time_ms": int(storage_time * 1000),
            "text_extraction_speed_chars_per_sec": len(markdown_result["text"]) / pymupdf4llm_time if pymupdf4llm_time > 0 else 0,
            "total_characters_extracted": len(markdown_result["text"]),
            "total_markdown_length": len(markdown_result["markdown"])
        })
        
        return DocumentContent(
            document_id=document_id,
            text=markdown_result["text"],
            markdown=markdown_result["markdown"],
            pages=metadata["pages"],
            metadata=metadata,
            span_mapping=markdown_result.get("span_mapping"),
            ocr_pages=ocr_pages,
            processing_method="pymupdf4llm"
        )
    
    
    async def _extract_with_pymupdf4llm(self, document_id: str, temp_file_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF4LLM with span mapping"""
        try:
            # Extract markdown with PyMuPDF4LLM with parallel processing
            n_workers = settings.parallel_processing_workers
            if n_workers == 0:
                # Auto-detect CPU cores 
                n_workers = multiprocessing.cpu_count()
            
            # Use parallel processing if more than 1 worker is configured
            # Note: PyMuPDF4LLM may not yet support n_jobs parameter, but this is prepared for future versions
            if n_workers > 1:
                try:
                    print(f"[INFO] Attempting parallel processing with {n_workers} workers")
                    markdown_output = pm4.to_markdown(temp_file_path, show_progress=True)
                    # TODO: Add n_jobs parameter when supported: n_jobs=n_workers
                except TypeError as e:
                    if 'n_jobs' in str(e):
                        print(f"[INFO] n_jobs parameter not supported in current PyMuPDF4LLM version, falling back to single-threaded")
                        markdown_output = pm4.to_markdown(temp_file_path)
                    else:
                        raise
            else:
                print(f"[INFO] Using single-threaded processing")
                markdown_output = pm4.to_markdown(temp_file_path)
            
            # Extract plain text for backward compatibility
            plain_text = self.storage_service._markdown_to_text(markdown_output)
            
            # Extract span mapping (character positions to bbox) - only if feature flag enabled
            span_mapping = None
            if settings.store_span_map:
                span_mapping = await self._extract_span_mapping(temp_file_path, plain_text)
            
            await self.storage_service.log_processing_event(document_id, "pymupdf4llm_extraction", {
                "markdown_length": len(markdown_output),
                "text_length": len(plain_text),
                "span_mapping_entries": len(span_mapping) if span_mapping else 0,
                "span_mapping_enabled": settings.store_span_map
            })
            
            return {
                "markdown": markdown_output,
                "text": plain_text,
                "span_mapping": span_mapping
            }
            
        except Exception as e:
            await self.storage_service.log_processing_event(document_id, "pymupdf4llm_error", {
                "error": str(e)
            })
            raise HTTPException(500, f"PyMuPDF4LLM extraction failed: {str(e)}")
    
    
    async def _extract_span_mapping(self, temp_file_path: str, plain_text: str) -> Optional[Dict[str, Any]]:
        """Extract character span to bounding box mapping for precise citations"""
        try:
            doc = pymupdf.open(temp_file_path)
            span_mapping = {
                "pages": {},
                "text_blocks": [],
                "total_chars": len(plain_text)
            }
            
            current_char_pos = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_dict = page.get_text("dict")
                
                page_spans = []
                page_char_start = current_char_pos
                
                # Extract text blocks with position information
                for block in page_dict.get("blocks", []):
                    if "lines" in block:  # Text block
                        block_bbox = block.get("bbox", [0, 0, 0, 0])
                        
                        for line in block["lines"]:
                            line_bbox = line.get("bbox", [0, 0, 0, 0])
                            
                            for span in line["spans"]:
                                span_text = span.get("text", "")
                                span_bbox = span.get("bbox", [0, 0, 0, 0])
                                
                                if span_text.strip():
                                    span_info = {
                                        "char_start": current_char_pos,
                                        "char_end": current_char_pos + len(span_text),
                                        "text": span_text,
                                        "bbox": span_bbox,  # [x0, y0, x1, y1]
                                        "page": page_num + 1,
                                        "font": span.get("font", ""),
                                        "size": span.get("size", 0),
                                        "color": span.get("color", 0)
                                    }
                                    
                                    page_spans.append(span_info)
                                    span_mapping["text_blocks"].append(span_info)
                                    current_char_pos += len(span_text)
                
                # Add page-level information
                span_mapping["pages"][page_num + 1] = {
                    "char_start": page_char_start,
                    "char_end": current_char_pos,
                    "spans": page_spans,
                    "page_bbox": list(page.rect)  # Convert IRect to list [x0, y0, x1, y1]
                }
                
                # Add page separator in character count
                if page_num < len(doc) - 1:
                    current_char_pos += 1  # For page breaks
            
            doc.close()
            return span_mapping
            
        except Exception as e:
            print(f"[WARNING] Span mapping extraction failed: {str(e)}")
            return None
    
    async def _check_text_density_and_ocr(self, document_id: str, temp_file_path: str,
                                        markdown_result: Dict[str, Any]) -> None:
        """Check text density and trigger OCR for low-density pages"""
        try:
            doc = pymupdf.open(temp_file_path)
            low_density_pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Calculate text density (characters per page)
                text_density = len(text.strip())
                
                if text_density < settings.text_density_threshold:
                    low_density_pages.append(page_num + 1)  # 1-based page numbers
            
            doc.close()
            
            if low_density_pages:
                await self.storage_service.log_processing_event(document_id, "low_density_pages_detected", {
                    "pages": low_density_pages,
                    "threshold": settings.text_density_threshold
                })
                
                # Implement OCR fallback for low-density pages
                if OCR_AVAILABLE:
                    print(f"[INFO] Running OCR on pages {low_density_pages}")
                    ocr_results = await self._perform_ocr(document_id, temp_file_path, low_density_pages)
                    
                    if ocr_results:
                        # Store OCR results
                        await self.storage_service.store_ocr_results(
                            document_id, low_density_pages, ocr_results
                        )
                        
                        # Update markdown result with OCR text
                        await self._merge_ocr_with_extraction(markdown_result, ocr_results)
                else:
                    print(f"[WARNING] Pages {low_density_pages} have low text density, but OCR not available")
            
        except Exception as e:
            await self.storage_service.log_processing_event(document_id, "density_check_error", {
                "error": str(e)
            })
    
    async def _extract_basic_metadata(self, temp_file_path: str) -> Dict[str, Any]:
        """Extract basic document metadata"""
        try:
            doc = pymupdf.open(temp_file_path)
            
            metadata = {
                "pages": len(doc),
                "title": doc.metadata.get("title", "").strip(),
                "author": doc.metadata.get("author", "").strip(),
                "subject": doc.metadata.get("subject", "").strip(),
                "creator": doc.metadata.get("creator", "").strip(),
                "processing_method": "pymupdf4llm"
            }
            
            doc.close()
            return metadata
            
        except Exception as e:
            return {
                "pages": 0,
                "title": "",
                "author": "",
                "subject": "",
                "creator": "",
                "processing_method": "pymupdf4llm",
                "metadata_error": str(e)
            }
    
    async def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            timeout = aiohttp.ClientTimeout(total=settings.download_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(str(url)) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Failed to download document: HTTP {response.status}"
                        )
                    
                    content = await response.read()
                    
                    # Check file size
                    if len(content) > settings.max_file_size:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
                        )
                    
                    return content
                    
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected download error: {str(e)}")
    
    async def _perform_ocr(self, document_id: str, temp_file_path: str, page_numbers: List[int]) -> Optional[Dict[str, Any]]:
        """Perform OCR on specified pages using Tesseract"""
        if not OCR_AVAILABLE:
            print("[OCR] OCR dependencies not available. Install: pip install pillow pytesseract")
            return None
        
        try:
            # Check if Tesseract is available
            try:
                # Configure Tesseract if path is provided
                if settings.tesseract_cmd:
                    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
                
                # Test Tesseract availability
                pytesseract.get_tesseract_version()
            except pytesseract.TesseractNotFoundError:
                print("[OCR] Tesseract executable not found. Please install Tesseract:")
                print("      Ubuntu/Debian: sudo apt-get install tesseract-ocr")
                print("      macOS: brew install tesseract")
                print("      Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
                return None
            except Exception as e:
                print(f"[OCR] Tesseract test failed: {str(e)}")
                return None
            
            doc = pymupdf.open(temp_file_path)
            ocr_results = {}
            
            for page_num in page_numbers:
                try:
                    page_index = page_num - 1  # Convert to 0-based indexing
                    if page_index >= len(doc):
                        continue
                    
                    page = doc[page_index]
                    
                    # Render page as image at high DPI
                    mat = pymupdf.Matrix(settings.ocr_dpi / 72.0, settings.ocr_dpi / 72.0)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    pil_image = PIL.Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    print(f"[OCR] Processing page {page_num} at {settings.ocr_dpi} DPI...")
                    
                    # Get text with confidence scores
                    ocr_data = pytesseract.image_to_data(
                        pil_image, 
                        lang=settings.ocr_languages,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Extract text
                    ocr_text = pytesseract.image_to_string(
                        pil_image,
                        lang=settings.ocr_languages
                    ).strip()
                    
                    if ocr_text:
                        ocr_results[page_num] = {
                            "text": ocr_text,
                            "char_count": len(ocr_text),
                            "confidence_data": ocr_data,
                            "dpi": settings.ocr_dpi,
                            "languages": settings.ocr_languages
                        }
                        
                        await self.storage_service.log_processing_event(document_id, f"ocr_page_{page_num}", {
                            "char_count": len(ocr_text),
                            "text_preview": ocr_text[:100]
                        })
                        
                        print(f"[OCR] Page {page_num}: extracted {len(ocr_text)} characters")
                    else:
                        print(f"[OCR] Page {page_num}: no text extracted")
                
                except Exception as page_error:
                    print(f"[OCR ERROR] Failed to process page {page_num}: {str(page_error)}")
                    await self.storage_service.log_processing_event(document_id, f"ocr_page_{page_num}_error", {
                        "error": str(page_error)
                    })
            
            doc.close()
            
            if ocr_results:
                print(f"[OCR] Successfully processed {len(ocr_results)} pages")
                return ocr_results
            else:
                print(f"[OCR] No text extracted from any pages")
                return None
        
        except Exception as e:
            print(f"[OCR ERROR] OCR processing failed: {str(e)}")
            await self.storage_service.log_processing_event(document_id, "ocr_error", {
                "error": str(e)
            })
            return None
    
    async def _merge_ocr_with_extraction(self, markdown_result: Dict[str, Any], ocr_results: Dict[str, Any]) -> None:
        """Merge OCR results with existing extraction"""
        try:
            # Add OCR text as additional sections
            ocr_text_sections = []
            
            for page_num, ocr_data in ocr_results.items():
                ocr_text = ocr_data.get("text", "")
                if ocr_text.strip():
                    ocr_text_sections.append(f"\n--- OCR Text from Page {page_num} ---\n{ocr_text}")
            
            if ocr_text_sections:
                # Append OCR text to existing text and markdown
                ocr_content = "\n".join(ocr_text_sections)
                markdown_result["text"] += ocr_content
                markdown_result["markdown"] += ocr_content
                
                print(f"[OCR] Merged OCR text: {len(ocr_content)} additional characters")
        
        except Exception as e:
            print(f"[OCR ERROR] Failed to merge OCR results: {str(e)}")