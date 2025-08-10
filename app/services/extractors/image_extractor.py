"""
Image OCR extractor for extracting text from image files
"""
import time
import io
from typing import List, Optional
from .base_extractor import BaseExtractor, ExtractionResult


class ImageExtractor(BaseExtractor):
    """Image text extraction using OCR"""
    
    @property
    def supported_mime_types(self) -> List[str]:
        return [
            'image/png',
            'image/jpeg',
            'image/jpg', 
            'image/gif',
            'image/bmp',
            'image/tiff',
            'image/tif',
            'image/webp'
        ]
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'webp']
    
    def extract_text(self, file_data: bytes, filename: str = "") -> ExtractionResult:
        """
        Extract text from images using OCR
        
        Args:
            file_data: Image file bytes
            filename: Original filename
            
        Returns:
            ExtractionResult with extracted text and metadata
        """
        start_time = time.time()
        
        # Validate file size (images can be large)
        self.validate_file_size(file_data, max_size_mb=50)
        
        try:
            # Try PaddleOCR first (if available), then fallback to pytesseract
            result = self._extract_with_paddleocr(file_data, filename, start_time)
            if result and result.text.strip():
                return result
            
            # Fallback to pytesseract
            return self._extract_with_tesseract(file_data, filename, start_time)
            
        except Exception as e:
            raise Exception(f"Image OCR processing failed: {str(e)}")
    
    def _extract_with_paddleocr(self, file_data: bytes, filename: str, start_time: float) -> Optional[ExtractionResult]:
        """Extract text using PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            import numpy as np
            from PIL import Image
            
            # Initialize PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            
            # Load image from bytes
            image = Image.open(io.BytesIO(file_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Perform OCR
            ocr_results = ocr.ocr(img_array, cls=True)
            
            # Extract text from results
            extracted_texts = []
            confidence_scores = []
            
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if line[1] else ""
                        confidence = line[1][1] if line[1] and len(line[1]) > 1 else 0.0
                        
                        if text and text.strip():
                            extracted_texts.append(text.strip())
                            confidence_scores.append(confidence)
            
            # Combine all text
            all_text = '\n'.join(extracted_texts)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            processing_time = time.time() - start_time
            
            metadata = {
                'file_type': 'image_ocr',
                'filename': filename,
                'image_format': image.format or 'Unknown',
                'image_size': image.size,
                'image_mode': image.mode,
                'ocr_engine': 'PaddleOCR',
                'text_lines_found': len(extracted_texts),
                'average_confidence': round(avg_confidence, 3),
                'processing_time_seconds': processing_time,
                'extractor': 'paddleocr'
            }
            
            print("Image OCR (PaddleOCR) processing completed:")
            print(f"  - Image: {image.size[0]}x{image.size[1]} {image.format}")
            print(f"  - Text lines: {len(extracted_texts)}")
            print(f"  - Average confidence: {avg_confidence:.3f}")
            print(f"  - Text length: {len(all_text)} characters")
            print(f"  - Processing time: {processing_time:.2f}s")
            
            return ExtractionResult(
                text=all_text,
                pages=1,
                metadata=metadata,
                processing_time=processing_time
            )
            
        except ImportError:
            print("PaddleOCR not available, falling back to Tesseract")
            return None
        except Exception as e:
            print(f"PaddleOCR failed: {e}, falling back to Tesseract")
            return None
    
    def _extract_with_tesseract(self, file_data: bytes, filename: str, start_time: float) -> ExtractionResult:
        """Extract text using Tesseract OCR"""
        try:
            import pytesseract
            from PIL import Image
            
            # Load image from bytes
            image = Image.open(io.BytesIO(file_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform OCR with confidence data
            try:
                # Try to get detailed OCR data
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                # Extract text and confidence
                extracted_texts = []
                confidence_scores = []
                
                for i, text in enumerate(ocr_data['text']):
                    if text.strip():
                        confidence = int(ocr_data['conf'][i])
                        if confidence > 30:  # Filter low confidence text
                            extracted_texts.append(text.strip())
                            confidence_scores.append(confidence)
                
                all_text = ' '.join(extracted_texts)
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
                
            except Exception:
                # Fallback to simple text extraction
                all_text = pytesseract.image_to_string(image).strip()
                avg_confidence = 0.0
                extracted_texts = [all_text] if all_text else []
            
            processing_time = time.time() - start_time
            
            metadata = {
                'file_type': 'image_ocr',
                'filename': filename,
                'image_format': image.format or 'Unknown',
                'image_size': image.size,
                'image_mode': image.mode,
                'ocr_engine': 'Tesseract',
                'text_elements_found': len(extracted_texts),
                'average_confidence': round(avg_confidence, 1),
                'processing_time_seconds': processing_time,
                'extractor': 'pytesseract'
            }
            
            print("Image OCR (Tesseract) processing completed:")
            print(f"  - Image: {image.size[0]}x{image.size[1]} {image.format}")
            print(f"  - Text elements: {len(extracted_texts)}")
            print(f"  - Average confidence: {avg_confidence:.1f}")
            print(f"  - Text length: {len(all_text)} characters")
            print(f"  - Processing time: {processing_time:.2f}s")
            
            # Handle case where no text was extracted
            if not all_text or len(all_text.strip()) < 5:
                all_text = f"[No readable text found in image: {filename}]\n[Image processed but contains no detectable text content]"
                metadata['note'] = 'No readable text detected in image'
            
            return ExtractionResult(
                text=all_text,
                pages=1,
                metadata=metadata,
                processing_time=processing_time
            )
            
        except ImportError:
            raise Exception("Neither PaddleOCR nor pytesseract is available - cannot process image files")
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Return a result indicating OCR failure
            metadata = {
                'file_type': 'image_ocr',
                'filename': filename,
                'ocr_engine': 'failed',
                'processing_time_seconds': processing_time,
                'extractor': 'none',
                'error': str(e)
            }
            
            return ExtractionResult(
                text=f"[OCR processing failed for image: {filename}]\n[Error: {str(e)}]\n[Recommendation: Ensure image quality is sufficient for text recognition]",
                pages=1,
                metadata=metadata,
                processing_time=processing_time
            )