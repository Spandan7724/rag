#!/usr/bin/env python3
"""
Direct Document Processing Pipeline
Handles structured data documents without vector search
"""
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.services.document_processor import get_document_processor
from app.services.table_extractor import get_table_extractor, LandmarkMapping
from app.services.document_type_detector import get_document_type_detector, DocumentType
from app.utils.debug import conditional_print

@dataclass
class DirectProcessingResult:
    """Result of direct document processing"""
    doc_id: str
    document_text: str
    landmark_mappings: Dict[str, List[LandmarkMapping]]
    processing_time: float
    document_type: DocumentType
    metadata: Dict[str, Any]
    pipeline_used: str = "direct_processing"

class DirectDocumentProcessor:
    """
    Direct document processing for structured data
    Bypasses vector search for documents that need complete content access
    """
    
    def __init__(self):
        """Initialize direct processor"""
        self.document_processor = get_document_processor()
        self.table_extractor = get_table_extractor()
        self.type_detector = get_document_type_detector()
        
        print("Direct Document Processor initialized for structured data processing")
    
    async def process_document_direct(self, url: str) -> DirectProcessingResult:
        """
        Process document using direct approach (no vector search)
        
        Args:
            url: Document URL to process
            
        Returns:
            DirectProcessingResult with complete document data
        """
        start_time = time.time()
        
        print(f"Starting direct document processing: {url}")
        
        # Generate document ID
        doc_id = self.document_processor.generate_doc_id(url)
        
        try:
            # Stage 1: Document processing (get complete text)
            processed_doc = await self.document_processor.process_document(url)
            
            print(f"Document processed: {len(processed_doc.text)} characters")
            
            # Stage 2: Document type detection
            type_result = self.type_detector.detect_document_type(
                processed_doc.text, 
                url
            )
            
            print(f"Document type detected: {type_result.document_type.value} "
                  f"(confidence: {type_result.confidence:.2f})")
            
            # Stage 3: Extract structured data based on document type
            landmark_mappings = {}
            
            if type_result.document_type == DocumentType.HACKRX_CHALLENGE:
                # Extract landmark mappings for HackRX challenges
                text_mappings = self.table_extractor.extract_landmark_mappings_from_text(
                    processed_doc.text
                )
                landmark_mappings = self.table_extractor.create_lookup_dict(text_mappings)
                
                conditional_print(f"Extracted {len(text_mappings)} landmark mappings "
                      f"for {len(landmark_mappings)} cities")
                
                # Log extracted mappings for verification
                for city, mappings_list in landmark_mappings.items():
                    landmarks = [m.landmark for m in mappings_list]
                    conditional_print(f"  {city.title()}: {landmarks}")
            
            elif type_result.document_type == DocumentType.STRUCTURED_DATA:
                # Extract structured data for other table-like documents
                text_mappings = self.table_extractor.extract_landmark_mappings_from_text(
                    processed_doc.text
                )
                landmark_mappings = self.table_extractor.create_lookup_dict(text_mappings)
                
                conditional_print(f"Extracted {len(text_mappings)} structured data elements")
            
            total_time = time.time() - start_time
            
            # Compile results
            result = DirectProcessingResult(
                doc_id=doc_id,
                document_text=processed_doc.text,
                landmark_mappings=landmark_mappings,
                processing_time=total_time,
                document_type=type_result.document_type,
                metadata={
                    "document_stats": {
                        "pages": processed_doc.pages,
                        "text_length": len(processed_doc.text),
                        "landmark_count": sum(len(mappings) for mappings in landmark_mappings.values()),
                        "city_count": len(landmark_mappings)
                    },
                    "type_detection": {
                        "detected_type": type_result.document_type.value,
                        "confidence": type_result.confidence,
                        "patterns": type_result.detected_patterns,
                        "suggested_pipeline": type_result.suggested_pipeline
                    },
                    "processing_info": {
                        "pipeline": "direct_processing",
                        "vector_search_bypassed": True,
                        "complete_text_available": True
                    }
                }
            )
            
            conditional_print(f"Direct document processing completed in {total_time:.2f}s")
            conditional_print(f"  - Document type: {type_result.document_type.value}")
            conditional_print(f"  - Text length: {len(processed_doc.text):,} characters")
            conditional_print(f"  - Landmark mappings: {sum(len(m) for m in landmark_mappings.values())}")
            conditional_print(f"  - Cities covered: {len(landmark_mappings)}")
            
            return result
            
        except Exception as e:
            print(f"Direct document processing failed: {str(e)}")
            raise
    
    def get_complete_document_text(self, doc_id: str) -> Optional[str]:
        """
        Get complete document text by doc_id (if stored)
        This would typically interface with a document cache or storage
        """
        # For now, this is a placeholder - in production you'd implement
        # document caching/storage to retrieve processed documents
        conditional_print(f"Retrieving complete text for document: {doc_id}")
        return None
    
    def extract_structured_data(
        self, 
        text: str, 
        document_type: DocumentType
    ) -> Dict[str, Any]:
        """
        Extract structured data based on document type
        
        Args:
            text: Complete document text
            document_type: Detected document type
            
        Returns:
            Extracted structured data
        """
        if document_type == DocumentType.HACKRX_CHALLENGE:
            # Extract landmark mappings
            text_mappings = self.table_extractor.extract_landmark_mappings_from_text(text)
            landmark_mappings = self.table_extractor.create_lookup_dict(text_mappings)
            
            return {
                "type": "landmark_mappings",
                "data": landmark_mappings,
                "count": len(text_mappings),
                "cities": len(landmark_mappings)
            }
        
        elif document_type == DocumentType.STRUCTURED_DATA:
            # Generic structured data extraction
            return {
                "type": "structured_data",
                "data": {},  # Placeholder for other structured data types
                "text_length": len(text)
            }
        
        else:
            # No structured extraction needed
            return {
                "type": "unstructured",
                "text_length": len(text)
            }
    
    def should_use_direct_processing(self, text: str, url: str = None) -> bool:
        """
        Determine if document should use direct processing
        
        Args:
            text: Document text content
            url: Optional document URL
            
        Returns:
            True if direct processing recommended
        """
        return self.type_detector.requires_direct_processing(text, url)


# Singleton instance
_direct_processor = None

def get_direct_processor() -> DirectDocumentProcessor:
    """Get or create direct processor instance"""
    global _direct_processor
    if _direct_processor is None:
        _direct_processor = DirectDocumentProcessor()
    return _direct_processor