"""
Adaptive Segmentation Service with multiple strategies
"""
import time
from typing import List, Optional
from datetime import datetime

from app.models.requests import DocumentContent
from app.models.clause import ClauseExtractionResult
from app.services.segmentation_strategies import (
    SegmentationStrategy,
    PatternBasedStrategy,
    SpacyNLPStrategy, 
    FallbackChunkingStrategy
)

class AdaptiveSegmentationService:
    """
    Adaptive document segmentation service using multiple strategies
    Tries different approaches until it finds one that works
    """
    
    def __init__(self):
        # Initialize strategies in order of preference
        self.strategies: List[SegmentationStrategy] = []
        
        # Try to initialize each strategy
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize available strategies"""
        strategies_to_try = [
            (PatternBasedStrategy, "Enhanced pattern-based segmentation"),
            (SpacyNLPStrategy, "NLP-based intelligent segmentation"),
            (FallbackChunkingStrategy, "Simple fallback segmentation")
        ]
        
        for strategy_class, description in strategies_to_try:
            try:
                strategy = strategy_class()
                self.strategies.append(strategy)
                print(f"[SUCCESS] Initialized: {description}")
            except Exception as e:
                print(f"[ERROR] Failed to initialize {strategy_class.__name__}: {str(e)}")
        
        if not self.strategies:
            # If all else fails, ensure we have at least the fallback
            self.strategies = [FallbackChunkingStrategy()]
            print("[INFO] Using minimal fallback strategy only")
        
        print(f"[INFO] Total strategies available: {len(self.strategies)}")
    
    def segment_document(self, document_content: DocumentContent) -> ClauseExtractionResult:
        """
        Segment document using the first successful strategy
        """
        print(f"[INFO] Starting adaptive segmentation for {len(document_content.text)} character document")
        
        for i, strategy in enumerate(self.strategies):
            try:
                print(f"\n[ATTEMPT] Trying strategy {i+1}/{len(self.strategies)}: {strategy.name}")
                
                # Attempt segmentation
                result = strategy.segment(document_content)
                
                # Check if result is valid
                if self._is_valid_result(result):
                    print(f"[SUCCESS] Strategy {strategy.name} completed successfully!")
                    print(f"   - Clauses found: {len(result.clauses)}")
                    print(f"   - Sections found: {result.sections_found}")
                    print(f"   - Processing time: {result.processing_time_seconds}s")
                    
                    # Add strategy info to metadata
                    self._add_strategy_metadata(result, strategy.name, i + 1)
                    return result
                else:
                    print(f"[FAILED] {strategy.name} produced invalid result:")
                    print(f"   - Clauses: {len(result.clauses)}")
                    print(f"   - Errors: {result.errors}")
                    continue
                    
            except Exception as e:
                print(f"[ERROR] {strategy.name} failed with exception: {str(e)}")
                continue
        
        # If all strategies failed, return empty result
        print("[CRITICAL] All segmentation strategies failed!")
        return ClauseExtractionResult(
            clauses=[],
            pages_processed=0,
            sections_found=0,
            total_chars=len(document_content.text),
            processing_time_seconds=0.0,
            errors=["All segmentation strategies failed"],
            warnings=[]
        )
    
    def _is_valid_result(self, result: ClauseExtractionResult) -> bool:
        """Check if segmentation result is acceptable with adaptive thresholds"""
        clause_count = len(result.clauses)
        
        # Dynamic thresholds based on document characteristics
        min_clauses = max(5, result.total_chars // 4000)  # Conservative estimate
        max_clauses = min(1000, result.total_chars // 80)  # Allow higher density
        
        # For definition-heavy documents, expect more clauses
        if result.sections_found > 0:
            definition_ratio = result.sections_found / max(1, result.pages_processed)
            if definition_ratio > 1.5:  # Many sections per page = definition document
                min_clauses = max(min_clauses, 20)
        
        is_valid = (
            clause_count >= min_clauses and
            clause_count <= max_clauses and
            len(result.errors) == 0  # No critical errors
        )
        
        # Debug logging for validation
        if not is_valid:
            print(f"[VALIDATION] Result invalid: {clause_count} clauses (expected {min_clauses}-{max_clauses}), {len(result.errors)} errors")
        
        return is_valid
    
    def _add_strategy_metadata(self, result: ClauseExtractionResult, strategy_name: str, attempt_number: int):
        """Add metadata about which strategy was used"""
        for clause in result.clauses:
            if 'segmentation_metadata' not in clause.metadata:
                clause.metadata['segmentation_metadata'] = {}
            
            clause.metadata['segmentation_metadata'].update({
                'strategy_used': strategy_name,
                'attempt_number': attempt_number,
                'total_strategies_available': len(self.strategies),
                'timestamp': datetime.now().isoformat()
            })
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return [strategy.name for strategy in self.strategies]
    
    def segment_with_specific_strategy(self, document_content: DocumentContent, 
                                     strategy_name: str) -> Optional[ClauseExtractionResult]:
        """Segment using a specific strategy by name"""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                print(f"[INFO] Using specific strategy: {strategy_name}")
                return strategy.segment(document_content)
        
        print(f"[ERROR] Strategy '{strategy_name}' not available")
        return None
    
    def get_strategy_performance(self) -> dict:
        """Get performance statistics for strategies (placeholder for future)"""
        return {
            'total_strategies': len(self.strategies),
            'available_strategies': self.get_available_strategies(),
            'default_order': [s.name for s in self.strategies]
        }