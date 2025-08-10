#!/usr/bin/env python3
"""
Document Type Detector Service
Robust, multi-layer document classification system
"""
import re
from typing import Dict, Any, List, Tuple
from enum import Enum
from dataclasses import dataclass
import urllib.parse

class DocumentType(Enum):
    """Document processing types"""
    HACKRX_CHALLENGE = "hackrx_challenge"
    SEMANTIC_SEARCH = "semantic_search"
    STRUCTURED_DATA = "structured_data"
    UNKNOWN = "unknown"

@dataclass
class DocumentTypeResult:
    """Document type detection result"""
    document_type: DocumentType
    confidence: float
    detected_patterns: list
    suggested_pipeline: str
    metadata: Dict[str, Any]

class DocumentTypeDetector:
    """
    Robust, multi-layer document classification system
    Uses URL analysis, content patterns, structure analysis, and statistical features
    """
    
    def __init__(self):
        """Initialize enhanced document type detector"""
        
        # === URL-Based Classification Hints ===
        self.url_indicators = {
            "hackrx": ["hackrx.in", "register.hackrx", "challenge", "hack", "puzzle"],
            "api": ["api", "utils", "get-", "token", "secret", "endpoint"],
            "news": ["news", "media", "press", "article", "report"],
            "documents": [".pdf", ".doc", ".txt", "document", "file", "blob"]
        }
        
        # === High-Confidence Pattern Categories ===
        self.pattern_categories = {
            "hackrx_definitive": [
                r"hackrx|hack\s*rx|register\.hackrx\.in",
                r"secret\s*token|get.*secret|parallel\s*world",
                r"mission\s*brief|challenge\s*document|puzzle\s*solve",
                r"flight\s*number.*get|getFirstCityFlightNumber|getSecondCityFlightNumber",
                r"favorite\s*city|favourite\s*city|submissions/myFavouriteCity"
            ],
            
            "hackrx_geographic": [
                r"landmark\s+current\s*location|decode.*city",
                r"gateway\s*of\s*india|taj\s*mahal|eiffel\s*tower|big\s*ben",
                r"delhi|mumbai|chennai|hyderabad|bangalore|kolkata",
                r"new\s*york|london|tokyo|paris|dubai|singapore",
                r"choose.*flight\s*path|final\s*deliverable"
            ],
            
            "structured_tables": [
                r"landmark\s+current\s*location|city\s+landmark",
                r"\|.*\|.*\||┌.*┐|╔.*╗",  # Table borders
                r"[🏛️🗼🕌🏖️🌉🏰❤️⛩️🌸👑🪨🏢🌞✨🗽🕰️🏟️🎭✝️🏙️🖼️🌊🪨🕍🏞️🗿⛪🛰️🌆]{3,}",  # Multiple emojis
                r"(?:.*\n){2,}.*(?:\w+\s+){2,}\w+",  # Multi-line structured data
                r"table|mapping|matrix|grid|list\s*of\s*items"
            ],
            
            "news_content": [
                r"news|breaking|report|announced|announcement",
                r"president|government|policy|minister|official",
                r"trump|biden|modi|putin|xi|leader|administration",
                r"tariff|trade|economy|investment|billion|million",
                r"apple|google|microsoft|company|corporation|business",
                r"august|january|february|march|april|may|june|july|september|october|november|december",
                r"2025|2024|2023|2022|today|yesterday|recently|latest"
            ],
            
            "formal_documents": [
                r"policy|insurance|terms\s*and\s*conditions|agreement",
                r"coverage|premium|claim|benefit|liability|deductible",
                r"article\s+\d+|section\s+\d+|chapter\s+\d+|clause\s+\d+",
                r"constitution|law|legal|regulation|statute|act",
                r"whereas|hereby|therefore|pursuant|accordance|compliance"
            ],
            
            "general_content": [
                r"content|text|document|information|data|description",
                r"statement|press\s*release|media|publication|paper",
                r"research|study|analysis|findings|conclusion|summary",
                r"introduction|background|methodology|discussion|results"
            ]
        }
        
        # === Content Structure Analysis ===
        self.structure_patterns = {
            "table_like": r"(?:\w+\s*\|\s*\w+|\w+\s+\w+\s*\n\s*\w+\s+\w+)",
            "list_like": r"(?:^\s*[-•*]\s+|\d+\.\s+)",
            "paragraph_like": r"(?:[A-Z][^.!?]*[.!?]){3,}",
            "heading_like": r"(?:^|\n)(?:[A-Z][A-Z\s]+|[A-Z][^a-z\n]+)(?:\n|$)",
            "api_response": r"\{.*\}|\[.*\]|<.*>|response|json|xml"
        }
    
    def detect_document_type(self, text: str, url: str = None) -> DocumentTypeResult:
        """
        Enhanced multi-layer document type detection
        
        Uses 5-layer analysis:
        1. URL Analysis - Extract hints from URL structure
        2. Pattern Matching - High-confidence patterns by category
        3. Structure Analysis - Detect document layout/format 
        4. Statistical Analysis - Text characteristics and features
        5. Confidence Scoring - Weighted decision with fallback logic
        """
        
        # === Layer 1: URL Analysis ===
        url_hints = self._analyze_url(url)
        
        # === Layer 2: Pattern Matching ===
        pattern_scores = self._analyze_patterns(text)
        
        # === Layer 3: Structure Analysis ===
        structure_scores = self._analyze_structure(text)
        
        # === Layer 4: Statistical Analysis ===
        text_stats = self._analyze_text_statistics(text)
        
        # === Layer 5: Confidence Scoring & Decision ===
        result = self._make_classification_decision(
            text, url, url_hints, pattern_scores, structure_scores, text_stats
        )
        
        return result
    
    def _analyze_url(self, url: str) -> Dict[str, float]:
        """Analyze URL for classification hints"""
        hints = {"hackrx": 0.0, "api": 0.0, "news": 0.0, "documents": 0.0}
        
        if not url:
            return hints
            
        url_lower = url.lower()
        parsed = urllib.parse.urlparse(url_lower)
        
        # Domain analysis
        domain = parsed.netloc + parsed.path
        
        for category, indicators in self.url_indicators.items():
            if category in hints:  # Only process categories we track
                for indicator in indicators:
                    if indicator in domain:
                        hints[category] += 0.3
                    
        # Special URL patterns
        if "hackrx.in" in domain:
            hints["hackrx"] += 0.5
        if any(x in domain for x in ["token", "secret", "get-", "api/"]):
            hints["api"] += 0.4
        if any(x in domain for x in ["news", "media", "press", "article"]):
            hints["news"] += 0.4
        if any(x in domain for x in [".pdf", "blob", "document"]):
            hints["documents"] += 0.3
            
        return hints
    
    def _analyze_patterns(self, text: str) -> Dict[str, Tuple[int, float]]:
        """Advanced pattern analysis with weighted scoring"""
        scores = {}
        
        for category, patterns in self.pattern_categories.items():
            matches = self._count_weighted_patterns(patterns, text)
            # Normalize by text length but with minimum impact
            text_length = max(len(text), 1000)
            normalized_score = min(matches / (text_length / 1000), 1.0)
            scores[category] = (matches, normalized_score)
            
        return scores
    
    def _analyze_structure(self, text: str) -> Dict[str, float]:
        """Analyze document structure and formatting"""
        scores = {}
        
        for structure_type, pattern in self.structure_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
            # Normalize by text lines to detect structure density
            lines = max(len(text.split('\n')), 1)
            scores[structure_type] = min(matches / lines, 1.0)
            
        return scores
    
    def _analyze_text_statistics(self, text: str) -> Dict[str, Any]:
        """Statistical analysis of text characteristics"""
        lines = text.split('\n')
        words = text.split()
        
        stats = {
            "text_length": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "avg_line_length": sum(len(line) for line in lines) / max(len(lines), 1),
            "avg_word_length": sum(len(word) for word in words) / max(len(words), 1),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "punctuation_density": sum(1 for c in text if c in ".,!?;:") / max(len(text), 1),
            "number_density": sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            "short_lines": sum(1 for line in lines if len(line.strip()) < 50),
            "empty_lines": sum(1 for line in lines if not line.strip())
        }
        
        # Derived features
        stats["structure_indicators"] = (
            stats["short_lines"] / max(stats["line_count"], 1) +
            stats["empty_lines"] / max(stats["line_count"], 1)
        )
        
        return stats
    
    def _make_classification_decision(self, text: str, url: str, url_hints: Dict, 
                                   pattern_scores: Dict, structure_scores: Dict, 
                                   text_stats: Dict) -> DocumentTypeResult:
        """Make final classification decision using all analysis layers"""
        
        detected_patterns = []
        metadata = {
            "url_hints": url_hints,
            "pattern_scores": pattern_scores, 
            "structure_scores": structure_scores,
            "text_stats": text_stats
        }
        
        # === HACKRX CHALLENGE Detection ===
        hackrx_score = 0.0
        hackrx_score += url_hints.get("hackrx", 0) * 0.6  # URL is strong signal
        hackrx_score += url_hints.get("api", 0) * 0.3     # API endpoints often hackrx
        hackrx_score += pattern_scores.get("hackrx_definitive", (0, 0))[1] * 0.8
        hackrx_score += pattern_scores.get("hackrx_geographic", (0, 0))[1] * 0.4
        
        # Special boost for definitive patterns
        if pattern_scores.get("hackrx_definitive", (0, 0))[0] > 0:
            hackrx_score += 0.3
            detected_patterns.append("hackrx_definitive_patterns")
            
        if hackrx_score > 0.5:
            return DocumentTypeResult(
                document_type=DocumentType.HACKRX_CHALLENGE,
                confidence=min(hackrx_score, 1.0),
                detected_patterns=detected_patterns + ["hackrx_classification"],
                suggested_pipeline="direct_processing",
                metadata={**metadata, "classification_method": "hackrx_weighted_scoring"}
            )
        
        # === STRUCTURED DATA Detection ===
        structured_score = 0.0
        structured_score += pattern_scores.get("structured_tables", (0, 0))[1] * 0.6
        structured_score += structure_scores.get("table_like", 0) * 0.5
        structured_score += structure_scores.get("list_like", 0) * 0.3
        
        # High structure indicators from text stats
        if text_stats["structure_indicators"] > 0.3:
            structured_score += 0.3
            
        # Only classify as structured if very clear and not semantic content
        semantic_indicators = (
            pattern_scores.get("news_content", (0, 0))[1] +
            pattern_scores.get("formal_documents", (0, 0))[1] +
            pattern_scores.get("general_content", (0, 0))[1]
        )
        
        if structured_score > 0.6 and structured_score > (semantic_indicators * 2) and hackrx_score < 0.3:
            return DocumentTypeResult(
                document_type=DocumentType.STRUCTURED_DATA,
                confidence=min(structured_score, 1.0),
                detected_patterns=detected_patterns + ["structured_data_classification"],
                suggested_pipeline="direct_processing",
                metadata={**metadata, "classification_method": "structured_weighted_scoring"}
            )
        
        # === SEMANTIC SEARCH Detection ===
        semantic_score = 0.0
        semantic_score += pattern_scores.get("news_content", (0, 0))[1] * 0.7
        semantic_score += pattern_scores.get("formal_documents", (0, 0))[1] * 0.6  
        semantic_score += pattern_scores.get("general_content", (0, 0))[1] * 0.5
        semantic_score += url_hints.get("news", 0) * 0.4
        semantic_score += url_hints.get("documents", 0) * 0.3
        
        # Boost for paragraph-like structure (typical of readable documents)
        if structure_scores.get("paragraph_like", 0) > 0.1:
            semantic_score += 0.2
            detected_patterns.append("paragraph_structure")
            
        # Boost for reasonable text statistics (readable content)
        if (text_stats["text_length"] > 200 and 
            text_stats["avg_word_length"] > 3 and 
            text_stats["punctuation_density"] > 0.02):
            semantic_score += 0.3
            detected_patterns.append("readable_text_statistics")
        
        # Very aggressive fallback to semantic search for general content
        if (semantic_score > 0.1 or 
            text_stats["text_length"] > 200 or
            (hackrx_score < 0.3 and structured_score < 0.5)):
            
            confidence = max(semantic_score, 0.7)  # High default confidence
            
            classification_reason = "pattern_matching" if semantic_score > 0.3 else "content_fallback"
            
            return DocumentTypeResult(
                document_type=DocumentType.SEMANTIC_SEARCH,
                confidence=confidence,
                detected_patterns=detected_patterns + ["semantic_search_classification"],
                suggested_pipeline="rag_pipeline",
                metadata={
                    **metadata, 
                    "classification_method": "semantic_weighted_scoring",
                    "classification_reason": classification_reason,
                    "semantic_score": semantic_score
                }
            )
        
        # === UNKNOWN Fallback ===
        return DocumentTypeResult(
            document_type=DocumentType.UNKNOWN,
            confidence=0.3,
            detected_patterns=detected_patterns + ["unknown_classification"],
            suggested_pipeline="rag_pipeline",  # Safe fallback
            metadata={**metadata, "classification_method": "fallback", "reason": "no_clear_classification"}
        )
    
    def _count_weighted_patterns(self, patterns: List[str], text: str) -> int:
        """Count pattern matches with weighted scoring for pattern importance"""
        total_score = 0
        for i, pattern in enumerate(patterns):
            matches = len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
            # Earlier patterns in list are more important
            weight = 1.0 - (i * 0.1)  # First pattern = 1.0, second = 0.9, etc.
            total_score += matches * max(weight, 0.3)  # Minimum weight of 0.3
        return int(total_score)
    
    def _count_pattern_matches(self, patterns: list, text: str) -> int:
        """Legacy method - use _count_weighted_patterns instead"""
        return self._count_weighted_patterns(patterns, text)
    
    def is_hackrx_document(self, text: str, url: str = None) -> bool:
        """Quick check if document is HackRX challenge type"""
        result = self.detect_document_type(text, url)
        return result.document_type == DocumentType.HACKRX_CHALLENGE
    
    def requires_direct_processing(self, text: str, url: str = None) -> bool:
        """Check if document should use direct processing instead of RAG"""
        result = self.detect_document_type(text, url)
        return result.suggested_pipeline == "direct_processing"


# Singleton instance
_document_type_detector = None

def get_document_type_detector() -> DocumentTypeDetector:
    """Get or create document type detector instance"""
    global _document_type_detector
    if _document_type_detector is None:
        _document_type_detector = DocumentTypeDetector()
    return _document_type_detector