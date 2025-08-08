#!/usr/bin/env python3
"""
Challenge Detector Service
Intelligently detects different types of challenges and routes them to appropriate handlers
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ChallengeType(Enum):
    """Types of challenges that can be detected"""
    GEOGRAPHIC_PUZZLE = "geographic_puzzle"
    FLIGHT_NUMBER = "flight_number" 
    SECRET_TOKEN = "secret_token"
    WEB_SCRAPING = "web_scraping"
    MULTILINGUAL_QA = "multilingual_qa"
    STANDARD_RAG = "standard_rag"
    COMPLEX_REASONING = "complex_reasoning"

@dataclass
class ChallengeDetection:
    """Result of challenge detection"""
    challenge_type: ChallengeType
    confidence: float
    detected_patterns: List[str]
    suggested_approach: str
    metadata: Dict[str, Any]

class ChallengeDetector:
    """
    Intelligent challenge detector that analyzes questions and documents
    to determine the appropriate solving approach
    """
    
    def __init__(self):
        """Initialize challenge detector with pattern definitions"""
        
        # Patterns for geographic puzzle detection
        self.geographic_patterns = [
            r'flight\s+number',
            r'parallel\s+world',
            r'landmark.*city|city.*landmark',
            r'gateway\s+of\s+india|taj\s+mahal|eiffel\s+tower|big\s+ben',
            r'mumbai|delhi|chennai|hyderabad|london|paris|tokyo',
            r'sachin.*world',
            r'twisted\s+map',
            r'real\s+world'
        ]
        
        # Patterns for flight number requests
        self.flight_patterns = [
            r'what\s+is\s+(my|the)\s+flight\s+number',
            r'give\s+me\s+(the\s+)?flight\s+number',
            r'find\s+(my|the)\s+flight',
            r'flight\s+(code|id|number)',
            r'which\s+flight'
        ]
        
        # Patterns for token/credential requests
        self.token_patterns = [
            r'secret\s+token',
            r'get.*token',
            r'authorization\s+token',
            r'bearer\s+token',
            r'access\s+token',
            r'api\s+key',
            r'credential'
        ]
        
        # Patterns for web scraping requests
        self.scraping_patterns = [
            r'go\s+to\s+(the\s+)?link',
            r'visit\s+the\s+url',
            r'fetch\s+from\s+website',
            r'scrape\s+(the\s+)?page',
            r'get\s+content\s+from',
            r'extract\s+from\s+webpage'
        ]
        
        # Patterns for multilingual content with enhanced language detection
        self.multilingual_patterns = {
            'hindi': [r'[\u0900-\u097F]+'],  # Devanagari (Hindi)
            'bengali': [r'[\u0980-\u09FF]+'],  # Bengali
            'malayalam': [
                r'[\u0D00-\u0D7F]+',  # Malayalam Unicode range
                r'ട്രംപ്',  # Trump in Malayalam
                r'ശുല്ക്കം',  # Shulkkam (tariff/tax)
                r'യുഎസ്',  # US in Malayalam
                r'കമ്പ്യൂട്ടർ',  # Computer in Malayalam
                r'സർക്കാർ',  # Government
                r'ബജറ്റ്',  # Budget
                r'കമ്പനി',  # Company
                r'വരുമാനം'  # Income/revenue
            ],
            'tamil': [r'[\u0B80-\u0BFF]+'],  # Tamil
            'telugu': [r'[\u0C00-\u0C7F]+']  # Telugu
        }
        
        # Patterns for complex reasoning
        self.reasoning_patterns = [
            r'compare.*and.*',
            r'analyze.*relationship',
            r'step\s+by\s+step',
            r'workflow|process|procedure',
            r'if.*then.*else',
            r'based\s+on.*determine',
            r'calculate.*using'
        ]
        
        # Document content indicators with enhanced multilingual support
        self.document_indicators = {
            'geographic_challenge': [
                'parallel world', 'landmark', 'Gateway of India', 'Mission Brief',
                'Sachin', 'twisted map', 'flight path', 'favorite city'
            ],
            'malayalam_news': [
                'ട്രംപ്',  # Trump
                'ശുല്ക്കം',  # Shulkkam (tariff)
                'യുഎസ്',  # US
                'സർക്കാർ',  # Government
                'വിപണി',  # Market/trading
                'august', '2025'
            ],
            'insurance_policy': [
                'policy', 'premium', 'coverage', 'exclusion', 'waiting period'
            ],
            'technical_manual': [
                'specification', 'engine oil', 'maintenance', 'tyre pressure'
            ]
        }
    
    def detect_challenge_type(self, questions: List[str], document_content: str = "") -> List[ChallengeDetection]:
        """
        Detect challenge types from questions and document content
        
        Args:
            questions: List of questions to analyze
            document_content: Optional document content for context
            
        Returns:
            List of challenge detections sorted by confidence
        """
        detections = []
        
        # Combine all questions for analysis
        combined_questions = " ".join(questions).lower()
        doc_content_lower = document_content.lower()
        
        # Detect geographic puzzle challenge
        geographic_detection = self._detect_geographic_puzzle(
            combined_questions, doc_content_lower
        )
        if geographic_detection.confidence > 0.3:
            detections.append(geographic_detection)
        
        # Detect flight number requests
        flight_detection = self._detect_flight_request(combined_questions)
        if flight_detection.confidence > 0.3:
            detections.append(flight_detection)
        
        # Detect token requests
        token_detection = self._detect_token_request(combined_questions)
        if token_detection.confidence > 0.3:
            detections.append(token_detection)
        
        # Detect web scraping needs
        scraping_detection = self._detect_web_scraping(combined_questions)
        if scraping_detection.confidence > 0.3:
            detections.append(scraping_detection)
        
        # Detect multilingual content
        multilingual_detection = self._detect_multilingual(combined_questions, doc_content_lower)
        if multilingual_detection.confidence > 0.3:
            detections.append(multilingual_detection)
        
        # Detect complex reasoning
        reasoning_detection = self._detect_complex_reasoning(combined_questions)
        if reasoning_detection.confidence > 0.3:
            detections.append(reasoning_detection)
        
        # If no specific challenge detected, default to standard RAG
        if not detections:
            detections.append(ChallengeDetection(
                challenge_type=ChallengeType.STANDARD_RAG,
                confidence=1.0,
                detected_patterns=["default"],
                suggested_approach="Use standard RAG pipeline with document retrieval and answer generation",
                metadata={"default_approach": True}
            ))
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return detections
    
    def _detect_geographic_puzzle(self, questions: str, doc_content: str) -> ChallengeDetection:
        """Detect geographic puzzle challenge"""
        patterns_found = []
        confidence = 0.0
        
        # Check question patterns
        for pattern in self.geographic_patterns:
            if re.search(pattern, questions, re.IGNORECASE):
                patterns_found.append(pattern)
                confidence += 0.2
        
        # Check document content indicators
        for indicator in self.document_indicators['geographic_challenge']:
            if indicator.lower() in doc_content:
                patterns_found.append(f"doc_indicator: {indicator}")
                confidence += 0.3
        
        # Boost confidence if multiple related patterns found
        if len(patterns_found) >= 3:
            confidence += 0.2
        
        confidence = min(confidence, 1.0)
        
        return ChallengeDetection(
            challenge_type=ChallengeType.GEOGRAPHIC_PUZZLE,
            confidence=confidence,
            detected_patterns=patterns_found,
            suggested_approach="Use multi-step geographic challenge solver with landmark mapping and API integration",
            metadata={
                "requires_api_calls": True,
                "needs_table_extraction": True,
                "multi_step_process": True
            }
        )
    
    def _detect_flight_request(self, questions: str) -> ChallengeDetection:
        """Detect flight number request"""
        patterns_found = []
        confidence = 0.0
        
        for pattern in self.flight_patterns:
            if re.search(pattern, questions, re.IGNORECASE):
                patterns_found.append(pattern)
                confidence += 0.4
        
        confidence = min(confidence, 1.0)
        
        return ChallengeDetection(
            challenge_type=ChallengeType.FLIGHT_NUMBER,
            confidence=confidence,
            detected_patterns=patterns_found,
            suggested_approach="Extract flight number from document or trigger geographic challenge workflow",
            metadata={
                "direct_extraction": True,
                "fallback_to_geographic": True
            }
        )
    
    def _detect_token_request(self, questions: str) -> ChallengeDetection:
        """Detect secret token request"""
        patterns_found = []
        confidence = 0.0
        
        for pattern in self.token_patterns:
            if re.search(pattern, questions, re.IGNORECASE):
                patterns_found.append(pattern)
                confidence += 0.5
        
        confidence = min(confidence, 1.0)
        
        return ChallengeDetection(
            challenge_type=ChallengeType.SECRET_TOKEN,
            confidence=confidence,
            detected_patterns=patterns_found,
            suggested_approach="Use web client to fetch secret token from API endpoint",
            metadata={
                "requires_web_scraping": True,
                "api_endpoint_call": True
            }
        )
    
    def _detect_web_scraping(self, questions: str) -> ChallengeDetection:
        """Detect web scraping requirements"""
        patterns_found = []
        confidence = 0.0
        
        for pattern in self.scraping_patterns:
            if re.search(pattern, questions, re.IGNORECASE):
                patterns_found.append(pattern)
                confidence += 0.4
        
        # Check for URL patterns
        url_pattern = r'https?://[^\s]+'
        if re.search(url_pattern, questions):
            patterns_found.append("url_detected")
            confidence += 0.3
        
        confidence = min(confidence, 1.0)
        
        return ChallengeDetection(
            challenge_type=ChallengeType.WEB_SCRAPING,
            confidence=confidence,
            detected_patterns=patterns_found,
            suggested_approach="Use web client to fetch and extract content from specified URLs",
            metadata={
                "needs_web_client": True,
                "extract_tokens": True
            }
        )
    
    def _detect_multilingual(self, questions: str, doc_content: str) -> ChallengeDetection:
        """Detect multilingual content with enhanced language identification"""
        patterns_found = []
        confidence = 0.0
        detected_languages = []
        
        combined_text = questions + " " + doc_content
        
        # Check each language's patterns
        for language, patterns in self.multilingual_patterns.items():
            language_found = False
            for pattern in patterns:
                if re.search(pattern, combined_text):
                    patterns_found.append(f"{language}_pattern: {pattern}")
                    language_found = True
                    
            if language_found:
                detected_languages.append(language)
                confidence += 0.4  # Higher confidence per language
        
        # Check for Malayalam news content specifically
        malayalam_news_count = 0
        for indicator in self.document_indicators['malayalam_news']:
            if indicator in doc_content:
                malayalam_news_count += 1
                
        if malayalam_news_count > 0:
            patterns_found.append(f"malayalam_news_content: {malayalam_news_count} indicators")
            if "malayalam" not in detected_languages:
                detected_languages.append("malayalam")
            confidence += 0.5  # High confidence for news content
        
        # Boost confidence for multiple language indicators
        if len(detected_languages) > 0:
            confidence += 0.2
            
        confidence = min(confidence, 1.0)
        
        # Determine primary language
        primary_language = detected_languages[0] if detected_languages else "unknown"
        
        # Enhanced metadata
        metadata = {
            "language_detected": primary_language,
            "all_languages": detected_languages,
            "needs_special_tokenization": True,
            "content_type": "news" if malayalam_news_count > 0 else "general",
            "malayalam_news_indicators": malayalam_news_count,
            "processing_hints": self._get_language_processing_hints(primary_language)
        }
        
        return ChallengeDetection(
            challenge_type=ChallengeType.MULTILINGUAL_QA,
            confidence=confidence,
            detected_patterns=patterns_found,
            suggested_approach=f"Use enhanced multilingual processing for {primary_language} content with specialized tokenization and cultural context awareness",
            metadata=metadata
        )
    
    def _get_language_processing_hints(self, language: str) -> Dict[str, str]:
        """Get processing hints for specific languages"""
        hints = {
            "malayalam": {
                "script": "Malayalam script (Brahmic)",
                "direction": "left-to-right",
                "common_domains": "news, government, business",
                "key_terms": "Trump (ട്രംപ്), tariff (ശുല്ക്കം), US (യുഎസ്)",
                "processing_note": "Pay attention to compound words and contextual meaning"
            },
            "hindi": {
                "script": "Devanagari script",
                "direction": "left-to-right",
                "common_domains": "general, official, media",
                "processing_note": "Handle conjunct consonants carefully"
            },
            "unknown": {
                "script": "Unknown",
                "direction": "unknown",
                "processing_note": "Use general multilingual processing"
            }
        }
        
        return hints.get(language, hints["unknown"])
    
    def _detect_complex_reasoning(self, questions: str) -> ChallengeDetection:
        """Detect complex reasoning requirements"""
        patterns_found = []
        confidence = 0.0
        
        for pattern in self.reasoning_patterns:
            if re.search(pattern, questions, re.IGNORECASE):
                patterns_found.append(pattern)
                confidence += 0.2
        
        # Check for question complexity indicators
        question_count = len(questions.split('?'))
        if question_count > 3:
            patterns_found.append("multiple_questions")
            confidence += 0.2
        
        # Check for conditional logic
        if re.search(r'\b(if|when|unless|provided|given)\b.*\b(then|should|will)\b', questions, re.IGNORECASE):
            patterns_found.append("conditional_logic")
            confidence += 0.3
        
        confidence = min(confidence, 1.0)
        
        return ChallengeDetection(
            challenge_type=ChallengeType.COMPLEX_REASONING,
            confidence=confidence,
            detected_patterns=patterns_found,
            suggested_approach="Use enhanced RAG with step-by-step reasoning and multiple context retrieval",
            metadata={
                "needs_step_by_step": True,
                "multi_turn_reasoning": True
            }
        )
    
    def get_primary_challenge(self, detections: List[ChallengeDetection]) -> ChallengeDetection:
        """Get the primary challenge type with highest confidence"""
        if not detections:
            return ChallengeDetection(
                challenge_type=ChallengeType.STANDARD_RAG,
                confidence=1.0,
                detected_patterns=["default"],
                suggested_approach="Use standard RAG pipeline",
                metadata={}
            )
        
        return detections[0]
    
    def should_use_geographic_solver(self, detections: List[ChallengeDetection]) -> bool:
        """Determine if geographic challenge solver should be used"""
        for detection in detections:
            if detection.challenge_type == ChallengeType.GEOGRAPHIC_PUZZLE and detection.confidence > 0.5:
                return True
            if detection.challenge_type == ChallengeType.FLIGHT_NUMBER and detection.confidence > 0.7:
                return True
        return False
    
    def should_use_web_client(self, detections: List[ChallengeDetection]) -> bool:
        """Determine if web client should be used"""
        for detection in detections:
            if detection.challenge_type in [ChallengeType.WEB_SCRAPING, ChallengeType.SECRET_TOKEN]:
                if detection.confidence > 0.5:
                    return True
        return False


# Singleton instance
_challenge_detector = None

def get_challenge_detector() -> ChallengeDetector:
    """Get or create challenge detector instance"""
    global _challenge_detector
    if _challenge_detector is None:
        _challenge_detector = ChallengeDetector()
    return _challenge_detector