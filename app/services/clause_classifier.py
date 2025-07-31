"""
Clause type classification system
"""
from typing import Dict, List, Set
from app.models.clause import ClauseType

class ClauseClassifier:
    """Classifies clauses based on content and context"""
    
    # Keywords for different clause types
    DEFINITION_KEYWORDS = {
        "means", "refers to", "defined as", "shall mean", "interpretation",
        "definition", "terminology", "glossary"
    }
    
    COVERAGE_KEYWORDS = {
        "covers", "covered", "includes", "benefits", "eligible", "entitled",
        "reimburse", "reimbursement", "pays", "payment", "benefit"
    }
    
    EXCLUSION_KEYWORDS = {
        "excludes", "excluded", "not covered", "except", "excluding",
        "limitation", "restrictions", "shall not", "does not cover"
    }
    
    CONDITION_KEYWORDS = {
        "subject to", "provided that", "if", "when", "unless", "requirements",
        "conditions", "terms", "stipulations", "prerequisites"
    }
    
    PROCEDURE_KEYWORDS = {
        "claims", "claim process", "submit", "application", "procedure",
        "steps", "process", "how to", "filing", "documentation"
    }
    
    BENEFIT_KEYWORDS = {
        "benefit", "allowance", "compensation", "sum insured", "limit",
        "maximum", "minimum", "amount", "percentage"
    }
    
    LIMITATION_KEYWORDS = {
        "limit", "maximum", "minimum", "up to", "not more than", "ceiling",
        "cap", "restricted to", "subject to limit"
    }
    
    # Enhanced section-based classification with fuzzy matching
    SECTION_TYPE_MAPPING = {
        # Definition sections
        "DEFINITIONS": ClauseType.DEFINITION,
        "DEFINITION": ClauseType.DEFINITION,
        
        # Coverage sections  
        "COVERAGE": ClauseType.COVERAGE,
        "COVER": ClauseType.COVERAGE,
        "OPERATIVE CLAUSE": ClauseType.COVERAGE,
        
        # Benefit sections
        "BENEFITS": ClauseType.BENEFIT,
        "BENEFIT": ClauseType.BENEFIT,
        "TABLE OF BENEFITS": ClauseType.BENEFIT,
        
        # Exclusion sections (case-insensitive, partial matching)
        "EXCLUSIONS": ClauseType.EXCLUSION,
        "EXCLUSION": ClauseType.EXCLUSION,
        "NOT COVERED": ClauseType.EXCLUSION,
        "ANNEXURE": ClauseType.EXCLUSION,  # Annexure lists are typically exclusions
        
        # Condition sections
        "CONDITIONS": ClauseType.CONDITION,
        "CONDITION": ClauseType.CONDITION,
        "TERMS": ClauseType.CONDITION,
        "GENERAL TERMS": ClauseType.CONDITION,
        "WAITING PERIOD": ClauseType.CONDITION,
        
        # Procedure sections
        "CLAIMS": ClauseType.PROCEDURE,
        "CLAIM PROCEDURE": ClauseType.PROCEDURE,
        "PROCEDURES": ClauseType.PROCEDURE,
        "PROCEDURE": ClauseType.PROCEDURE,
        
        # Limitation sections
        "LIMITATIONS": ClauseType.LIMITATION,
        "LIMITATION": ClauseType.LIMITATION,
        
        # General sections
        "GENERAL": ClauseType.GENERAL,
        "PREAMBLE": ClauseType.GENERAL,
        "GRIEVANCE": ClauseType.GENERAL,
        "REDRESSAL": ClauseType.GENERAL
    }
    
    @classmethod
    def classify_clause(cls, text: str, section_heading: str = "") -> ClauseType:
        """Classify a clause based on its content and section"""
        text_lower = text.lower()
        section_upper = section_heading.upper()
        
        # First, try section-based classification with partial matching
        for section_key, clause_type in cls.SECTION_TYPE_MAPPING.items():
            if section_key in section_upper:
                return clause_type
        
        # Additional context-aware overrides for specific cases
        if section_upper and "EXCLUSION" in section_upper or "NOT COVERED" in section_upper:
            return ClauseType.EXCLUSION
        if section_upper and "BENEFIT" in section_upper or "TABLE" in section_upper:
            return ClauseType.BENEFIT
        
        # Then, use keyword-based classification
        scores = {}
        
        # Score each clause type based on keyword matches
        scores[ClauseType.DEFINITION] = cls._calculate_keyword_score(
            text_lower, cls.DEFINITION_KEYWORDS
        )
        scores[ClauseType.COVERAGE] = cls._calculate_keyword_score(
            text_lower, cls.COVERAGE_KEYWORDS
        )
        scores[ClauseType.EXCLUSION] = cls._calculate_keyword_score(
            text_lower, cls.EXCLUSION_KEYWORDS
        )
        scores[ClauseType.CONDITION] = cls._calculate_keyword_score(
            text_lower, cls.CONDITION_KEYWORDS
        )
        scores[ClauseType.PROCEDURE] = cls._calculate_keyword_score(
            text_lower, cls.PROCEDURE_KEYWORDS
        )
        scores[ClauseType.BENEFIT] = cls._calculate_keyword_score(
            text_lower, cls.BENEFIT_KEYWORDS
        )
        scores[ClauseType.LIMITATION] = cls._calculate_keyword_score(
            text_lower, cls.LIMITATION_KEYWORDS
        )
        
        # Return the type with highest score, or GENERAL if no clear match
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return ClauseType.GENERAL
    
    @classmethod
    def _calculate_keyword_score(cls, text: str, keywords: Set[str]) -> float:
        """Calculate score based on keyword matches"""
        score = 0.0
        words = text.split()
        
        for keyword in keywords:
            if keyword in text:
                # Give higher score for exact matches
                if keyword in words:
                    score += 2.0
                else:
                    score += 1.0
        
        # Normalize by text length to avoid bias toward longer texts
        return score / max(len(words), 1)
    
    @classmethod
    def get_confidence_score(cls, text: str, predicted_type: ClauseType) -> float:
        """Get confidence score for the classification"""
        text_lower = text.lower()
        
        # Get keyword set for predicted type
        keyword_sets = {
            ClauseType.DEFINITION: cls.DEFINITION_KEYWORDS,
            ClauseType.COVERAGE: cls.COVERAGE_KEYWORDS,
            ClauseType.EXCLUSION: cls.EXCLUSION_KEYWORDS,
            ClauseType.CONDITION: cls.CONDITION_KEYWORDS,
            ClauseType.PROCEDURE: cls.PROCEDURE_KEYWORDS,
            ClauseType.BENEFIT: cls.BENEFIT_KEYWORDS,
            ClauseType.LIMITATION: cls.LIMITATION_KEYWORDS,
        }
        
        if predicted_type not in keyword_sets:
            return 0.5  # Medium confidence for GENERAL type
        
        score = cls._calculate_keyword_score(text_lower, keyword_sets[predicted_type])
        
        # Convert to confidence score (0-1)
        return min(score, 1.0)
    
    @classmethod
    def classify_with_confidence(cls, text: str, section_heading: str = "") -> tuple[ClauseType, float]:
        """Classify clause and return confidence score"""
        clause_type = cls.classify_clause(text, section_heading)
        confidence = cls.get_confidence_score(text, clause_type)
        return clause_type, confidence
    
    @classmethod
    def extract_key_terms(cls, text: str, clause_type: ClauseType) -> List[str]:
        """Extract key terms from text based on clause type"""
        text_lower = text.lower()
        key_terms = []
        
        # Get relevant keywords for the clause type
        keyword_sets = {
            ClauseType.DEFINITION: cls.DEFINITION_KEYWORDS,
            ClauseType.COVERAGE: cls.COVERAGE_KEYWORDS,
            ClauseType.EXCLUSION: cls.EXCLUSION_KEYWORDS,
            ClauseType.CONDITION: cls.CONDITION_KEYWORDS,
            ClauseType.PROCEDURE: cls.PROCEDURE_KEYWORDS,
            ClauseType.BENEFIT: cls.BENEFIT_KEYWORDS,
            ClauseType.LIMITATION: cls.LIMITATION_KEYWORDS,
        }
        
        if clause_type in keyword_sets:
            keywords = keyword_sets[clause_type]
            for keyword in keywords:
                if keyword in text_lower:
                    key_terms.append(keyword)
        
        return key_terms