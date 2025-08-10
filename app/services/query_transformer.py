"""
Query Transformer Service for Multi-Query Decomposition

This service uses LLMs to break down complex queries into multiple sub-queries
to improve retrieval coverage in RAG systems, particularly for policy documents.
"""

import time
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio

from app.core.config import settings
from app.services.copilot_provider import get_copilot_provider

logger = logging.getLogger(__name__)


@dataclass
class QueryTransformationResult:
    """Result from query transformation"""
    original_query: str
    sub_queries: List[str]
    transformation_successful: bool
    processing_time: float
    model_info: Dict[str, Any]
    error_message: Optional[str] = None


class QueryTransformer:
    """
    Transforms complex user queries into multiple focused sub-queries
    for better retrieval coverage in RAG systems.
    """
    
    def __init__(self):
        """Initialize the query transformer"""
        self.provider_type = settings.llm_provider
        self.model = settings.llm_model
        self.temperature = settings.transformation_temperature
        self.max_sub_queries = settings.max_sub_queries
        self.min_query_length = settings.min_query_length
        self.timeout = settings.query_transformation_timeout
        
        logger.info(f"Initialized Query Transformer with {self.provider_type} provider")
        
    def _should_transform_query(self, query: str) -> bool:
        """
        Determine if a query should be transformed based on complexity indicators
        
        Args:
            query: The user's query
            
        Returns:
            True if query should be transformed, False otherwise
        """
        if not settings.enable_query_transformation:
            return False
            
        if len(query.strip()) < self.min_query_length:
            return False
            
        # Enhanced complexity detection for insurance policy queries
        query_lower = query.lower()
        
        # Multi-part indicators (traditional approach)
        multi_part_indicators = [
            'compare', 'difference', 'versus', 'vs', 'both', 'either',
            'benefits and', 'process and', 'coverage and', 'terms and',
            'policy and', 'claim and', 'premium and', 'waiting and',
            'grace and', 'exclusion and', 'inclusion and'
        ]
        multi_part_score = sum(1 for indicator in multi_part_indicators if indicator in query_lower)
        
        # Only truly complex insurance patterns that indicate multi-part questions
        insurance_complex_patterns = [
            # Multi-concept policy questions
            'waiting period', 'grace period', 'pre-existing', 'claim settlement',
            'premium payment', 'policy term', 'sum insured',
            
            # Process-related questions (inherently multi-step)
            'how to claim', 'documents required', 'claim process', 'settlement process',
            'when can i', 'eligibility',
            
            # Comparative questions (need multiple data points)
            'what is covered', 'what is not covered', 'compare', 'difference',
            
            # Complex medical condition queries
            'pre-existing diseases', 'waiting period for', 'exclusion and',
            'benefits and', 'coverage and', 'process and'
        ]
        insurance_score = sum(1 for pattern in insurance_complex_patterns if pattern in query_lower)
        
        # Question complexity indicators
        question_complexity = [
            'what are the', 'how do', 'explain', 'describe', 'list', 'enumerate',
            'what conditions', 'which procedures', 'what benefits', 'how much'
        ]
        question_score = sum(1 for q in question_complexity if q in query_lower)
        
        # Balanced transformation logic for insurance queries
        # Transform if:
        # 1. Has 2+ multi-part indicators (definitely complex)
        # 2. Has 2+ insurance complexity patterns (multiple concepts)
        # 3. Has complex question structure AND multiple insurance terms
        # 4. Long queries (>60 chars) with insurance terms (likely complex)
        
        query_length = len(query.strip())
        
        return (multi_part_score >= 2 or 
                insurance_score >= 1 or 
                (question_score >= 1 and insurance_score >= 1) or
                (query_length > 80 and ('and' in query_lower or 'or' in query_lower)))
    
    def _create_transformation_prompt(self, query: str) -> str:
        """
        Create a prompt for LLM to decompose the query
        
        Args:
            query: The original user query
            
        Returns:
            Formatted prompt for query decomposition
        """
        return f"""You are a query decomposition expert for insurance policy document retrieval systems.

Your task is to break down complex queries into focused sub-queries that will help retrieve relevant information from policy documents.

RULES:
1. Break the query into 2-{self.max_sub_queries} specific, focused sub-questions
2. Each sub-query should target a distinct concept or aspect
3. Ensure sub-queries are comprehensive and don't lose important context
4. Maintain the original intent and scope
5. Use clear, specific language that will match policy document content

EXAMPLES:

Original: "Compare the hospitalization benefits and claim settlement process"
Sub-queries:
1. What are the hospitalization benefits covered in the policy?
2. What is the claim settlement process for hospitalization?
3. How do hospitalization benefits and claim procedures relate?

Original: "What are the waiting periods and grace periods for different benefits?"  
Sub-queries:
1. What are the waiting periods for different types of benefits?
2. What are the grace periods for premium payments and coverage?
3. How do waiting periods and grace periods differ in the policy?

Original: "Explain the exclusions and inclusions for pre-existing diseases"
Sub-queries:
1. What conditions are excluded for pre-existing diseases?
2. What coverage is included for pre-existing diseases?
3. How are pre-existing diseases defined in the policy?

Now decompose this query:
"{query}"

Respond with a JSON array of strings, each representing a focused sub-query:
["sub-query 1", "sub-query 2", ...]"""

    async def _call_llm_for_transformation(self, prompt: str) -> str:
        """
        Call the LLM to perform query transformation
        
        Args:
            prompt: The transformation prompt
            
        Returns:
            LLM response containing sub-queries
        """
        if self.provider_type == "copilot":
            provider = get_copilot_provider(model=self.model)
            response = await provider.generate_answer(
                prompt=prompt,
                temperature=self.temperature
            )
            if response.error:
                raise Exception(f"Copilot error: {response.error}")
            return response.content
        
        else:
            # Fallback to Copilot for unknown providers
            logger.warning(f"Unknown provider '{self.provider_type}', falling back to Copilot")
            response = await provider.generate_answer(prompt, temperature=self.temperature)
            if response.error:
                raise Exception(f"Copilot error: {response.error}")
            return response.content

    def _parse_sub_queries(self, llm_response: str) -> List[str]:
        """
        Parse LLM response to extract sub-queries
        
        Args:
            llm_response: Raw response from LLM
            
        Returns:
            List of parsed sub-queries
        """
        try:
            # Try to parse as JSON first
            if llm_response.strip().startswith('['):
                sub_queries = json.loads(llm_response.strip())
                if isinstance(sub_queries, list):
                    return [str(q).strip() for q in sub_queries if q.strip()]
            
            # Fallback: try to extract JSON from the response
            import re
            json_match = re.search(r'\[(.*?)\]', llm_response, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'
                sub_queries = json.loads(json_str)
                if isinstance(sub_queries, list):
                    return [str(q).strip() for q in sub_queries if q.strip()]
            
            # Final fallback: split by newlines and clean
            lines = llm_response.split('\n')
            sub_queries = []
            for line in lines:
                line = line.strip()
                # Remove numbering and quotes
                line = re.sub(r'^\d+\.\s*', '', line)
                line = line.strip('"\'')
                if line and len(line) > 10:  # Reasonable length check
                    sub_queries.append(line)
            
            return sub_queries[:self.max_sub_queries]  # Limit to max
            
        except Exception as e:
            logger.warning(f"Failed to parse sub-queries from LLM response: {e}")
            return []

    async def transform_query(self, query: str) -> QueryTransformationResult:
        """
        Transform a complex query into multiple focused sub-queries
        
        Args:
            query: The original user query
            
        Returns:
            QueryTransformationResult with sub-queries or original query
        """
        start_time = time.time()
        
        # Check if transformation is needed
        if not self._should_transform_query(query):
            return QueryTransformationResult(
                original_query=query,
                sub_queries=[query],  # Use original as single sub-query
                transformation_successful=False,
                processing_time=time.time() - start_time,
                model_info={"provider": "none", "reason": "transformation_not_needed"},
                error_message=None
            )
        
        try:
            # Create transformation prompt
            prompt = self._create_transformation_prompt(query)
            
            # Call LLM with timeout
            llm_response = await asyncio.wait_for(
                self._call_llm_for_transformation(prompt),
                timeout=self.timeout
            )
            
            # Parse sub-queries
            sub_queries = self._parse_sub_queries(llm_response)
            
            # Validate results
            if not sub_queries or len(sub_queries) < 2:
                # Transformation failed, use original query
                return QueryTransformationResult(
                    original_query=query,
                    sub_queries=[query],
                    transformation_successful=False,
                    processing_time=time.time() - start_time,
                    model_info={
                        "provider": self.provider_type,
                        "model": self.model,
                        "reason": "insufficient_sub_queries"
                    },
                    error_message="LLM did not generate enough sub-queries"
                )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Query transformation successful: {len(sub_queries)} sub-queries in {processing_time:.2f}s")
            logger.debug(f"Original: {query}")
            for i, sub_q in enumerate(sub_queries, 1):
                logger.debug(f"Sub-query {i}: {sub_q}")
            
            return QueryTransformationResult(
                original_query=query,
                sub_queries=sub_queries,
                transformation_successful=True,
                processing_time=processing_time,
                model_info={
                    "provider": self.provider_type,
                    "model": self.model,
                    "temperature": self.temperature
                },
                error_message=None
            )
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.warning(f"Query transformation timeout after {processing_time:.2f}s")
            
            return QueryTransformationResult(
                original_query=query,
                sub_queries=[query],
                transformation_successful=False,
                processing_time=processing_time,
                model_info={"provider": self.provider_type, "model": self.model},
                error_message=f"Transformation timeout after {self.timeout}s"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Query transformation failed: {str(e)}")
            
            return QueryTransformationResult(
                original_query=query,
                sub_queries=[query],
                transformation_successful=False,
                processing_time=processing_time,
                model_info={"provider": self.provider_type, "model": self.model},
                error_message=str(e)
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get query transformer statistics"""
        return {
            "provider": self.provider_type,
            "model": self.model,
            "max_sub_queries": self.max_sub_queries,
            "transformation_enabled": settings.enable_query_transformation,
            "min_query_length": self.min_query_length,
            "timeout": self.timeout
        }


# Singleton instance
_query_transformer = None


def get_query_transformer() -> QueryTransformer:
    """Get singleton query transformer instance"""
    global _query_transformer
    if _query_transformer is None:
        _query_transformer = QueryTransformer()
    return _query_transformer