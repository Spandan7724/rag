#!/usr/bin/env python3
"""
LLM-Powered Challenge Solver - Let the LLM do all the reasoning
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from app.services.web_client import WebClient
from app.services.enhanced_answer_generator import get_enhanced_answer_generator
from app.services.cache_manager import get_cache_manager


@dataclass 
class DataSource:
    """A source of data for the LLM"""
    name: str
    data: Any
    source_type: str  # "document", "api", "computed"
    confidence: float = 1.0
    processing_time: float = 0.0


@dataclass
class LLMSolverResult:
    """Result from LLM-powered solving"""
    answer: str
    reasoning: str
    data_sources_used: List[str]
    processing_time: float
    confidence: float
    success: bool
    error: Optional[str] = None


class LLMChallengeSolver:
    """
    LLM-powered generalized challenge solver
    
    Philosophy: Feed the LLM ALL available data and let it figure out the solution
    Benefits:
    - Truly generalized (works for ANY challenge)  
    - Self-adapting (LLM improves its reasoning over time)
    - Simple architecture (no complex step orchestration)
    - Fast (single LLM call vs multiple API round trips)
    """
    
    def __init__(self):
        self.answer_generator = get_enhanced_answer_generator()
        self.cache_manager = get_cache_manager()
        self.web_client = None
    
    async def solve_challenge(
        self,
        question: str,
        document_content: Optional[str] = None,
        document_id: Optional[str] = None,
        additional_context: Dict[str, Any] = None,
        enable_cache: bool = True
    ) -> LLMSolverResult:
        """
        Solve any challenge by gathering data and letting LLM reason
        
        Args:
            question: The challenge question
            document_content: Available document content
            document_id: Document ID if available
            additional_context: Extra context (team_id, etc.)
            enable_cache: Whether to use caching
            
        Returns:
            LLM solver result with answer and reasoning
        """
        start_time = time.time()
        
        # Check cache first
        if enable_cache:
            cache_key = self._get_cache_key(question, document_id, additional_context)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                cached_result.processing_time = time.time() - start_time
                return cached_result
        
        try:
            # Step 1: Gather ALL available data sources
            data_sources = await self._gather_data_sources(
                question, document_content, document_id, additional_context
            )
            
            # Step 2: Let LLM solve with all data
            result = await self._solve_with_llm(question, data_sources)
            result.processing_time = time.time() - start_time
            
            # Cache successful results
            if enable_cache and result.success:
                await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            return LLMSolverResult(
                answer="",
                reasoning=f"Challenge solving failed: {str(e)}",
                data_sources_used=[],
                processing_time=time.time() - start_time,
                confidence=0.0,
                success=False,
                error=str(e)
            )
    
    async def _gather_data_sources(
        self,
        question: str,
        document_content: Optional[str],
        document_id: Optional[str], 
        additional_context: Dict[str, Any]
    ) -> List[DataSource]:
        """
        Gather ALL potentially useful data sources in parallel
        """
        data_sources = []
        tasks = []
        
        # Always include the question context
        if additional_context:
            data_sources.append(DataSource(
                name="context",
                data=additional_context,
                source_type="context"
            ))
        
        # Document content (highest priority)
        if document_content:
            data_sources.append(DataSource(
                name="document",
                data=document_content,
                source_type="document",
                confidence=0.9
            ))
        
        # Parallel data gathering tasks
        if self._should_try_api_calls(question):
            # Try common API endpoints that might be relevant
            tasks.append(self._try_get_user_city(additional_context))
            tasks.append(self._try_get_secret_token(additional_context))
            
        # Add web scraping if URL detected
        if self._has_url(question):
            tasks.append(self._try_web_scraping(question))
        
        # Execute data gathering tasks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, DataSource):
                    data_sources.append(result)
                elif isinstance(result, Exception):
                    print(f"Data gathering failed: {result}")
        
        return data_sources
    
    async def _solve_with_llm(
        self,
        question: str,
        data_sources: List[DataSource]
    ) -> LLMSolverResult:
        """
        Let the LLM solve the challenge with all available data
        """
        # Construct comprehensive prompt with all data
        prompt = self._build_comprehensive_prompt(question, data_sources)
        
        # Get LLM response
        try:
            # Use your existing answer generator
            llm_response = await self.answer_generator.generate_direct_answer(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent reasoning
            )
            
            # Parse response for answer and reasoning
            answer, reasoning = self._parse_llm_response(llm_response)
            
            return LLMSolverResult(
                answer=answer,
                reasoning=reasoning,
                data_sources_used=[ds.name for ds in data_sources],
                processing_time=0.0,  # Will be set by caller
                confidence=self._estimate_confidence(answer, data_sources),
                success=True
            )
            
        except Exception as e:
            return LLMSolverResult(
                answer="",
                reasoning=f"LLM processing failed: {str(e)}",
                data_sources_used=[ds.name for ds in data_sources],
                processing_time=0.0,
                confidence=0.0,
                success=False,
                error=str(e)
            )
    
    def _build_comprehensive_prompt(
        self,
        question: str,
        data_sources: List[DataSource]
    ) -> str:
        """
        Build a comprehensive prompt with all available data
        """
        prompt_parts = [
            "You are an intelligent problem solver. Analyze the question and all available data to provide the best answer.",
            "",
            f"QUESTION: {question}",
            "",
            "AVAILABLE DATA:"
        ]
        
        # Add each data source
        for i, source in enumerate(data_sources, 1):
            prompt_parts.extend([
                f"\n{i}. {source.name.upper()} ({source.source_type}):",
                f"{self._format_data_for_prompt(source.data)}",
                ""
            ])
        
        prompt_parts.extend([
            "INSTRUCTIONS:",
            "1. Analyze ALL available data sources",
            "2. Determine what information is needed to answer the question",
            "3. Use logical reasoning to connect the data to the question",
            "4. Provide a direct, specific answer",
            "5. If you need to make API calls or web requests, explain what you would call",
            "6. If the answer requires multiple steps, work through them logically",
            "",
            "RESPONSE FORMAT:",
            "REASONING: [Your step-by-step reasoning]",
            "ANSWER: [Direct answer to the question]",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    def _format_data_for_prompt(self, data: Any) -> str:
        """Format data appropriately for the LLM prompt"""
        if isinstance(data, str):
            # Truncate very long strings
            if len(data) > 2000:
                return data[:2000] + "... [truncated]"
            return data
        elif isinstance(data, dict):
            return str(data)
        elif isinstance(data, list):
            return str(data)
        else:
            return str(data)
    
    def _parse_llm_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract answer and reasoning"""
        lines = response.strip().split('\n')
        reasoning_lines = []
        answer_lines = []
        current_section = None
        
        for line in lines:
            if line.startswith("REASONING:"):
                current_section = "reasoning"
                reasoning_lines.append(line.replace("REASONING:", "").strip())
            elif line.startswith("ANSWER:"):
                current_section = "answer"
                answer_lines.append(line.replace("ANSWER:", "").strip())
            elif current_section == "reasoning":
                reasoning_lines.append(line)
            elif current_section == "answer":
                answer_lines.append(line)
        
        reasoning = "\n".join(reasoning_lines).strip()
        answer = "\n".join(answer_lines).strip()
        
        # Fallback if parsing fails
        if not answer:
            answer = response.strip()
            reasoning = "Direct response"
        
        return answer, reasoning
    
    def _estimate_confidence(self, answer: str, data_sources: List[DataSource]) -> float:
        """Estimate confidence in the answer based on data quality"""
        if not answer or "I don't know" in answer.lower():
            return 0.1
        
        # Higher confidence with more data sources
        base_confidence = min(0.5 + (len(data_sources) * 0.1), 0.9)
        
        # Higher confidence with document data
        has_document = any(ds.source_type == "document" for ds in data_sources)
        if has_document:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    # Helper methods for data gathering
    async def _try_get_user_city(self, context: Dict[str, Any]) -> Optional[DataSource]:
        """Try to get user's favorite city"""
        team_id = context.get("team_id", "2836") if context else "2836"
        
        try:
            if not self.web_client:
                self.web_client = WebClient()
            
            start_time = time.time()
            city = await self.web_client.hackrx_get_favorite_city(team_id)
            processing_time = time.time() - start_time
            
            if city:
                return DataSource(
                    name="user_favorite_city",
                    data={"city": city, "team_id": team_id},
                    source_type="api",
                    processing_time=processing_time
                )
        except Exception as e:
            print(f"Failed to get user city: {e}")
        
        return None
    
    async def _try_get_secret_token(self, context: Dict[str, Any]) -> Optional[DataSource]:
        """Try to get secret token"""
        team_id = context.get("team_id", "2836") if context else "2836"
        
        try:
            if not self.web_client:
                self.web_client = WebClient()
            
            start_time = time.time()
            token = await self.web_client.hackrx_get_secret_token(team_id)
            processing_time = time.time() - start_time
            
            if token:
                return DataSource(
                    name="secret_token",
                    data={"token": token, "team_id": team_id},
                    source_type="api",
                    processing_time=processing_time
                )
        except Exception as e:
            print(f"Failed to get secret token: {e}")
        
        return None
    
    async def _try_web_scraping(self, question: str) -> Optional[DataSource]:
        """Try web scraping if URL detected in question"""
        import re
        urls = re.findall(r'https?://[^\s]+', question)
        
        if urls:
            try:
                if not self.web_client:
                    self.web_client = WebClient()
                
                start_time = time.time()
                response = await self.web_client.fetch_url(urls[0])
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    return DataSource(
                        name="web_content",
                        data={"url": urls[0], "content": response.content[:1000]},
                        source_type="api",
                        processing_time=processing_time
                    )
            except Exception as e:
                print(f"Web scraping failed: {e}")
        
        return None
    
    def _should_try_api_calls(self, question: str) -> bool:
        """Determine if we should try API calls based on question"""
        api_indicators = [
            "flight", "token", "secret", "city", "favorite", 
            "api", "get", "retrieve", "fetch"
        ]
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in api_indicators)
    
    def _has_url(self, question: str) -> bool:
        """Check if question contains URLs"""
        import re
        return bool(re.search(r'https?://', question))
    
    # Caching methods
    def _get_cache_key(
        self,
        question: str,
        document_id: Optional[str],
        additional_context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for LLM solver result"""
        import hashlib
        
        cache_data = {
            "question": question.lower().strip(),
            "document_id": document_id,
        }
        
        if additional_context:
            cache_data["context"] = additional_context
        
        cache_str = str(sorted(cache_data.items()))
        return f"llm_solver:{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[LLMSolverResult]:
        """Get cached solver result"""
        # Implementation would use your cache manager
        return None
    
    async def _cache_result(self, cache_key: str, result: LLMSolverResult):
        """Cache solver result"""
        # Implementation would use your cache manager
        pass


# Singleton
_llm_solver = None

def get_llm_challenge_solver() -> LLMChallengeSolver:
    """Get singleton LLM challenge solver"""
    global _llm_solver
    if _llm_solver is None:
        _llm_solver = LLMChallengeSolver()
    return _llm_solver