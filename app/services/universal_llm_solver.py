#!/usr/bin/env python3
"""
Universal LLM Solver - Let the LLM decide EVERYTHING
"""
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from app.services.enhanced_answer_generator import get_enhanced_answer_generator
from app.services.universal_web_client import UniversalWebClient, ExtractionRule
from app.services.cache_manager import get_cache_manager


@dataclass
class UniversalSolverResult:
    """Result from universal solving"""
    answer: str
    processing_time: float
    success: bool
    reasoning: Optional[str] = None
    api_calls_made: List[str] = None
    error: Optional[str] = None


class UniversalLLMSolver:
    """
    Universal LLM-powered solver
    
    Philosophy:
    1. Give LLM ALL available context
    2. Let LLM decide what it needs to do
    3. If LLM requests actions, execute them
    4. Return LLM's final answer
    
    No hardcoded assumptions about challenge types, API calls, or data needs.
    """
    
    def __init__(self):
        self.answer_generator = get_enhanced_answer_generator()
        self.cache_manager = get_cache_manager()
    
    async def solve(
        self,
        question: str,
        document_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[str]] = None
    ) -> UniversalSolverResult:
        """
        Solve any question using pure LLM reasoning
        
        Args:
            question: The question to answer
            document_content: Any available document content
            context: Any additional context (team_id, URLs, etc.)
            available_tools: List of tools/APIs available to the LLM
            
        Returns:
            Universal solver result
        """
        start_time = time.time()
        
        try:
            # Step 1: Build comprehensive context for LLM
            full_context = self._build_complete_context(
                question, document_content, context, available_tools
            )
            
            # Step 2: Let LLM analyze and decide what to do
            llm_response = await self._get_llm_decision(full_context)
            
            # Step 3: Execute any actions LLM requested
            execution_results = await self._execute_llm_actions(llm_response)
            
            # Step 4: Get final answer from LLM with execution results
            final_answer = await self._get_final_answer(
                full_context, llm_response, execution_results
            )
            
            return UniversalSolverResult(
                answer=final_answer.get("answer", ""),
                processing_time=time.time() - start_time,
                success=True,
                reasoning=final_answer.get("reasoning"),
                api_calls_made=execution_results.get("calls_made", [])
            )
            
        except Exception as e:
            return UniversalSolverResult(
                answer="",
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _build_complete_context(
        self,
        question: str,
        document_content: Optional[str],
        context: Optional[Dict[str, Any]],
        available_tools: Optional[List[str]]
    ) -> str:
        """
        Build complete context for the LLM with everything available
        """
        context_parts = [
            "You are a highly intelligent problem solver. You can analyze any question and determine the best approach to answer it.",
            "",
            f"QUESTION: {question}",
            ""
        ]
        
        # Add document content if available
        if document_content:
            # Limit document length for context window
            if len(document_content) > 8000:
                doc_preview = document_content[:8000] + "... [content truncated]"
            else:
                doc_preview = document_content
                
            context_parts.extend([
                "DOCUMENT CONTENT:",
                doc_preview,
                ""
            ])
        
        # Add additional context
        if context:
            context_parts.extend([
                "ADDITIONAL CONTEXT:",
                json.dumps(context, indent=2),
                ""
            ])
        
        # Add available tools
        default_tools = [
            "web_scraping - fetch content from URLs",
            "api_calls - make HTTP requests to APIs",
            "document_search - search within provided documents",
            "data_extraction - extract structured data from text"
        ]
        
        tools_list = available_tools or default_tools
        context_parts.extend([
            "AVAILABLE TOOLS:",
        ])
        for tool in tools_list:
            context_parts.append(f"- {tool}")
        
        context_parts.extend([
            "",
            "INSTRUCTIONS:",
            "1. Analyze the question and available data",
            "2. Determine if you can answer directly from the given information",
            "3. If you need additional data, specify exactly what actions to take",
            "4. Provide your reasoning and final answer",
            "",
            "RESPONSE FORMAT:",
            "```json",
            "{",
            '  "analysis": "Your analysis of what is needed",',
            '  "can_answer_directly": true/false,',
            '  "actions_needed": [',
            '    {',
            '      "type": "api_call/web_scraping/other",',
            '      "description": "what to do",',
            '      "parameters": {"key": "value"}',
            '    }',
            '  ],',
            '  "direct_answer": "answer if you can provide it now",',
            '  "reasoning": "your step-by-step reasoning"',
            "}",
            "```"
        ])
        
        return "\n".join(context_parts)
    
    async def _get_llm_decision(self, context: str) -> Dict[str, Any]:
        """
        Get LLM's decision on how to proceed
        """
        # Use your existing answer generator with a dummy search result
        from app.services.vector_store import SearchResult, ChunkMetadata
        
        # Create dummy search result containing our context
        dummy_metadata = ChunkMetadata(
            doc_id="context",
            chunk_id=0,
            text_preview="LLM Context",
            token_count=len(context.split()),
            page=1,
            heading="Context",
            section_number="1",
            chunk_type="context",
            char_start=0,
            char_end=len(context)
        )
        
        dummy_search_result = SearchResult(
            chunk_index=0,
            similarity_score=0.0,
            text=context,
            metadata=dummy_metadata
        )
        
        response = await self.answer_generator.generate_answer(
            question="Analyze the following context and determine how to proceed:",
            search_results=[dummy_search_result],
            max_context_length=16000,
            multilingual_mode=False
        )
        
        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n```', response.answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                # Fallback - try to parse entire response as JSON
                return json.loads(response.answer)
        except json.JSONDecodeError:
            # Fallback to text analysis
            return {
                "analysis": "Could not parse JSON response",
                "can_answer_directly": True,
                "direct_answer": response.answer,
                "reasoning": "Fallback to direct response"
            }
    
    async def _execute_llm_actions(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute any actions the LLM requested
        """
        results = {"calls_made": [], "results": []}
        
        if not llm_response.get("actions_needed"):
            return results
        
        for action in llm_response["actions_needed"]:
            try:
                action_type = action.get("type", "")
                description = action.get("description", "")
                parameters = action.get("parameters", {})
                
                print(f"Executing action: {action_type} - {description}")
                
                if action_type == "web_scraping":
                    result = await self._execute_web_scraping(parameters)
                elif action_type == "api_call":
                    result = await self._execute_api_call(parameters)
                else:
                    result = {"error": f"Unknown action type: {action_type}"}
                
                results["calls_made"].append(f"{action_type}: {description}")
                results["results"].append({
                    "action": action,
                    "result": result
                })
                
            except Exception as e:
                results["results"].append({
                    "action": action,
                    "error": str(e)
                })
        
        return results
    
    async def _execute_web_scraping(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web scraping action using universal web client"""
        url = parameters.get("url")
        if not url:
            return {"error": "No URL provided for web scraping"}
        
        try:
            async with UniversalWebClient() as client:
                # Check if specific extraction patterns are provided
                extraction_patterns = parameters.get("extract", {})
                
                if extraction_patterns:
                    # Use pattern-based extraction
                    rules = client.create_extraction_rules_from_patterns(extraction_patterns)
                    result = await client.fetch_and_extract(url, rules)
                    
                    return {
                        "url": result.url,
                        "status_code": result.status_code,
                        "success": result.success,
                        "extracted_data": result.extracted_data,
                        "content_preview": result.raw_content[:500] + "..." if len(result.raw_content) > 500 else result.raw_content
                    }
                else:
                    # Basic fetch
                    result = await client.quick_fetch(url)
                    
                    return {
                        "url": result.url,
                        "status_code": result.status_code,
                        "success": result.success,
                        "content": result.raw_content[:2000] + "..." if len(result.raw_content) > 2000 else result.raw_content,
                        "content_type": result.content_type
                    }
        except Exception as e:
            return {"error": str(e), "url": url}
    
    async def _execute_api_call(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic API call using universal web client"""
        endpoint = parameters.get("endpoint")
        method = parameters.get("method", "GET")
        data = parameters.get("data", {})
        headers = parameters.get("headers", {})
        
        if not endpoint:
            return {"error": "No endpoint provided for API call"}
        
        try:
            async with UniversalWebClient() as client:
                result = await client.quick_fetch(
                    url=endpoint,
                    method=method,
                    headers=headers,
                    data=data if method.upper() != "GET" else None
                )
                
                return {
                    "endpoint": result.url,
                    "method": method,
                    "status_code": result.status_code,
                    "response": result.raw_content[:1000] + "..." if len(result.raw_content) > 1000 else result.raw_content,
                    "success": result.success,
                    "content_type": result.content_type
                }
        except Exception as e:
            return {"error": str(e), "endpoint": endpoint}
    
    async def _get_final_answer(
        self,
        original_context: str,
        llm_decision: Dict[str, Any],
        execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get final answer from LLM with all results
        """
        # If LLM said it could answer directly, return that
        if llm_decision.get("can_answer_directly") and llm_decision.get("direct_answer"):
            return {
                "answer": llm_decision["direct_answer"],
                "reasoning": llm_decision.get("reasoning", "Direct answer from initial analysis")
            }
        
        # Otherwise, give LLM the execution results and ask for final answer
        final_context_parts = [
            original_context,
            "",
            "EXECUTION RESULTS:",
            json.dumps(execution_results, indent=2),
            "",
            "Now provide your final answer based on all the information above.",
            "Format your response as:",
            "REASONING: [your reasoning]",
            "ANSWER: [final answer]"
        ]
        
        final_context = "\n".join(final_context_parts)
        
        # Use answer generator again
        from app.services.vector_store import SearchResult, ChunkMetadata
        
        dummy_metadata = ChunkMetadata(
            doc_id="final_context",
            chunk_id=0,
            text_preview="Final Context",
            token_count=len(final_context.split()),
            page=1,
            heading="Final Context",
            section_number="1",
            chunk_type="context",
            char_start=0,
            char_end=len(final_context)
        )
        
        dummy_search_result = SearchResult(
            chunk_index=0,
            similarity_score=0.0,
            text=final_context,
            metadata=dummy_metadata
        )
        
        response = await self.answer_generator.generate_answer(
            question="Provide the final answer based on the context and execution results:",
            search_results=[dummy_search_result],
            max_context_length=20000
        )
        
        # Parse the response
        lines = response.answer.split('\n')
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
        
        reasoning = "\n".join(reasoning_lines).strip() or "LLM processing completed"
        answer = "\n".join(answer_lines).strip() or response.answer
        
        return {
            "answer": answer,
            "reasoning": reasoning
        }


# Singleton
_universal_solver = None

def get_universal_llm_solver() -> UniversalLLMSolver:
    """Get singleton universal LLM solver"""
    global _universal_solver
    if _universal_solver is None:
        _universal_solver = UniversalLLMSolver()
    return _universal_solver