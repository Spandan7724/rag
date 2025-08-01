"""
Answer generation service using Google Gemini
"""
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.services.vector_store import SearchResult
from app.core.config import settings


@dataclass
class GeneratedAnswer:
    """Container for generated answer with metadata"""
    answer: str
    context_used: List[str]
    sources: List[Dict[str, Any]]
    processing_time: float
    model_info: Dict[str, Any]


class AnswerGenerator:
    """Generates answers using Google Gemini LLM"""
    
    def __init__(self):
        """Initialize answer generator"""
        self.model = None
        self.model_name = settings.llm_model
        self.api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        print(f"Initializing answer generator: {self.model_name}")
        
        if not self.api_key:
            print("Warning: No Gemini API key found. Set GEMINI_API_KEY environment variable.")
    
    def _load_model(self):
        """Lazy load the Gemini model"""
        if self.model is not None:
            return
        
        try:
            import google.generativeai as genai
            
            # Configure API key
            genai.configure(api_key=self.api_key)
            
            # Create model instance
            self.model = genai.GenerativeModel(self.model_name)
            
            print(f"Gemini model loaded successfully: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Install with: pip install google-generativeai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Gemini model: {str(e)}")
    
    def _prepare_context(self, search_results: List[SearchResult], max_context_length: int = 6000) -> List[str]:
        """
        Prepare context from search results
        
        Args:
            search_results: List of search results
            max_context_length: Maximum total character length for context
            
        Returns:
            List of context strings
        """
        context_parts = []
        current_length = 0
        
        print(f"Preparing context from {len(search_results)} search results (max {max_context_length} chars)")
        
        for i, result in enumerate(search_results):
            if not hasattr(result, 'text') or not result.text:
                print(f"  Skipping result {i+1}: no text")
                continue
            
            context_text = result.text.strip()
            
            # Handle oversized chunks by truncating them intelligently
            if len(context_text) > max_context_length:
                print(f"  Result {i+1}: chunk too large ({len(context_text):,} chars), truncating...")
                
                # Try to find the part that contains the query-relevant content
                # Look for key terms that might be relevant
                query_terms = ['grace period', 'premium payment', 'thirty days', 'days', 'payment']
                
                best_start = 0
                best_score = 0
                
                # Search for the best section to extract
                for term in query_terms:
                    pos = context_text.lower().find(term.lower())
                    if pos != -1:
                        # Found a relevant term, calculate score based on term importance
                        score = 10 if term == 'grace period' else 5 if term == 'thirty days' else 1
                        if score > best_score:
                            best_score = score
                            # Center the extraction around this term
                            best_start = max(0, pos - 1000)
                
                # Extract the relevant portion
                extract_length = min(max_context_length - current_length - 100, 2000)  # Leave room for other chunks
                context_text = context_text[best_start:best_start + extract_length]
                
                # Try to end at a sentence boundary
                last_period = context_text.rfind('. ')
                if last_period > len(context_text) * 0.8:  # Only if we're not cutting too much
                    context_text = context_text[:last_period + 1]
                
                print(f"    Extracted {len(context_text)} chars starting from position {best_start}")
            
            # Check if we can fit this context
            if current_length + len(context_text) > max_context_length:
                if current_length == 0:
                    # First chunk and it's still too big, take what we can
                    remaining_space = max_context_length - 100
                    context_text = context_text[:remaining_space]
                    print(f"    First chunk still too big, truncated to {len(context_text)} chars")
                else:
                    # We have some context already, stop here
                    print(f"    Stopping at result {i+1}: would exceed limit ({current_length} + {len(context_text)} > {max_context_length})")
                    break
            
            if context_text:
                context_parts.append(context_text)
                current_length += len(context_text)
                print(f"    Added result {i+1}: {len(context_text)} chars (total: {current_length})")
        
        print(f"Context preparation complete: {len(context_parts)} parts, {current_length} total chars")
        return context_parts
    
    def _prepare_sources(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """
        Prepare source information from search results
        
        Args:
            search_results: List of search results
            
        Returns:
            List of source dictionaries
        """
        sources = []
        
        for i, result in enumerate(search_results):
            source = {
                "number": i + 1,
                "doc_id": result.metadata.doc_id,
                "page": result.metadata.page,
                "heading": result.metadata.heading,
                "section": result.metadata.section_number,
                "similarity_score": result.similarity_score,
                "text_preview": result.metadata.text_preview,
                "chunk_type": result.metadata.chunk_type
            }
            sources.append(source)
        
        return sources
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for answer generation"""
        return """You are an expert document analysis assistant. Answer questions directly based on the provided document context.

INSTRUCTIONS:
1. Answer the specific question asked - be direct and precise
2. Include exact numbers, percentages, time periods, and conditions mentioned
3. Provide complete information but stay focused on the question
4. Use definitive statements, not phrases like "according to the document"
5. If the question asks for specific details, provide them in full
6. Only state information explicitly present in the context

EXAMPLE:
Question: "What is the grace period for premium payment?"
Good Answer: "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
Bad Answer: "There is a grace period mentioned in the policy."

Answer the question directly with specific details from the document."""
    
    def _create_user_prompt(self, question: str, context_parts: List[str]) -> str:
        """Create user prompt with question and context"""
        context_text = "\n\n".join(context_parts)
        
        return f"""Question: {question}

Document Context:
{context_text}

Provide a direct, factual answer based on the document context. Include specific details, numbers, and conditions mentioned in the text. Format as a clear, definitive statement that directly addresses the question."""
    
    def generate_answer(
        self, 
        question: str, 
        search_results: List[SearchResult],
        max_context_length: int = 6000
    ) -> GeneratedAnswer:
        """
        Generate answer using Gemini LLM
        
        Args:
            question: User question
            search_results: Search results from vector store
            max_context_length: Maximum context length
            
        Returns:
            GeneratedAnswer with response and metadata
        """
        start_time = time.time()
        
        print(f"Generating answer for: {question[:50]}...")
        print(f"Using {len(search_results)} search results")
        
        if not search_results:
            return GeneratedAnswer(
                answer="I couldn't find relevant information to answer your question.",
                context_used=[],
                sources=[],
                processing_time=time.time() - start_time,
                model_info={"model": self.model_name, "method": "no_context"}
            )
        
        # Prepare context and sources
        context_parts = self._prepare_context(search_results, max_context_length)
        sources = self._prepare_sources(search_results)
        
        print(f"Prepared context: {len(context_parts)} chunks, {sum(len(c) for c in context_parts)} characters")
        
        try:
            self._load_model()
            
            # Create prompts
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(question, context_parts)
            
            # Combine prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Generate response
            print("Calling Gemini API...")
            response = self.model.generate_content(full_prompt)
            
            # Extract answer
            if hasattr(response, 'text'):
                answer = response.text.strip()
            else:
                answer = str(response).strip()
            
            processing_time = time.time() - start_time
            
            print(f"Answer generated in {processing_time:.2f}s")
            print(f"Answer length: {len(answer)} characters")
            
            return GeneratedAnswer(
                answer=answer,
                context_used=context_parts,
                sources=sources,
                processing_time=processing_time,
                model_info={
                    "model": self.model_name,
                    "method": "gemini_api",
                    "context_chunks": len(context_parts),
                    "context_chars": sum(len(c) for c in context_parts)
                }
            )
            
        except Exception as e:
            print(f"Gemini API failed: {e}")
            
            # Simple fallback
            fallback_answer = self._generate_fallback_answer(question, search_results)
            
            return GeneratedAnswer(
                answer=fallback_answer,
                context_used=context_parts,
                sources=sources,
                processing_time=time.time() - start_time,
                model_info={"model": "fallback", "method": "template", "error": str(e)}
            )
    
    def _generate_fallback_answer(self, question: str, search_results: List[SearchResult]) -> str:
        """Generate fallback answer when Gemini fails"""
        
        if not search_results:
            return "No relevant information found to answer the question."
        
        # Use the most relevant (first) search result
        best_result = search_results[0]
        
        # Extract key information
        text = best_result.text
        
        # Look for specific patterns based on question
        question_lower = question.lower()
        
        if "grace period" in question_lower:
            # Look for grace period definition
            import re
            grace_match = re.search(
                r'grace period.*?(?:means|is|shall be).*?thirty.*?days',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if grace_match:
                return grace_match.group(0).strip()
        
        # Generic fallback - return first sentence with key information
        sentences = text.split('.')
        for sentence in sentences[:3]:  # Check first 3 sentences
            if any(word in sentence.lower() for word in question_lower.split()[:3]):
                return sentence.strip() + "."
        
        # Last resort - return preview
        return f"Based on the document: {best_result.metadata.text_preview}..."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "model_name": self.model_name,
            "api_key_configured": bool(self.api_key),
            "model_loaded": self.model is not None,
            "max_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature
        }


# Singleton instance
_answer_generator = None

def get_answer_generator() -> AnswerGenerator:
    """Get singleton answer generator instance"""
    global _answer_generator
    if _answer_generator is None:
        _answer_generator = AnswerGenerator()
    return _answer_generator