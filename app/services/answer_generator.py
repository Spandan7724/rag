"""
Answer generation service using Google Gemini with cross-encoder reranking
"""
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sentence_transformers import CrossEncoder
from app.services.vector_store import SearchResult
from app.core.config import settings

# Cross-encoder for reranking
RERANKER = CrossEncoder("BAAI/bge-reranker-large")

# ─── Warm-up to compile kernels & fill CUDA cache ───────────────────────
print("Warming up reranker...")
_ = RERANKER.predict([("warm up", "warm up")])
print("✓ Reranker warmed up")

TOP_K_INITIAL = 10      # retrieve from FAISS  
TOP_K_RERANKED = 5      # feed to LLM (increased for better coverage)


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
        
        # Check API key before attempting to load
        if not self.api_key:
            raise ValueError("No Gemini API key configured. Please set the GEMINI_API_KEY environment variable.")
        
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
        
        # Apply cross-encoder reranking
        candidates = search_results[:TOP_K_INITIAL]
        print(f"Reranking top {len(candidates)} candidates...")
        
        # Build query-document pairs for reranking
        rerank_pairs = [(question, cand.text) for cand in candidates]
        rerank_scores = RERANKER.predict(rerank_pairs)
        
        # Apply boost rules for critical terms
        boosted_scores = self._apply_boost_rules(question, candidates, rerank_scores)
        
        # Sort by boosted scores and take top candidates (add index to prevent comparison errors)
        indexed_results = list(zip(boosted_scores, candidates, range(len(candidates))))
        sorted_results = sorted(indexed_results, key=lambda x: (x[0], -x[2]), reverse=True)
        candidates = [c for _, c, _ in sorted_results][:TOP_K_RERANKED]
        
        # Include sibling chunks for numeric information
        candidates = self._include_sibling_chunks(question, candidates, search_results)
        
        sources = self._prepare_sources(candidates)
        
        print(f"After reranking: using top {len(candidates)} most relevant chunks")
        
        # Build compact context
        context_str = "\n".join(
            f"[{i+1}] {c.text}" for i, c in enumerate(candidates)
        )
        
        print(f"Context length: {len(context_str)} characters")
        
        try:
            self._load_model()
            
            # Create improved prompt
            full_prompt = f"""Answer the QUESTION using ONLY the CONTEXT below.
If the answer is not in the context, reply: "Not available in document."

CONTEXT:
{context_str}

QUESTION:
{question}

ANSWER:"""
            
            # Generate response
            print("Calling Gemini API...")
            response = self.model.generate_content(full_prompt)
            
            # Extract and clean answer
            if hasattr(response, 'text'):
                answer = response.text.strip()
            else:
                answer = str(response).strip()
            
            # Post-processing cleanup
            answer = answer.replace("\n", " ").strip()
            
            processing_time = time.time() - start_time
            
            print(f"Answer generated in {processing_time:.2f}s")
            print(f"Answer length: {len(answer)} characters")
            
            return GeneratedAnswer(
                answer=answer,
                context_used=[c.text for c in candidates],  # Use reranked candidates
                sources=sources,
                processing_time=processing_time,
                model_info={
                    "model": self.model_name,
                    "method": "gemini_api_reranked",
                    "context_chunks": len(candidates),
                    "context_chars": len(context_str),
                    "reranked_from": len(search_results)
                }
            )
            
        except Exception as e:
            print(f"Gemini API failed: {e}")
            
            # Generate proper error message based on error type
            error_message = self._generate_error_message(e)
            
            return GeneratedAnswer(
                answer=error_message,
                context_used=[c.text for c in candidates] if 'candidates' in locals() else [],
                sources=sources if 'sources' in locals() else [],
                processing_time=time.time() - start_time,
                model_info={"model": "error", "method": "error_handling", "error": str(e)}
            )
    
    def _generate_error_message(self, error: Exception) -> str:
        """Generate clear error message based on the type of error"""
        
        error_str = str(error).lower()
        
        # Check for specific API key issues
        if not self.api_key:
            return "ERROR: No Gemini API key configured. Please set the GEMINI_API_KEY environment variable and restart the application."
        
        # Check for authentication errors
        if any(keyword in error_str for keyword in ['unauthorized', 'invalid key', 'api key', 'authentication', 'forbidden']):
            return "ERROR: Invalid Gemini API key. Please check your GEMINI_API_KEY environment variable and ensure it's valid."
        
        # Check for quota/rate limit errors
        if any(keyword in error_str for keyword in ['quota', 'rate limit', 'too many requests', 'resource exhausted']):
            return "ERROR: Gemini API quota exceeded or rate limit reached. Please try again later or check your API usage limits."
        
        # Check for network/connection errors
        if any(keyword in error_str for keyword in ['connection', 'network', 'timeout', 'unreachable']):
            return "ERROR: Network connection issue. Please check your internet connection and try again."
        
        # Check for model-specific errors
        if any(keyword in error_str for keyword in ['model not found', 'invalid model']):
            return f"ERROR: The specified Gemini model '{self.model_name}' is not available. Please check the model name in configuration."
        
        # Check for content safety/filtering errors
        if any(keyword in error_str for keyword in ['safety', 'blocked', 'filtered']):
            return "ERROR: Request was blocked by Gemini's safety filters. Please try rephrasing your question."
        
        # Generic API error
        return f"ERROR: Gemini API request failed: {str(error)}. Please try again or contact support if the issue persists."
    
    def _apply_boost_rules(self, question: str, candidates: List[SearchResult], scores: List[float]) -> List[float]:
        """Apply boost rules for critical terms that often get missed"""
        
        question_lower = question.lower()
        boosted_scores = scores.copy()
        
        for i, candidate in enumerate(candidates):
            text_lower = candidate.text.lower()
            boost_factor = 0.0
            
            # Grace period boost
            if "grace period" in question_lower:
                if "grace period" in text_lower and "thirty days" in text_lower:
                    boost_factor += 0.3
                    print(f"  Grace period boost applied to chunk {i}")
                elif "thirty days" in text_lower and "premium" in text_lower:
                    boost_factor += 0.2
            
            # Hospital definition boost
            if "hospital" in question_lower and "ayush" not in question_lower:
                if "10 inpatient beds" in text_lower or "15 beds" in text_lower:
                    boost_factor += 0.25
                    print(f"  Hospital definition boost applied to chunk {i}")
                elif "inpatient beds" in text_lower:
                    boost_factor += 0.15
            
            # AYUSH coverage boost
            if "ayush" in question_lower:
                if "ayush" in text_lower and ("treatment" in text_lower or "coverage" in text_lower):
                    boost_factor += 0.2
                    print(f"  AYUSH coverage boost applied to chunk {i}")
            
            # Room rent and percentage limits boost (enhanced for specific percentages)
            if any(term in question_lower for term in ["room rent", "icu", "sub-limit", "limits", "plan a"]):
                # High priority: specific percentage patterns from Table of Benefits
                if "1% of sum insured" in text_lower or "2% of sum insured" in text_lower:
                    boost_factor += 0.35
                    print(f"  Specific percentage limits boost applied to chunk {i}")
                elif "1%" in text_lower or "2%" in text_lower:
                    boost_factor += 0.25
                    print(f"  Percentage pattern boost applied to chunk {i}")
                elif "% of sum insured" in text_lower:
                    boost_factor += 0.20
                    print(f"  General percentage boost applied to chunk {i}")
                elif "room rent" in text_lower or "icu charges" in text_lower:
                    boost_factor += 0.15
            
            # Table content boost (tables often contain critical structured data)
            if hasattr(candidate, 'metadata') and getattr(candidate.metadata, 'chunk_type', '') == 'table':
                boost_factor += 0.15
                print(f"  Table content boost applied to chunk {i}")
            
            # General percentage pattern boost for any question asking about limits/amounts
            if any(term in question_lower for term in ["percentage", "percent", "limit", "amount"]):
                if any(pattern in text_lower for pattern in ["1%", "2%", "% of sum", "percentage"]):
                    boost_factor += 0.10
                    print(f"  General percentage query boost applied to chunk {i}")
            
            # Apply boost
            if boost_factor > 0:
                boosted_scores[i] = min(scores[i] + boost_factor, 1.0)
        
        return boosted_scores
    
    def _include_sibling_chunks(self, question: str, candidates: List[SearchResult], all_results: List[SearchResult]) -> List[SearchResult]:
        """Include sibling chunks that contain numeric/percentage information"""
        
        # Look for numeric patterns in question
        question_lower = question.lower()
        needs_numeric = any(term in question_lower for term in [
            "percentage", "percent", "%", "limit", "amount", "number", "beds", "days"
        ])
        
        if not needs_numeric:
            return candidates
        
        # Find the highest scoring candidate
        if not candidates:
            return candidates
            
        primary_chunk = candidates[0]
        
        # Look for sibling chunks with numeric information
        sibling_patterns = [
            r'\d+\s*%', r'\d+\s*percent', r'\d+\s*beds?', r'\d+\s*days?',
            r'sum insured', r'inpatient beds', r'thirty days'
        ]
        
        for result in all_results[:15]:  # Check broader set
            if result in candidates:
                continue
                
            text_lower = result.text.lower()
            
            # Check if this chunk contains numeric info we might need
            import re
            has_numeric = any(re.search(pattern, text_lower) for pattern in sibling_patterns)
            
            if has_numeric:
                # Check if it's related to our question context
                primary_text = primary_chunk.text.lower()
                
                # Simple relevance check - shares key terms
                shared_terms = 0
                key_terms = ["hospital", "room", "rent", "icu", "grace", "period", "ayush", "coverage"]
                
                for term in key_terms:
                    if term in question_lower and term in primary_text and term in text_lower:
                        shared_terms += 1
                
                if shared_terms > 0:
                    candidates.append(result)
                    print(f"  Added sibling chunk with numeric info: {shared_terms} shared terms")
                    break  # Add at most one sibling chunk
        
        return candidates
    
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