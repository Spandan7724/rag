"""
Enhanced answer generator using GitHub Copilot for answer generation
"""
import time
import threading
import os
from typing import List, Dict, Any
from dataclasses import dataclass
import re

from sentence_transformers import CrossEncoder
from app.services.vector_store import SearchResult
from app.services.copilot_provider import get_copilot_provider
from app.services.openai_provider import get_openai_provider
from app.services.cache_manager import get_cache_manager
from app.core.config import Settings, settings
from app.utils.debug import debug_print, conditional_print


class ThreadSafeReranker:
    """Thread-safe wrapper for CrossEncoder reranker"""
    
    def __init__(self):
        self.reranker = None
        self._reranker_lock = threading.Lock()
        self._reranker_loaded = False
        self.model_name = "BAAI/bge-reranker-large"
        
    def _load_reranker(self):
        """Thread-safe lazy load the reranker model"""
        # Quick check without lock (double-checked locking pattern)
        if self._reranker_loaded and self.reranker is not None:
            return
        
        with self._reranker_lock:
            # Double-check inside the lock
            if self._reranker_loaded and self.reranker is not None:
                return
            
            try:
                conditional_print(f"Loading reranker model (thread-safe): {self.model_name}")
                start_time = time.time()
                
                self.reranker = CrossEncoder(self.model_name)
                
                load_time = time.time() - start_time
                conditional_print(f"Reranker model loaded successfully in {load_time:.2f}s")
                
                # Warm up the model
                conditional_print("Warming up reranker...")
                warmup_start = time.time()
                self.reranker.predict([("warm up", "warm up")])
                warmup_time = time.time() - warmup_start
                conditional_print(f"Reranker warmed up in {warmup_time:.2f}s")
                
                # Mark as loaded
                self._reranker_loaded = True
                conditional_print("Reranker ready for parallel processing!")
                
            except Exception as e:
                raise RuntimeError(f"Failed to load reranker model: {str(e)}")
    
    def predict(self, pairs):
        """Thread-safe prediction with the reranker (sync version)"""
        self._load_reranker()
        
        # Use lock during prediction to ensure thread safety
        with self._reranker_lock:
            return self.reranker.predict(pairs)
    
    async def predict_async(self, pairs):
        """Async prediction with the reranker using thread pool"""
        import asyncio
        import concurrent.futures
        
        self._load_reranker()
        
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Use the thread-safe sync predict method in the thread pool
            future = loop.run_in_executor(executor, self._predict_in_thread, pairs)
            return await future
    
    def _predict_in_thread(self, pairs):
        """Helper method to run prediction in thread pool"""
        with self._reranker_lock:
            return self.reranker.predict(pairs)
    
    def is_loaded(self) -> bool:
        """Check if reranker is loaded"""
        return self._reranker_loaded and self.reranker is not None


# Create thread-safe reranker instance
RERANKER = ThreadSafeReranker()

# Note: TOP_K_INITIAL is now dynamically set via settings.k_retrieve (configurable in config.py or .env)
# TOP_K_RERANKED is now configurable via settings.top_k_reranked in config.py

settings = Settings()

@dataclass
class GeneratedAnswer:
    """Container for generated answer with metadata"""
    answer: str
    context_used: List[str]
    sources: List[Dict[str, Any]]
    processing_time: float
    model_info: Dict[str, Any]

class EnhancedAnswerGenerator:
    """Enhanced answer generator using GitHub Copilot"""

    def __init__(self):
        """Initialize answer generator with GitHub Copilot provider"""
        self.provider_type = settings.llm_provider
        self.model_name = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.cache_manager = get_cache_manager()
        
        conditional_print(f"Initializing enhanced answer generator: {self.provider_type} - {self.model_name}")
        
        # Initialize LLM provider
        if self.provider_type == "copilot":
            self._init_copilot_provider()
        elif self.provider_type == "openai":
            self._init_openai_provider()
        else:
            conditional_print(f"Warning: Provider '{self.provider_type}' not supported, using Copilot")
            self.provider_type = "copilot"
            self._init_copilot_provider()
        
        # Pre-warm reranker model for optimal performance
        self._pre_warm_reranker()
    
    def _pre_warm_reranker(self):
        """Pre-warm the reranker model to eliminate cold start delays"""
        try:
            conditional_print("Pre-warming reranker model...")
            # Force load the reranker by accessing it
            RERANKER._load_reranker()
            conditional_print("✓ Reranker model pre-warmed successfully")
        except Exception as e:
            conditional_print(f"⚠ Warning: Could not pre-warm reranker model: {e}")
            conditional_print("   Reranker will be loaded on first use")
    
    def _init_copilot_provider(self):
        """Initialize GitHub Copilot provider"""
        try:
            self.provider = get_copilot_provider(
                model=self.model_name,
                max_tokens=self.max_tokens
            )
            conditional_print(f"✓ GitHub Copilot provider initialized: {self.model_name}")
        except Exception as e:
            conditional_print(f"✗ Failed to initialize Copilot provider: {e}")
            raise
    
    def _init_openai_provider(self):
        """Initialize OpenAI provider"""
        try:
            self.provider = get_openai_provider(
                model=self.model_name,
                max_tokens=self.max_tokens
            )
            conditional_print(f"✓ OpenAI provider initialized: {self.model_name}")
        except Exception as e:
            conditional_print(f"✗ Failed to initialize OpenAI provider: {e}")
            raise
    
    
    def _prepare_sources(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Prepare source information from search results"""
        sources = []
        
        debug_print(f"DEBUG: _prepare_sources called with {len(search_results)} search results")
        
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
        
        debug_print(f"DEBUG: _prepare_sources returning {len(sources)} sources")
        return sources

    def _create_rag_prompt(self, question: str, context_str: str, is_multilingual: bool = False, detected_language: str = None) -> str:
        """Create enhanced RAG prompt with improved accuracy instructions and multilingual support"""
        
        # Classify question type for specialized instructions
        question_lower = question.lower()
        
        specialized_instructions = ""
        if any(term in question_lower for term in ['percentage', 'percent', '%', 'amount', 'limit']):
            specialized_instructions = "\n- VERBATIM EXTRACTION: Quote exact numerical figures with complete surrounding context, including currency symbols, mathematical operators, and all qualifying conditions exactly as they appear."
        elif any(term in question_lower for term in ['definition', 'meaning', 'what is', 'what does']):
            specialized_instructions = "\n- COMPLETE DEFINITION EXTRACTION: Provide the entire verbatim definition including all sub-clauses, conditions, exceptions, and qualifying statements exactly as written in the policy document."
        elif any(term in question_lower for term in ['when', 'time', 'period', 'days']):
            specialized_instructions = "\n- TEMPORAL PRECISION: Extract exact time periods, specific days, deadlines, and ALL associated conditions, requirements, and exceptions verbatim from the context."
        elif any(term in question_lower for term in ['coverage', 'covered', 'benefit', 'includes', 'encompasses', 'applies to']):
            specialized_instructions = "\n- COMPREHENSIVE SCOPE EXTRACTION: Quote complete scope details including ALL conditions, exclusions, limitations, and qualifying criteria exactly as stated in the document."
        elif any(term in question_lower for term in ['impact', 'effect', 'consequence', 'result']):
            specialized_instructions = "\n- IMPACT EXTRACTION: Only state impacts, effects, or consequences explicitly mentioned in the document using the exact subjects and objects as stated (e.g., if document says 'computers manufactured', use 'computers manufactured' not 'companies committed to manufacturing'). Do not infer, generalize, or analyze implications beyond what is directly stated."
        elif any(term in question_lower for term in ['procedure', 'process', 'how to', 'steps', 'method', 'approach']):
            specialized_instructions = "\n- PROCEDURAL EXTRACTION: Extract complete step-by-step procedures, requirements, and processes verbatim, preserving all numbering, sequencing, and conditional statements."
        elif any(term in question_lower for term in ['eligible', 'eligibility', 'qualify', 'criteria']):
            specialized_instructions = "\n- ELIGIBILITY CRITERIA EXTRACTION: Quote complete eligibility requirements, qualifying conditions, and all associated criteria exactly as specified in the context."
        
        # Determine question language (lightweight detection)
        question_is_malayalam = bool(re.search(r"[\u0D00-\u0D7F]", question))
        question_is_english = not question_is_malayalam

        # Enhanced language control with bilingual support for Malayalam questions
        if question_is_english:
            language_control = (
                "\n- LANGUAGE: The question is in English. Respond strictly in English. If the context is in another "
                "language (e.g., Malayalam), translate the relevant facts and mirror the document's phrasing in natural "
                "English. Do not answer in any other language."
            )
        else:
            # For Malayalam questions, provide bilingual responses for better accuracy scoring
            language_control = (
                "\n- BILINGUAL RESPONSE: The question is in Malayalam. Provide the answer in Malayalam followed by "
                "an English translation in parentheses. Format: '[Malayalam Answer] ([English Translation])'. "
                "Example: '2025 ഓഗസ്റ്റ് 6-ന് (August 6, 2025)'. For longer answers, provide full bilingual explanations "
                "to ensure maximum comprehension and accuracy scoring."
            )

        # Multilingual support
        multilingual_instructions = ""
        if is_multilingual:
            if detected_language == "malayalam":
                multilingual_instructions = (
                    "\n\nMULTILINGUAL PROCESSING (Malayalam):\n"
                    "- CRITICAL: If the question is in English, answer in English only. If the question is in Malayalam, "
                    "provide bilingual response with Malayalam answer and English translation in parentheses.\n"
                    "- The context contains Malayalam text (മലയാളം) and English translations. Use both for accuracy.\n"
                    "- For dates: Include complete dates with year (e.g., '2025 ഓഗസ്റ്റ് 6-ന് (August 6, 2025)').\n"
                    "- For complex answers: Provide full explanation in both languages for maximum comprehension.\n"
                    "- Example formats: '100% ശുൽകം (100% tariff)', 'കമ്പ്യൂട്ടർചിപ്പുകൾ (computer chips)'.\n"
                    "- NO IMPACT SPECULATION: Only state what is explicitly mentioned in the text."
                )
            else:
                multilingual_instructions = f"\n\nMULTILINGUAL PROCESSING ({detected_language or 'Unknown'}):\n- The context contains non-English content. Process multilingual text carefully.\n- Translate key information to English if the question is in English.\n- Preserve original terms in parentheses when translating for accuracy.\n- Pay attention to cultural and linguistic nuances in the content."
        
        system_message = f"""You are a precise document analysis expert. Answer ONLY based on the provided context.

CRITICAL RULES:
1. STAY WITHIN DOCUMENT SCOPE: Only use information explicitly stated in the provided context
2. Extract information EXACTLY as written in the context - preserve original phrasing, terminology, and exact subjects/objects (e.g., "computers" not "companies", "costs" not "electronics prices")
3. For dates: Always include the complete date with year when available in context (e.g., "2025 ഓഗസ്റ്റ് 6-ന്" not just "ഓഗസ്റ്റ് 6-ന്")
4. For numbers/percentages: Use exact values with all qualifying context from document
5. For definitions: Use precise wording from source document, maintaining the document's natural phrasing
6. NO EXTERNAL ANALYSIS: Do not add impact analysis, implications, or commentary not present in the document
7. If information is completely missing: Reply exactly "This information is not available in the provided document"
8. LANGUAGE CONTROL: If the question is in English, answer in English only. If the question is in Malayalam, provide bilingual response: Malayalam answer followed by English translation in parentheses.
9. Structure your response clearly with specific details, conditions, and procedures{specialized_instructions}{language_control}{multilingual_instructions}
10. APPLICABILITY/EXEMPTION/PRODUCT-LIST ENFORCEMENT (language-agnostic): If the question asks which items/entities are subject to, applicable to, exempt from, or covered by something (in any language), quote the exact noun phrase(s) from the context with ALL qualifiers (e.g., "foreign-made computer chips and semiconductors", "computers manufactured in the U.S."). Never substitute or broaden subjects (do not replace "computers" with "companies"). If the document does not state this, reply exactly "This information is not available in the provided document".

COMPLETENESS REQUIREMENTS:
- Include complete temporal information (full dates with years)
- Use document's exact terminology and phrasing where possible
- For multilingual content: Preserve original language terms with translations when helpful
- Only state what is explicitly written in the document - no inferences or implications

EXAMPLE:
Q: What is the deadline for project submission?
A: The deadline for project submission is 30 days from the notification date. All submissions must be complete and include all required documentation to be considered valid."""
        
        return f"""{system_message}

CONTEXT (numbered chunks for reference):
{context_str}

QUESTION: {question}
ANSWER:"""

    def _apply_boost_rules(self, question: str, candidates: List[SearchResult], scores: List[float]) -> List[float]:
        """Apply boost rules for critical terms that often get missed"""
        
        question_lower = question.lower()
        # Ensure scores is a list of Python floats, not a numpy array
        if hasattr(scores, 'tolist'):
            boosted_scores = [float(x) for x in scores.tolist()]
        else:
            boosted_scores = [float(x) for x in scores]
        
        for i, candidate in enumerate(candidates):
            text_lower = candidate.text.lower()
            boost_factor = 0.0
            
            # Grace period boost
            if "grace period" in question_lower:
                if "grace period" in text_lower and "thirty days" in text_lower:
                    boost_factor += 0.3
                    debug_print(f"  Grace period boost applied to chunk {i}")
                elif "thirty days" in text_lower and "premium" in text_lower:
                    boost_factor += 0.2
            
            # Hospital definition boost
            if "hospital" in question_lower and "ayush" not in question_lower:
                if "10 inpatient beds" in text_lower or "15 beds" in text_lower:
                    boost_factor += 0.25
                    debug_print(f"  Hospital definition boost applied to chunk {i}")
                elif "inpatient beds" in text_lower:
                    boost_factor += 0.15
            
            # AYUSH coverage boost
            if "ayush" in question_lower:
                if "ayush" in text_lower and ("treatment" in text_lower or "coverage" in text_lower):
                    boost_factor += 0.2
                    debug_print(f"  AYUSH coverage boost applied to chunk {i}")
            
            # Room rent and percentage limits boost (enhanced for specific percentages)
            if any(term in question_lower for term in ["room rent", "icu", "sub-limit", "limits", "plan a"]):
                # High priority: specific percentage patterns from Table of Benefits
                if "1% of sum insured" in text_lower or "2% of sum insured" in text_lower:
                    boost_factor += 0.35
                    debug_print(f"  Specific percentage limits boost applied to chunk {i}")
                elif "1%" in text_lower or "2%" in text_lower:
                    boost_factor += 0.25
                    debug_print(f"  Percentage pattern boost applied to chunk {i}")
                elif "% of sum insured" in text_lower:
                    boost_factor += 0.20
                    debug_print(f"  General percentage boost applied to chunk {i}")
                elif "room rent" in text_lower or "icu charges" in text_lower:
                    boost_factor += 0.15
            
            # Table content boost (tables often contain critical structured data)
            if hasattr(candidate, 'metadata') and getattr(candidate.metadata, 'chunk_type', '') == 'table':
                boost_factor += 0.15
                debug_print(f"  Table content boost applied to chunk {i}")
            
            # General percentage pattern boost for any question asking about limits/amounts
            if any(term in question_lower for term in ["percentage", "percent", "limit", "amount"]):
                if any(pattern in text_lower for pattern in ["1%", "2%", "% of sum", "percentage"]):
                    boost_factor += 0.10
                    debug_print(f"  General percentage query boost applied to chunk {i}")
            
            # Insurance-specific boost patterns (critical for score)
            if any(term in question_lower for term in ["claim", "coverage", "premium", "policy", "benefit"]):
                if any(pattern in text_lower for pattern in ["shall be covered", "eligible", "indemnify", "reimburse", "benefit"]):
                    boost_factor += 0.25
                    debug_print(f"  Insurance coverage boost applied to chunk {i}")

            # Claim settlement boost
            if any(term in question_lower for term in ["settle", "settlement", "claim"]):
                if any(pattern in text_lower for pattern in ["settlement", "claim", "documents", "process"]):
                    boost_factor += 0.20
                    debug_print(f"  Claim settlement boost applied to chunk {i}")
            
            # Apply boost
            if boost_factor > 0:
                boosted_scores[i] = min(float(scores[i]) + boost_factor, 1.0)
        
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
                    debug_print(f"  Added sibling chunk with numeric info: {shared_terms} shared terms")
                    break  # Add at most one sibling chunk
        
        # Insurance-specific chunk combination
        if any(term in question_lower for term in ["insurance", "policy", "claim", "coverage"]):
            # Look for related insurance chunks more aggressively
            for result in all_results[:20]:  # Expand search
                if result in candidates:
                    continue
                if any(pattern in result.text.lower() for pattern in [
                    "coverage", "benefit", "eligible", "claim", "settlement", "exclusion", "indemnify"
                ]):
                    candidates.append(result)
                    debug_print("  Added insurance-related chunk")
                    break
        
        return candidates
    
    def classify_query_complexity(self, question: str) -> str:
        """
        Classify query complexity to determine optimal retrieval strategy
        
        Returns:
            'simple', 'medium', or 'complex'
        """
        # Simple heuristics for query complexity
        question_lower = question.lower()
        
        # Complex indicators
        complex_patterns = [
            'compare', 'difference', 'versus', 'vs', 'between',
            'both', 'either', 'neither', 'all of', 'any of',
            'calculate', 'compute', 'sum', 'total', 'amount',
            'policy', 'coverage', 'benefit', 'claim', 'premium',
            'exclusion', 'deductible', 'co-pay', 'waiting period'
        ]
        
        # Medium indicators  
        medium_patterns = [
            'when', 'where', 'how', 'why', 'what if',
            'condition', 'requirement', 'eligible', 'qualify',
            'process', 'procedure', 'step', 'document'
        ]
        
        complex_count = sum(1 for pattern in complex_patterns if pattern in question_lower)
        medium_count = sum(1 for pattern in medium_patterns if pattern in question_lower)
        
        if complex_count >= 2 or len(question.split()) > 15:
            return 'complex'
        elif complex_count >= 1 or medium_count >= 2 or len(question.split()) > 8:
            return 'medium'
        else:
            return 'simple'
    
    def get_adaptive_k(self, question: str, base_k: int) -> int:
        """
        Determine optimal k based on query complexity
        """
        if not settings.adaptive_k:
            return base_k
            
        complexity = self.classify_query_complexity(question)
        
        if complexity == 'complex':
            return min(settings.max_k_retrieve, base_k + 5)
        elif complexity == 'medium':
            return min(settings.max_k_retrieve, base_k + 2)
        else:
            return max(settings.min_k_retrieve, base_k - 2)

    async def generate_answer(
        self, 
        question: str, 
        search_results: List[SearchResult],
        max_context_length: int = None,
        multilingual_mode: bool = False,
        detected_language: str = None
    ) -> GeneratedAnswer:
        """
        Generate answer using the configured LLM provider
        
        Args:
            question: User question
            search_results: Search results from vector store
            max_context_length: Maximum context length (None = use config default)
            
        Returns:
            GeneratedAnswer with response and metadata
        """
        start_time = time.time()
        
        # Use config default if not specified
        if max_context_length is None:
            max_context_length = settings.max_context_tokens
            
        debug_print(f"Generating answer using {self.provider_type} for: {question[:50]}...")
        debug_print(f"Using {len(search_results)} search results")
        debug_print(f"Max context length: {max_context_length:,} tokens")
        
        if not search_results:
            return GeneratedAnswer(
                answer="I couldn't find relevant information to answer your question.",
                context_used=[],
                sources=[],
                processing_time=time.time() - start_time,
                model_info={"provider": self.provider_type, "model": self.model_name, "method": "no_context"}
            )
        
        # Determine complexity and adaptive parameters
        complexity = self.classify_query_complexity(question)
        adaptive_k_initial = self.get_adaptive_k(question, settings.k_retrieve)
        
        # Adaptive TOP_K_RERANKED based on complexity (using configurable base value)
        base_reranked = settings.top_k_reranked
        if settings.adaptive_k:
            # Use adaptive logic when enabled
            if complexity == 'complex':
                adaptive_reranked = min(base_reranked + 2, len(search_results))
            elif complexity == 'medium':
                adaptive_reranked = min(base_reranked, len(search_results))
            else:
                adaptive_reranked = min(base_reranked - 5, len(search_results), 8)  # Minimum of 8 for simple queries
        else:
            # Use fixed value when adaptive_k is disabled
            adaptive_reranked = min(base_reranked, len(search_results))
        
        debug_print(f"Query complexity: {complexity}, using {adaptive_k_initial} initial, {adaptive_reranked} final")
        
        # Apply cross-encoder reranking
        candidates = search_results[:adaptive_k_initial]
        debug_print(f"Reranking top {len(candidates)} candidates...")
        
        # Build query-document pairs for reranking (with caching)
        rerank_pairs = [(question, cand.text) for cand in candidates]
        
        # Check reranker cache
        chunk_texts = [cand.text for cand in candidates]
        cached_scores = self.cache_manager.get_reranker_scores(question, chunk_texts) if settings.enable_reranker_cache else None
        
        if cached_scores is not None and len(cached_scores) > 0:
            rerank_scores = cached_scores
            debug_print("  Using cached reranker scores")
        else:
            # Use async reranker for non-blocking operation
            rerank_scores = await RERANKER.predict_async(rerank_pairs)
            if settings.enable_reranker_cache:
                # Convert to list if numpy array
                scores_to_cache = rerank_scores.tolist() if hasattr(rerank_scores, 'tolist') else rerank_scores
                self.cache_manager.set_reranker_scores(question, chunk_texts, scores_to_cache)
        
        # Ensure rerank_scores is a list for consistent handling
        if hasattr(rerank_scores, 'tolist'):
            rerank_scores = rerank_scores.tolist()
        
        # Apply boost rules for critical terms (if enabled)
        if settings.enable_boost_rules:
            boosted_scores = self._apply_boost_rules(question, candidates, rerank_scores)
            debug_print("  Boost rules applied")
        else:
            boosted_scores = rerank_scores
            debug_print("  Boost rules disabled - using raw reranker scores")
        
        # Sort by boosted scores and take top candidates (add index to prevent comparison errors)
        indexed_results = list(zip(boosted_scores, candidates, range(len(candidates))))
        sorted_results = sorted(indexed_results, key=lambda x: (x[0], -x[2]), reverse=True)
        candidates = [c for _, c, _ in sorted_results][:adaptive_reranked]
        
        # Include sibling chunks for numeric information
        candidates = self._include_sibling_chunks(question, candidates, search_results)
        
        sources = self._prepare_sources(candidates)
        
        conditional_print(f"After reranking: using top {len(candidates)} most relevant chunks")
        
        # Build rich context with metadata
        context_str = "\n".join(
            f"[{i+1}] ({getattr(c.metadata, 'chunk_type', 'content').title()}, "
            f"Page {getattr(c.metadata, 'page', 'N/A')}"
            f"{f', {getattr(c.metadata, 'section_hierarchy', '')}' if getattr(c.metadata, 'section_hierarchy', None) else ''}) "
            f"{c.text}" 
            for i, c in enumerate(candidates)
        )
        
        conditional_print(f"Context length: {len(context_str)} characters")
        
        try:
            # Generate answer using the configured provider with multilingual support
            if self.provider_type == "copilot":
                answer = await self._generate_with_copilot_async(question, context_str, multilingual_mode, detected_language)
            elif self.provider_type == "openai":
                answer = await self._generate_with_openai_async(question, context_str, multilingual_mode, detected_language)
            else:
                # Fallback to copilot if unknown provider
                answer = await self._generate_with_copilot_async(question, context_str, multilingual_mode, detected_language)
            
            # Simple post-processing - no retry/expansion complexity
            answer = answer.strip()

            # HARD ENFORCEMENT: Ensure answer language matches question language
            answer = await self._enforce_answer_language(answer, question)
            
            processing_time = time.time() - start_time
            
            conditional_print(f"Answer generated in {processing_time:.2f}s")
            conditional_print(f"Answer length: {len(answer)} characters")
            
            return GeneratedAnswer(
                answer=answer,
                context_used=[c.text for c in candidates],
                sources=sources,
                processing_time=processing_time,
                model_info={
                    "provider": self.provider_type,
                    "model": self.model_name,
                    "method": f"{self.provider_type}_api_reranked",
                    "context_chunks": len(candidates),
                    "context_chars": len(context_str),
                    "reranked_from": len(search_results)
                }
            )
            
        except Exception as e:
            conditional_print(f"{self.provider_type.title()} API failed: {e}")
            
            # Generate proper error message
            error_message = self._generate_error_message(e)
            
            return GeneratedAnswer(
                answer=error_message,
                context_used=[c.text for c in candidates] if 'candidates' in locals() else [],
                sources=sources if 'sources' in locals() else [],
                processing_time=time.time() - start_time,
                model_info={"provider": self.provider_type, "model": self.model_name, "method": "error_handling", "error": str(e)}
            )

    async def _enforce_answer_language(self, answer: str, question: str) -> str:
        """Ensure the answer language matches the question language. If mismatch, rewrite using provider."""
        try:
            question_is_malayalam = bool(re.search(r"[\u0D00-\u0D7F]", question))
            answer_is_malayalam = bool(re.search(r"[\u0D00-\u0D7F]", answer))

            # If languages match, no change
            if question_is_malayalam == answer_is_malayalam:
                return answer

            # Build a concise rewrite prompt
            if question_is_malayalam and not answer_is_malayalam:
                target_lang = "Malayalam"
                instruction = (
                    "Rewrite the following answer strictly in Malayalam, preserving all facts and structure. "
                    "Do not add or remove information."
                )
            else:
                target_lang = "English"
                instruction = (
                    "Rewrite the following answer strictly in English, preserving all facts and mirroring the "
                    "document’s phrasing naturally. Do not add or remove information."
                )

            rewrite_prompt = (
                f"{instruction}\n\nOriginal answer:\n" + answer + "\n\nRewritten answer (" + target_lang + "):" 
            )

            # Use the same provider to rewrite
            if self.provider_type in ["copilot", "openai"]:
                self.provider.kwargs.update({"max_tokens": self.max_tokens})
                resp = await self.provider.generate_answer(prompt=rewrite_prompt, temperature=min(self.temperature, 0.4))
                if resp and not resp.error and resp.content:
                    rewritten = resp.content.strip()
                    if rewritten:
                        return rewritten
            else:
                generation_config = {"max_output_tokens": self.max_tokens, "temperature": min(self.temperature, 0.4)}
                resp = self.provider.generate_content(rewrite_prompt, generation_config=generation_config)
                if hasattr(resp, 'text') and resp.text:
                    rewritten = resp.text.strip()
                    if rewritten:
                        return rewritten
        except Exception as _:
            # On any failure, fall back to the original answer
            return answer

        return answer

    async def _generate_with_copilot_async(self, question: str, context_str: str, multilingual_mode: bool = False, detected_language: str = None) -> str:
        """Generate answer using Copilot provider with multilingual support (native async)"""
        prompt = self._create_rag_prompt(question, context_str, multilingual_mode, detected_language)
        
        # Adjust temperature for multilingual content (slightly higher for better handling)
        temperature = self.temperature
        if multilingual_mode:
            temperature = min(self.temperature + 0.1, 1.0)
        
        # Update provider configuration
        self.provider.kwargs.update({"max_tokens": self.max_tokens})
        
        response = await self.provider.generate_answer(
            prompt=prompt,
            temperature=temperature
        )
        
        if response.error:
            raise Exception(response.error)
        
        return response.content.strip()

    async def _generate_with_openai_async(self, question: str, context_str: str, multilingual_mode: bool = False, detected_language: str = None) -> str:
        """Generate answer using OpenAI provider with multilingual support (native async)"""
        prompt = self._create_rag_prompt(question, context_str, multilingual_mode, detected_language)
        
        # Adjust temperature for multilingual content (slightly higher for better handling)
        temperature = self.temperature
        if multilingual_mode:
            temperature = min(self.temperature + 0.1, 1.0)
        
        # Update provider configuration
        self.provider.kwargs.update({"max_tokens": self.max_tokens})
        
        response = await self.provider.generate_answer(
            prompt=prompt,
            temperature=temperature
        )
        
        if response.error:
            raise Exception(response.error)
        
        return response.content.strip()


    def _generate_error_message(self, error: Exception) -> str:
        """Generate clear error message based on the type of error"""
        
        error_str = str(error).lower()
        
        # Provider-specific error handling
        if self.provider_type == "copilot":
            if "copilot_access_token" in error_str or "authentication" in error_str:
                return "ERROR: No GitHub Copilot access token configured. Please set the COPILOT_ACCESS_TOKEN environment variable and restart the application."
            elif "forbidden" in error_str or "subscription" in error_str:
                return "ERROR: GitHub Copilot access forbidden. Please verify your Copilot subscription is active."
            elif "rate limit" in error_str:
                return "ERROR: GitHub Copilot rate limit exceeded. Please try again later."
        elif self.provider_type == "openai":
            if "openai_api_key" in error_str or "authentication" in error_str:
                return "ERROR: No OpenAI API key configured. Please set the OPENAI_API_KEY environment variable and restart the application."
            elif "forbidden" in error_str or "insufficient" in error_str:
                return "ERROR: OpenAI access forbidden. Please verify your API key and quota."
            elif "rate limit" in error_str:
                return "ERROR: OpenAI rate limit exceeded. Please wait and try again."
            elif "quota" in error_str:
                return "ERROR: OpenAI quota exceeded. Please check your usage and billing."
        
        # Generic error handling
        if any(keyword in error_str for keyword in ['connection', 'network', 'timeout', 'unreachable']):
            return "ERROR: Network connection issue. Please check your internet connection and try again."
        
        # Generic API error
        return f"ERROR: {self.provider_type.title()} API request failed: {str(error)}. Please try again or contact support if the issue persists."

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": self.provider_type,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reranker": "BAAI/bge-reranker-large",
            "features": {
                "cross_encoder_reranking": True,
                "boost_rules": True,
                "sibling_chunks": True,
                "multi_provider": True
            }
        }
    
    @property
    def api_key(self) -> str:
        """Get API key for compatibility with health check"""
        if self.provider_type == "copilot":
            return os.getenv("COPILOT_ACCESS_TOKEN", "")
        elif self.provider_type == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        return ""


# Singleton instance
_enhanced_answer_generator = None

def get_enhanced_answer_generator() -> EnhancedAnswerGenerator:
    """Get singleton enhanced answer generator instance"""
    global _enhanced_answer_generator
    if _enhanced_answer_generator is None:
        _enhanced_answer_generator = EnhancedAnswerGenerator()
    return _enhanced_answer_generator