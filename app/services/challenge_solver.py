#!/usr/bin/env python3
"""
Challenge Solver Service
Multi-step reasoning engine for HackRX geographic puzzles and complex challenges
"""
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .web_client import WebClient
from .table_extractor import LandmarkMapping, get_table_extractor
from .vector_store import get_vector_store
from app.utils.debug import conditional_print
# Avoid circular import - rag_coordinator will be imported lazily when needed

class ChallengeStep(Enum):
    """Challenge solving steps"""
    INITIALIZE = "initialize"
    GET_CITY = "get_city"
    MAP_LANDMARK = "map_landmark"
    GET_FLIGHT = "get_flight"
    GET_TOKEN = "get_token"
    VALIDATE = "validate"
    COMPLETE = "complete"

@dataclass
class StepResult:
    """Result of a challenge step"""
    step: ChallengeStep
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0
    
@dataclass
class ChallengeContext:
    """Context for challenge solving"""
    document_id: str
    landmark_mappings: Dict[str, List[LandmarkMapping]] = field(default_factory=dict)
    city: Optional[str] = None
    landmarks: List[LandmarkMapping] = field(default_factory=list)
    flight_numbers: List[str] = field(default_factory=list)
    secret_token: Optional[str] = None
    steps_completed: List[StepResult] = field(default_factory=list)
    total_processing_time: float = 0.0

class ChallengeSolver:
    """
    Multi-step reasoning engine for geographic and challenge-based puzzles
    Handles the complete HackRX challenge workflow
    """
    
    def __init__(self):
        """Initialize challenge solver"""
        self.web_client = None
        self.table_extractor = get_table_extractor()
        self.vector_store = None
        # rag_coordinator will be imported lazily to avoid circular imports
    
    async def setup(self):
        """Initialize services"""
        self.web_client = WebClient()
        self.vector_store = get_vector_store()
        # rag_coordinator lazy loading handled in _ask_document_for_landmark
    
    async def load_landmark_mappings(self, document_id: str, document_text: str = None) -> Dict[str, List[LandmarkMapping]]:
        """
        Load landmark mappings from processed documents or direct text
        
        Args:
            document_id: ID of the challenge document
            document_text: Optional complete document text for direct processing
            
        Returns:
            Dictionary mapping cities to landmark data
        """
        mappings = {}
        
        try:
            # If complete document text is provided, use it directly (preferred)
            if document_text:
                print(f"Using direct document text for landmark extraction ({len(document_text)} chars)")
                
                # Extract mappings from complete text
                text_mappings = self.table_extractor.extract_landmark_mappings_from_text(document_text)
                
                # Create lookup dictionary
                mappings = self.table_extractor.create_lookup_dict(text_mappings)
                
                print(f"Direct extraction: {len(text_mappings)} mappings for {len(mappings)} cities")
                
            else:
                # Fallback to vector store approach (legacy)
                print(f"Falling back to vector store chunk reconstruction for {document_id}")
                
                if self.vector_store and document_id in self.vector_store.documents:
                    # Reconstruct from chunks
                    doc_chunks = []
                    for i in range(len(self.vector_store.chunk_texts)):
                        metadata = self.vector_store.chunk_metadata[i]
                        if metadata.doc_id == document_id:
                            doc_chunks.append(self.vector_store.chunk_texts[i])
                    
                    # Combine all chunks to get full document text
                    full_text = "\n".join(doc_chunks)
                    
                    # Extract mappings from reconstructed text
                    text_mappings = self.table_extractor.extract_landmark_mappings_from_text(full_text)
                    
                    # Create lookup dictionary
                    mappings = self.table_extractor.create_lookup_dict(text_mappings)
                    
                    print(f"Vector store extraction: {len(text_mappings)} mappings for {len(mappings)} cities")
            
        except Exception as e:
            print(f"Error loading landmark mappings: {e}")
        
        return mappings
    
    async def solve_geographic_challenge_direct(
        self, 
        document_id: str, 
        document_text: str, 
        hack_team: str = "2836"
    ) -> ChallengeContext:
        """
        Solve HackRX geographic challenge using direct document processing
        
        Args:
            document_id: ID of the challenge document
            document_text: Complete document text (no chunking/vector search)
            hack_team: Team identifier for API calls
            
        Returns:
            Complete challenge context with results
        """
        conditional_print(f"Starting direct geographic challenge solving for {document_id}")
        
        context = ChallengeContext(document_id=document_id)
        start_time = time.time()
        
        await self.setup()
        
        # Step 1: Initialize with direct text - Load landmark mappings directly
        step_result = await self._execute_step(
            ChallengeStep.INITIALIZE,
            lambda: self._step_initialize_direct(context, document_text)
        )
        context.steps_completed.append(step_result)
        
        if not step_result.success:
            context.total_processing_time = time.time() - start_time
            return context
        
        # Rest of the steps remain the same
        # Step 2: Get favorite city from API
        step_result = await self._execute_step(
            ChallengeStep.GET_CITY,
            lambda: self._step_get_city(context)
        )
        context.steps_completed.append(step_result)
        
        if not step_result.success:
            context.total_processing_time = time.time() - start_time
            return context
        
        # Step 3: Map city to landmarks
        step_result = await self._execute_step(
            ChallengeStep.MAP_LANDMARK,
            lambda: self._step_map_landmark(context)
        )
        context.steps_completed.append(step_result)
        
        if not step_result.success:
            context.total_processing_time = time.time() - start_time
            return context
        
        # Step 4: Get flight numbers based on landmarks
        step_result = await self._execute_step(
            ChallengeStep.GET_FLIGHT,
            lambda: self._step_get_flight(context)
        )
        context.steps_completed.append(step_result)
        
        # Step 5: Get secret token (parallel with flight)
        token_result = await self._execute_step(
            ChallengeStep.GET_TOKEN,
            lambda: self._step_get_token(context, hack_team)
        )
        context.steps_completed.append(token_result)
        
        # Step 6: Validate and complete
        final_result = await self._execute_step(
            ChallengeStep.COMPLETE,
            lambda: self._step_complete(context)
        )
        context.steps_completed.append(final_result)
        
        context.total_processing_time = time.time() - start_time
        
        conditional_print(f"Direct geographic challenge completed in {context.total_processing_time:.2f}s")
        return context
    
    async def solve_geographic_challenge(self, document_id: str, hack_team: str = "2836") -> ChallengeContext:
        """
        Solve the complete HackRX geographic challenge
        
        Args:
            document_id: ID of the challenge document
            hack_team: Team identifier for API calls
            
        Returns:
            Complete challenge context with results
        """
        context = ChallengeContext(document_id=document_id)
        start_time = time.time()
        
        await self.setup()
        
        # Step 1: Initialize - Load landmark mappings
        step_result = await self._execute_step(
            ChallengeStep.INITIALIZE,
            lambda: self._step_initialize(context)
        )
        context.steps_completed.append(step_result)
        
        if not step_result.success:
            context.total_processing_time = time.time() - start_time
            return context
        
        # Step 2: Get favorite city from API
        step_result = await self._execute_step(
            ChallengeStep.GET_CITY,
            lambda: self._step_get_city(context)
        )
        context.steps_completed.append(step_result)
        
        if not step_result.success:
            context.total_processing_time = time.time() - start_time
            return context
        
        # Step 3: Map city to landmark
        step_result = await self._execute_step(
            ChallengeStep.MAP_LANDMARK,
            lambda: self._step_map_landmark(context)
        )
        context.steps_completed.append(step_result)
        
        if not step_result.success:
            context.total_processing_time = time.time() - start_time
            return context
        
        # Step 4: Get flight number based on landmark
        step_result = await self._execute_step(
            ChallengeStep.GET_FLIGHT,
            lambda: self._step_get_flight(context)
        )
        context.steps_completed.append(step_result)
        
        # Step 5: Get secret token (parallel with flight)
        token_result = await self._execute_step(
            ChallengeStep.GET_TOKEN,
            lambda: self._step_get_token(context, hack_team)
        )
        context.steps_completed.append(token_result)
        
        # Step 6: Validate and complete
        final_result = await self._execute_step(
            ChallengeStep.COMPLETE,
            lambda: self._step_complete(context)
        )
        context.steps_completed.append(final_result)
        
        context.total_processing_time = time.time() - start_time
        return context
    
    async def _execute_step(self, step: ChallengeStep, step_function) -> StepResult:
        """Execute a single challenge step with timing and error handling"""
        start_time = time.time()
        
        try:
            result = await step_function()
            processing_time = time.time() - start_time
            
            return StepResult(
                step=step,
                success=True,
                data=result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return StepResult(
                step=step,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    async def _step_initialize_direct(self, context: ChallengeContext, document_text: str) -> Dict[str, Any]:
        """Step 1: Initialize with direct document text and load landmark mappings"""
        conditional_print("Step 1: Loading landmark mappings from direct document text...")
        
        mappings = await self.load_landmark_mappings(context.document_id, document_text)
        context.landmark_mappings = mappings
        
        return {
            "mappings_loaded": sum(len(mapping_list) for mapping_list in mappings.values()),
            "cities_covered": len(mappings),
            "available_cities": list(mappings.keys()),
            "processing_method": "direct_text"
        }
    
    async def _step_initialize(self, context: ChallengeContext) -> Dict[str, Any]:
        """Step 1: Initialize and load landmark mappings (legacy vector store method)"""
        conditional_print("Step 1: Loading landmark mappings...")
        
        mappings = await self.load_landmark_mappings(context.document_id)
        context.landmark_mappings = mappings
        
        return {
            "mappings_loaded": sum(len(mapping_list) for mapping_list in mappings.values()),
            "cities_covered": len(mappings),
            "available_cities": list(mappings.keys()),
            "processing_method": "vector_store"
        }
    
    async def _step_get_city(self, context: ChallengeContext) -> Dict[str, Any]:
        """Step 2: Get favorite city from API"""
        conditional_print("Step 2: Getting favorite city from API...")
        
        async with WebClient() as client:
            city = await client.hackrx_get_city()
            
            if not city:
                raise Exception("Failed to get favorite city from API")
            
            context.city = city
            print(f"Received city: {city}")
            
            return {"city": city}
    
    async def _step_map_landmark(self, context: ChallengeContext) -> Dict[str, Any]:
        """Step 3: Map city to landmarks using loaded mappings (supports multiple landmarks)"""
        print(f"Step 3: Mapping city '{context.city}' to landmarks...")
        
        if not context.city:
            raise Exception("No city available for mapping")
        
        # Look up city in mappings (case-insensitive)
        city_key = context.city.lower()
        
        if city_key in context.landmark_mappings:
            landmark_mappings = context.landmark_mappings[city_key]
            context.landmarks = landmark_mappings
            
            landmarks_info = []
            for mapping in landmark_mappings:
                landmarks_info.append({
                    "landmark": mapping.landmark,
                    "category": mapping.category
                })
                print(f"Found mapping: {context.city} -> {mapping.landmark}")
            
            return {
                "landmarks": landmarks_info,
                "count": len(landmark_mappings)
            }
        else:
            # Try partial matching
            for mapped_city, mapping_list in context.landmark_mappings.items():
                if context.city.lower() in mapped_city or mapped_city in context.city.lower():
                    context.landmarks = mapping_list
                    landmarks_info = []
                    for mapping in mapping_list:
                        landmarks_info.append({
                            "landmark": mapping.landmark,
                            "category": mapping.category
                        })
                        print(f"Found partial mapping: {context.city} -> {mapping.landmark}")
                    
                    return {
                        "landmarks": landmarks_info,
                        "match_type": "partial",
                        "count": len(mapping_list)
                    }
            
            # If no mapping found, use RAG to ask the document
            answer = await self._ask_document_for_landmark(context.city, context.document_id)
            
            raise Exception(f"No landmark mapping found for city: {context.city}. Available cities: {list(context.landmark_mappings.keys())}")
    
    async def _step_get_flight(self, context: ChallengeContext) -> Dict[str, Any]:
        """Step 4: Get flight numbers for all landmarks"""
        print(f"Step 4: Getting flight numbers for {len(context.landmarks)} landmarks...")
        
        if not context.landmarks:
            raise Exception("No landmarks available for flight lookup")
        
        flight_results = []
        async with WebClient() as client:
            for landmark in context.landmarks:
                try:
                    flight_number = await client.hackrx_get_flight_number(landmark.landmark)
                    
                    if flight_number:
                        context.flight_numbers.append(flight_number)
                        flight_results.append({
                            "landmark": landmark.landmark,
                            "flight_number": flight_number,
                            "success": True
                        })
                        print(f"Received flight number for {landmark.landmark}: {flight_number}")
                    else:
                        flight_results.append({
                            "landmark": landmark.landmark,
                            "error": "No flight number received",
                            "success": False
                        })
                        print(f"Failed to get flight number for landmark: {landmark.landmark}")
                        
                except Exception as e:
                    flight_results.append({
                        "landmark": landmark.landmark,
                        "error": str(e),
                        "success": False
                    })
                    print(f"Error getting flight number for {landmark.landmark}: {e}")
        
        if not context.flight_numbers:
            raise Exception("Failed to get any flight numbers for the landmarks")
        
        return {
            "flight_numbers": context.flight_numbers,
            "flight_results": flight_results,
            "successful_calls": len(context.flight_numbers),
            "total_landmarks": len(context.landmarks)
        }
    
    async def _step_get_token(self, context: ChallengeContext, hack_team: str) -> Dict[str, Any]:
        """Step 5: Get secret token"""
        print("Step 5: Getting secret token...")
        
        async with WebClient() as client:
            token = await client.hackrx_get_secret_token(hack_team)
            
            context.secret_token = token
            
            if token:
                print(f"Received secret token: {token[:10]}...")
            else:
                print("WARNING: No secret token received")
            
            return {"secret_token": token}
    
    async def _step_complete(self, context: ChallengeContext) -> Dict[str, Any]:
        """Step 6: Validate and complete challenge"""
        print("Step 6: Completing challenge...")
        
        # Validate we have the required information
        required_fields = ['city', 'landmarks', 'flight_numbers']
        missing_fields = []
        
        for field in required_fields:
            value = getattr(context, field)
            if not value or (isinstance(value, list) and len(value) == 0):
                missing_fields.append(field)
        
        success = len(missing_fields) == 0
        
        result = {
            "success": success,
            "city": context.city,
            "landmarks": [landmark.landmark for landmark in context.landmarks] if context.landmarks else [],
            "flight_numbers": context.flight_numbers,
            "secret_token": context.secret_token,
            "missing_fields": missing_fields,
            "total_steps": len(context.steps_completed),
            "processing_time": context.total_processing_time
        }
        
        if success:
            print("Challenge completed successfully!")
            landmark_names = [landmark.landmark for landmark in context.landmarks]
            print(f"Final Results for {context.city}:")
            for i, (landmark, flight_num) in enumerate(zip(landmark_names, context.flight_numbers)):
                print(f"  {i+1}. {landmark} -> Flight Number: {flight_num}")
        else:
            print(f"Challenge incomplete. Missing: {missing_fields}")
        
        return result
    
    async def _ask_document_for_landmark(self, city: str, document_id: str) -> str:
        """Use RAG to ask the document about landmark for a city"""
        try:
            # Lazy import to avoid circular dependency
            from .rag_coordinator import get_rag_coordinator
            rag_coordinator = get_rag_coordinator()
            
            question = f"What landmark is located in {city} according to the parallel world mapping?"
            
            response = await rag_coordinator.answer_question(
                question=question,
                doc_id=document_id,
                k_retrieve=10
            )
            
            return response.answer
            
        except Exception as e:
            print(f"Error asking document about {city}: {e}")
            return "Unknown landmark"
    
    def get_challenge_summary(self, context: ChallengeContext) -> Dict[str, Any]:
        """Get a summary of the challenge execution"""
        total_time = sum(step.processing_time for step in context.steps_completed)
        successful_steps = [step for step in context.steps_completed if step.success]
        
        return {
            "challenge_id": context.document_id,
            "total_steps": len(context.steps_completed),
            "successful_steps": len(successful_steps),
            "completion_rate": len(successful_steps) / len(context.steps_completed) if context.steps_completed else 0,
            "total_processing_time": total_time,
            "average_step_time": total_time / len(context.steps_completed) if context.steps_completed else 0,
            "final_result": {
                "city": context.city,
                "landmarks": [landmark.landmark for landmark in context.landmarks] if context.landmarks else [],
                "flight_numbers": context.flight_numbers,
                "landmark_count": len(context.landmarks),
                "flight_count": len(context.flight_numbers),
                "secret_token": context.secret_token[:10] + "..." if context.secret_token else None
            },
            "step_details": [
                {
                    "step": step.step.value,
                    "success": step.success,
                    "time": step.processing_time,
                    "error": step.error
                }
                for step in context.steps_completed
            ]
        }


# Singleton instance
_challenge_solver = None

async def get_challenge_solver() -> ChallengeSolver:
    """Get or create challenge solver instance"""
    global _challenge_solver
    if _challenge_solver is None:
        _challenge_solver = ChallengeSolver()
    return _challenge_solver

async def solve_hackrx_challenge(document_id: str, hack_team: str = "2836") -> Dict[str, Any]:
    """
    Convenience function to solve HackRX challenge
    
    Args:
        document_id: ID of the challenge document
        hack_team: Team identifier
        
    Returns:
        Challenge results and summary
    """
    solver = await get_challenge_solver()
    context = await solver.solve_geographic_challenge(document_id, hack_team)
    summary = solver.get_challenge_summary(context)
    
    # Format multiple flight numbers as answer
    if context.flight_numbers:
        if len(context.flight_numbers) == 1:
            answer = context.flight_numbers[0]
        else:
            answer = f"Flight Numbers: {', '.join(context.flight_numbers)}"
    else:
        answer = None
    
    return {
        "context": context,
        "summary": summary,
        "success": len(context.flight_numbers) > 0,
        "answer": answer
    }