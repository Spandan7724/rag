#!/usr/bin/env python3
"""
Challenge Solver Service
Multi-step reasoning engine for HackRX geographic puzzles and complex challenges
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .web_client import WebClient, get_web_client
from .table_extractor import TableExtractor, LandmarkMapping, get_table_extractor
from .vector_store import get_vector_store
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
    landmark_mappings: Dict[str, LandmarkMapping] = field(default_factory=dict)
    city: Optional[str] = None
    landmark: Optional[LandmarkMapping] = None
    flight_number: Optional[str] = None
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
    
    async def load_landmark_mappings(self, document_id: str) -> Dict[str, LandmarkMapping]:
        """
        Load landmark mappings from processed documents
        
        Args:
            document_id: ID of the challenge document
            
        Returns:
            Dictionary mapping cities to landmark data
        """
        mappings = {}
        
        try:
            # Get document info from vector store
            if document_id in self.vector_store.documents:
                doc_info = self.vector_store.documents[document_id]
                
                # Try to find the PDF path in blob storage or parsed documents
                # For now, we'll extract from the document text content
                doc_chunks = []
                for i in range(len(self.vector_store.chunk_texts)):
                    metadata = self.vector_store.chunk_metadata[i]
                    if metadata.doc_id == document_id:
                        doc_chunks.append(self.vector_store.chunk_texts[i])
                
                # Combine all chunks to get full document text
                full_text = "\n".join(doc_chunks)
                
                # Extract mappings from text
                text_mappings = self.table_extractor.extract_landmark_mappings_from_text(full_text)
                
                # Create lookup dictionary
                mappings = self.table_extractor.create_lookup_dict(text_mappings)
                
                print(f"Loaded {len(mappings)} landmark mappings for document {document_id}")
            
        except Exception as e:
            print(f"Error loading landmark mappings: {e}")
        
        return mappings
    
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
    
    async def _step_initialize(self, context: ChallengeContext) -> Dict[str, Any]:
        """Step 1: Initialize and load landmark mappings"""
        print("Step 1: Loading landmark mappings...")
        
        mappings = await self.load_landmark_mappings(context.document_id)
        context.landmark_mappings = mappings
        
        return {
            "mappings_loaded": len(mappings),
            "available_cities": list(mappings.keys())
        }
    
    async def _step_get_city(self, context: ChallengeContext) -> Dict[str, Any]:
        """Step 2: Get favorite city from API"""
        print("Step 2: Getting favorite city from API...")
        
        async with WebClient() as client:
            city = await client.hackrx_get_city()
            
            if not city:
                raise Exception("Failed to get favorite city from API")
            
            context.city = city
            print(f"Received city: {city}")
            
            return {"city": city}
    
    async def _step_map_landmark(self, context: ChallengeContext) -> Dict[str, Any]:
        """Step 3: Map city to landmark using loaded mappings"""
        print(f"Step 3: Mapping city '{context.city}' to landmark...")
        
        if not context.city:
            raise Exception("No city available for mapping")
        
        # Look up city in mappings (case-insensitive)
        city_key = context.city.lower()
        
        if city_key in context.landmark_mappings:
            landmark_mapping = context.landmark_mappings[city_key]
            context.landmark = landmark_mapping
            
            print(f"Found mapping: {context.city} -> {landmark_mapping.landmark}")
            
            return {
                "landmark": landmark_mapping.landmark,
                "category": landmark_mapping.category,
                "emoji": landmark_mapping.emoji
            }
        else:
            # Try partial matching
            for mapped_city, mapping in context.landmark_mappings.items():
                if context.city.lower() in mapped_city or mapped_city in context.city.lower():
                    context.landmark = mapping
                    print(f"Found partial mapping: {context.city} -> {mapping.landmark}")
                    
                    return {
                        "landmark": mapping.landmark,
                        "category": mapping.category,
                        "emoji": mapping.emoji,
                        "match_type": "partial"
                    }
            
            # If no mapping found, use RAG to ask the document
            answer = await self._ask_document_for_landmark(context.city, context.document_id)
            
            raise Exception(f"No landmark mapping found for city: {context.city}. Available cities: {list(context.landmark_mappings.keys())}")
    
    async def _step_get_flight(self, context: ChallengeContext) -> Dict[str, Any]:
        """Step 4: Get flight number based on landmark type"""
        print(f"Step 4: Getting flight number for landmark '{context.landmark.landmark}'...")
        
        if not context.landmark:
            raise Exception("No landmark available for flight lookup")
        
        async with WebClient() as client:
            flight_number = await client.hackrx_get_flight_number(context.landmark.landmark)
            
            if not flight_number:
                raise Exception(f"Failed to get flight number for landmark: {context.landmark.landmark}")
            
            context.flight_number = flight_number
            print(f"Received flight number: {flight_number}")
            
            return {"flight_number": flight_number}
    
    async def _step_get_token(self, context: ChallengeContext, hack_team: str) -> Dict[str, Any]:
        """Step 5: Get secret token"""
        print("Step 5: Getting secret token...")
        
        async with WebClient() as client:
            token = await client.hackrx_get_secret_token(hack_team)
            
            context.secret_token = token
            
            if token:
                print(f"Received secret token: {token[:10]}...")
            else:
                print("⚠️ No secret token received")
            
            return {"secret_token": token}
    
    async def _step_complete(self, context: ChallengeContext) -> Dict[str, Any]:
        """Step 6: Validate and complete challenge"""
        print("Step 6: Completing challenge...")
        
        # Validate we have the required information
        required_fields = ['city', 'landmark', 'flight_number']
        missing_fields = []
        
        for field in required_fields:
            if not getattr(context, field):
                missing_fields.append(field)
        
        success = len(missing_fields) == 0
        
        result = {
            "success": success,
            "city": context.city,
            "landmark": context.landmark.landmark if context.landmark else None,
            "flight_number": context.flight_number,
            "secret_token": context.secret_token,
            "missing_fields": missing_fields,
            "total_steps": len(context.steps_completed),
            "processing_time": context.total_processing_time
        }
        
        if success:
            print("Challenge completed successfully!")
            print(f"Final Answer - Flight Number: {context.flight_number}")
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
                "landmark": context.landmark.landmark if context.landmark else None,
                "flight_number": context.flight_number,
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
    
    return {
        "context": context,
        "summary": summary,
        "success": context.flight_number is not None,
        "answer": context.flight_number
    }