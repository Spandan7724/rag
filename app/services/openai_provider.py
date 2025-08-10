"""
Custom OpenAI provider for RAG system
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OpenAIResponse:
    """Response from OpenAI API"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    processing_time: float = 0.0
    error: Optional[str] = None

class OpenAIProvider:
    """OpenAI LLM provider for RAG system"""
    
    def __init__(self, model: str = "gpt-4o", **kwargs):
        self.model = model
        self.api_base = "https://api.openai.com/v1"
        self.kwargs = kwargs
        
        # Validate authentication
        self.api_token = os.getenv("OPENAI_API_KEY")
        if not self.api_token:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Set up required headers for OpenAI
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json", 
            "Accept": "application/json",
            "User-Agent": "RAG-System/1.0"
        }
        
        logger.info(f"Initialized OpenAI provider with model: {model}")
    
    async def generate_answer(self, prompt: str, temperature: float = 0.1) -> OpenAIResponse:
        """
        Generate answer using OpenAI API
        
        Args:
            prompt: The prompt to send to OpenAI
            temperature: Temperature for response generation
            
        Returns:
            OpenAIResponse with the generated answer
        """
        start_time = time.time()
        
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "max_tokens": self.kwargs.get("max_tokens", 2048),
                **{k: v for k, v in self.kwargs.items() if k != "max_tokens"}
            }
            
            logger.debug(f"Sending request to OpenAI API with model: {self.model}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"OpenAI API error {response.status_code}: {error_text}")
                    
                    # Handle specific error cases
                    if response.status_code == 401:
                        error_msg = "OpenAI authentication failed. Check your OPENAI_API_KEY."
                    elif response.status_code == 403:
                        error_msg = "OpenAI access forbidden. Verify your API key permissions."
                    elif response.status_code == 429:
                        error_msg = "OpenAI rate limit exceeded. Please wait and try again."
                    elif response.status_code == 400:
                        try:
                            error_data = response.json()
                            error_detail = error_data.get("error", {}).get("message", error_text)
                            error_msg = f"OpenAI API error: {error_detail}"
                        except:
                            error_msg = f"OpenAI API error 400: {error_text}"
                    else:
                        error_msg = f"OpenAI API error {response.status_code}: {error_text}"
                    
                    return OpenAIResponse(
                        content="",
                        model=self.model,
                        processing_time=processing_time,
                        error=error_msg
                    )
                
                response_data = response.json()
                
                # Extract the response content
                choice = response_data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "").strip()
                
                # Extract usage information
                usage = None
                if "usage" in response_data:
                    usage_data = response_data["usage"]
                    usage = {
                        "prompt_tokens": usage_data.get("prompt_tokens", 0),
                        "completion_tokens": usage_data.get("completion_tokens", 0),
                        "total_tokens": usage_data.get("total_tokens", 0)
                    }
                
                logger.info(f"OpenAI response generated in {processing_time:.2f}s")
                logger.debug(f"Response length: {len(content)} characters")
                
                return OpenAIResponse(
                    content=content,
                    model=response_data.get("model", self.model),
                    usage=usage,
                    processing_time=processing_time
                )
                
        except httpx.TimeoutException:
            processing_time = time.time() - start_time
            error_msg = "OpenAI API request timed out"
            logger.error(error_msg)
            return OpenAIResponse(
                content="",
                model=self.model,
                processing_time=processing_time,
                error=error_msg
            )
        except httpx.RequestError as e:
            processing_time = time.time() - start_time
            error_msg = f"OpenAI API request failed: {e}"
            logger.error(error_msg)
            return OpenAIResponse(
                content="",
                model=self.model,
                processing_time=processing_time,
                error=error_msg
            )
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"OpenAI unexpected error: {e}"
            logger.error(error_msg)
            return OpenAIResponse(
                content="",
                model=self.model,
                processing_time=processing_time,
                error=error_msg
            )
    
    async def stream_answer(self, prompt: str, temperature: float = 0.1) -> AsyncGenerator[str, None]:
        """
        Stream answer generation using OpenAI API
        
        Args:
            prompt: The prompt to send to OpenAI
            temperature: Temperature for response generation
            
        Yields:
            Streaming content chunks
        """
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "max_tokens": self.kwargs.get("max_tokens", 2048),
                "stream": True,
                **{k: v for k, v in self.kwargs.items() if k not in ["max_tokens", "stream"]}
            }
            
            logger.debug("Starting streaming request to OpenAI API")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"OpenAI streaming error {response.status_code}: {error_text}")
                        yield f"Error: OpenAI streaming failed: {response.status_code}"
                        return
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            
                            if data.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data)
                                
                                # Extract delta content
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    
                                    if content:
                                        yield content
                                        
                            except json.JSONDecodeError:
                                continue  # Skip invalid JSON lines
                                
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield f"Error: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """
        Get available models from OpenAI
        """
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "o1-preview",
            "o1-mini"
        ]
    
    async def test_connection(self) -> bool:
        """
        Test connection to OpenAI API
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = await self.generate_answer("Hello", temperature=0.1)
            return response.error is None
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "openai",
            "model_name": self.model,
            "api_base": self.api_base,
            "features": {
                "streaming": True,
                "temperature_control": True,
                "max_tokens_control": True,
                "function_calling": True
            }
        }


# Singleton instance for easy usage
_openai_provider = None

def get_openai_provider(model: str = "gpt-4o", **kwargs) -> OpenAIProvider:
    """Get singleton OpenAI provider instance"""
    global _openai_provider
    if _openai_provider is None or _openai_provider.model != model:
        _openai_provider = OpenAIProvider(model=model, **kwargs)
    return _openai_provider
