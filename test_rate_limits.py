#!/usr/bin/env python3
"""
test_rate_limits.py

Rapid-fire test of GitHub Copilot per-second and per-minute rate limits
"""

import logging
import os
import asyncio
import time
import argparse
from collections import defaultdict, Counter
import httpx
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncGenerator
logger = logging.getLogger(__name__)
@dataclass
class CopilotResponse:
    """Response from Copilot API"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    processing_time: float = 0.0
    error: Optional[str] = None

class CopilotProvider:
    """GitHub Copilot LLM provider for RAG system"""
    
    def __init__(self, model: str = "gpt-4.1-2025-04-14", **kwargs):
        self.model = model
        self.api_base = "https://api.githubcopilot.com"
        self.kwargs = kwargs
        
        # Validate authentication
        self.api_token = os.getenv("COPILOT_ACCESS_TOKEN")
        if not self.api_token:
            raise ValueError("COPILOT_ACCESS_TOKEN environment variable is required")
        
        # Set up required headers for GitHub Copilot
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json", 
            "Accept": "application/json",
            "Copilot-Integration-Id": "vscode-chat",
            "editor-version": "VSCode/1.85.0",
            "User-Agent": "RAG-System/1.0"
        }
        
        logger.info(f"Initialized GitHub Copilot provider with model: {model}")
    
    async def generate_answer(self, prompt: str, temperature: float = 0.1) -> CopilotResponse:
        """
        Generate answer using GitHub Copilot API
        
        Args:
            prompt: The prompt to send to Copilot
            temperature: Temperature for response generation
            
        Returns:
            CopilotResponse with the generated answer
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
            
            logger.debug(f"Sending request to Copilot API with model: {self.model}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"GitHub Copilot API error {response.status_code}: {error_text}")
                    
                    # Handle specific error cases
                    if response.status_code == 401:
                        error_msg = "GitHub Copilot authentication failed. Check your COPILOT_ACCESS_TOKEN."
                    elif response.status_code == 403:
                        error_msg = "GitHub Copilot access forbidden. Verify your Copilot subscription."
                    elif response.status_code == 429:
                        error_msg = "GitHub Copilot rate limit exceeded. Please wait and try again."
                    else:
                        error_msg = f"GitHub Copilot API error {response.status_code}: {error_text}"
                    
                    return CopilotResponse(
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
                
                logger.info(f"Copilot response generated in {processing_time:.2f}s")
                logger.debug(f"Response length: {len(content)} characters")
                
                return CopilotResponse(
                    content=content,
                    model=response_data.get("model", self.model),
                    usage=usage,
                    processing_time=processing_time
                )
                
        except httpx.TimeoutException:
            processing_time = time.time() - start_time
            error_msg = "GitHub Copilot API request timed out"
            logger.error(error_msg)
            return CopilotResponse(
                content="",
                model=self.model,
                processing_time=processing_time,
                error=error_msg
            )
        except httpx.RequestError as e:
            processing_time = time.time() - start_time
            error_msg = f"GitHub Copilot API request failed: {e}"
            logger.error(error_msg)
            return CopilotResponse(
                content="",
                model=self.model,
                processing_time=processing_time,
                error=error_msg
            )
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"GitHub Copilot unexpected error: {e}"
            logger.error(error_msg)
            return CopilotResponse(
                content="",
                model=self.model,
                processing_time=processing_time,
                error=error_msg
            )
    
    async def stream_answer(self, prompt: str, temperature: float = 0.1) -> AsyncGenerator[str, None]:
        """
        Stream answer generation using GitHub Copilot API
        
        Args:
            prompt: The prompt to send to Copilot
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
            
            logger.debug(f"Starting streaming request to Copilot API")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"GitHub Copilot streaming error {response.status_code}: {error_text}")
                        yield f"Error: GitHub Copilot streaming failed: {response.status_code}"
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
            logger.error(f"GitHub Copilot streaming error: {e}")
            yield f"Error: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """
        Get available models from GitHub Copilot
        """
        return [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4",
            "gpt-4-turbo",
            "claude-3.5-sonnet",
            "claude-3.5-haiku",
            "claude-3-opus",
            "claude-3-sonnet", 
            "claude-3-haiku",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "o1-preview",
            "o1-mini"
        ]
    
    async def test_connection(self) -> bool:
        """
        Test connection to GitHub Copilot API
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = await self.generate_answer("Hello", temperature=0.1)
            return response.error is None
        except Exception as e:
            logger.error(f"Copilot connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "github_copilot",
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
_copilot_provider = None

def get_copilot_provider(model: str = "gpt-4o", **kwargs) -> CopilotProvider:
    """Get singleton Copilot provider instance"""
    global _copilot_provider
    if _copilot_provider is None or _copilot_provider.model != model:
        _copilot_provider = CopilotProvider(model=model, **kwargs)
    return _copilot_provider

async def send_request(provider, prompt: str):
    """Send a single request and return its timestamp and status."""
    ts = time.time()
    resp = await provider.generate_answer(prompt)
    # treat any 429‐style error as rate‐limit hit
    status = 'rate_limited' if resp.error and 'rate limit' in resp.error.lower() else 'success'
    return ts, status

async def worker(provider, prompt: str, end_time: float, results: List):
    """Continuously send requests until end_time, recording results."""
    while time.time() < end_time:
        ts, status = await send_request(provider, prompt)
        results.append((ts, status))

async def main():
    parser = argparse.ArgumentParser(description='Test GitHub Copilot rate limits')
    parser.add_argument('--duration',   type=int,   default=60,
                        help='Test duration in seconds (default: 60)')
    parser.add_argument('--concurrency',type=int,   default=5,
                        help='Number of parallel workers (default: 5)')
    parser.add_argument('--prompt',     type=str,   default='Hello',
                        help='Prompt text to send (default: "Hello")')
    parser.add_argument('--model',      type=str,   default='gpt-4o',
                        help='Copilot model name (default: gpt-4o)')
    args = parser.parse_args()

    provider = get_copilot_provider(model=args.model)
    end_time = time.time() + args.duration
    results = []

    # launch N workers
    tasks = [
        asyncio.create_task(worker(provider, args.prompt, end_time, results))
        for _ in range(args.concurrency)
    ]
    await asyncio.gather(*tasks)

    # aggregate per-second and per-minute counts
    per_sec = defaultdict(Counter)
    per_min = defaultdict(Counter)
    for ts, status in results:
        sec    = int(ts)
        minute = sec // 60
        per_sec[sec][status] += 1
        per_min[minute][status] += 1

    print('\nRequests per second:')
    for sec in sorted(per_sec):
        tstr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sec))
        cnt  = per_sec[sec]
        print(f'  {tstr}: success={cnt["success"]}, rate_limited={cnt["rate_limited"]}')

    print('\nRequests per minute:')
    for minute in sorted(per_min):
        msec = minute * 60
        mstr = time.strftime('%Y-%m-%d %H:%M', time.localtime(msec))
        cnt  = per_min[minute]
        print(f'  {mstr}: success={cnt["success"]}, rate_limited={cnt["rate_limited"]}')

if __name__ == '__main__':
    asyncio.run(main())
