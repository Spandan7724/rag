#!/usr/bin/env python3
"""
Web Client Service
Handles web scraping, API integration, and HTML content extraction for HackRX challenges
"""
import asyncio
import aiohttp
import time
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin, urlparse

@dataclass
class WebResponse:
    """Web response data structure"""
    url: str
    status_code: int
    content: str
    headers: Dict[str, str]
    is_json: bool
    is_html: bool
    processing_time: float
    error: Optional[str] = None

class WebClient:
    """
    Web client for API integration and web scraping
    Handles HackRX endpoints, HTML parsing, and token extraction
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """Initialize web client"""
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session = None
        
        # Default headers to mimic browser requests
        self.default_headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            headers=self.default_headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_url(self, url: str, headers: Optional[Dict[str, str]] = None) -> WebResponse:
        """
        Fetch content from a URL with retry logic
        
        Args:
            url: Target URL
            headers: Optional additional headers
            
        Returns:
            WebResponse object with content and metadata
        """
        start_time = time.time()
        
        # Merge headers
        request_headers = self.default_headers.copy()
        if headers:
            request_headers.update(headers)
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url, headers=request_headers) as response:
                    processing_time = time.time() - start_time
                    content = await response.text()
                    
                    # Determine content type
                    content_type = response.headers.get('content-type', '').lower()
                    is_json = 'application/json' in content_type
                    is_html = 'text/html' in content_type
                    
                    return WebResponse(
                        url=url,
                        status_code=response.status,
                        content=content,
                        headers=dict(response.headers),
                        is_json=is_json,
                        is_html=is_html,
                        processing_time=processing_time,
                        error=None if response.status < 400 else f"HTTP {response.status}"
                    )
                    
            except asyncio.TimeoutError as e:
                last_error = f"Timeout after {self.timeout.total}s"
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except aiohttp.ClientError as e:
                last_error = f"Client error: {str(e)}"
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                break
        
        # Return error response if all attempts failed
        processing_time = time.time() - start_time
        return WebResponse(
            url=url,
            status_code=0,
            content="",
            headers={},
            is_json=False,
            is_html=False,
            processing_time=processing_time,
            error=last_error
        )
    
    def extract_html_text(self, html_content: str, remove_scripts: bool = True) -> str:
        """
        Extract clean text content from HTML
        
        Args:
            html_content: Raw HTML content
            remove_scripts: Whether to remove script and style tags
            
        Returns:
            Clean text content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            if remove_scripts:
                # Remove script and style tags
                for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                    tag.decompose()
            
            # Extract text with some formatting
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            return f"HTML parsing error: {str(e)}"
    
    def extract_json_data(self, json_content: str) -> Dict[str, Any]:
        """
        Parse JSON content safely
        
        Args:
            json_content: Raw JSON string
            
        Returns:
            Parsed JSON data or error dict
        """
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing error: {str(e)}"}
    
    def extract_tokens_from_html(self, html_content: str) -> List[str]:
        """
        Extract potential tokens, codes, or identifiers from HTML content
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            List of potential tokens found
        """
        tokens = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for common token patterns
            text = soup.get_text()
            
            # Pattern 1: Long alphanumeric strings (potential tokens)
            token_patterns = [
                r'[A-Za-z0-9]{20,}',  # Long alphanumeric strings
                r'[A-Fa-f0-9]{32,}',  # Hex strings (MD5, SHA etc)
                r'[A-Za-z0-9+/]{20,}={0,2}',  # Base64-like strings
                r'Bearer\s+([A-Za-z0-9+/=\-_.]+)',  # Bearer tokens
                r'token["\']?\s*:\s*["\']?([A-Za-z0-9+/=\-_.]+)',  # JSON tokens
            ]
            
            for pattern in token_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                tokens.extend(matches)
            
            # Look in specific HTML elements
            for element in soup.find_all(['input', 'meta', 'script']):
                # Check for tokens in value attributes
                if element.get('value'):
                    value = element.get('value')
                    if len(value) > 15 and re.match(r'^[A-Za-z0-9+/=\-_.]+$', value):
                        tokens.append(value)
                
                # Check for tokens in content attributes
                if element.get('content'):
                    content = element.get('content')
                    if len(content) > 15 and re.match(r'^[A-Za-z0-9+/=\-_.]+$', content):
                        tokens.append(content)
            
            # Remove duplicates and sort by length (longer tokens first)
            unique_tokens = list(set(tokens))
            unique_tokens.sort(key=len, reverse=True)
            
            return unique_tokens
            
        except Exception as e:
            return [f"Token extraction error: {str(e)}"]
    
    async def hackrx_get_city(self) -> Optional[str]:
        """
        Get favorite city from HackRX endpoint
        
        Returns:
            City name or None if failed
        """
        url = "https://register.hackrx.in/submissions/myFavouriteCity"
        
        try:
            print(f"Attempting to fetch favorite city from: {url}")
            response = await self.fetch_url(url)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"Response content: {response.content[:400]}...")
                if response.is_json:
                    data = self.extract_json_data(response.content)
                    print(f"Parsed JSON data: {data}")
                    
                    # Handle HackRX API response structure: {"success": true, "data": {"city": "..."}}
                    city = None
                    if isinstance(data, dict):
                        # Try nested data structure first
                        if 'data' in data and isinstance(data['data'], dict):
                            city = data['data'].get('city') or data['data'].get('favourite_city') or data['data'].get('name')
                        
                        # Try root level as fallback
                        if not city:
                            city = data.get('city') or data.get('favourite_city') or data.get('name')
                    
                    if city:
                        print(f"Extracted city: {city}")
                        return city
                else:
                    # If it's plain text, try to extract city name
                    text = response.content.strip().strip('"\'')
                    if text:
                        print(f"Extracted city from text: {text}")
                        return text
            
            print(f"Failed to get city. Status: {response.status_code}, Error: {response.error}")
            return None
            
        except Exception as e:
            print(f"Error getting favorite city: {e}")
            return None
    
    async def hackrx_get_secret_token(self, hack_team: str = "2836") -> Optional[str]:
        """
        Get secret token from HackRX endpoint
        
        Args:
            hack_team: Team identifier
            
        Returns:
            Secret token or None if failed
        """
        url = f"https://register.hackrx.in/utils/get-secret-token?hackTeam={hack_team}"
        
        try:
            response = await self.fetch_url(url)
            
            if response.status_code == 200:
                if response.is_json:
                    data = self.extract_json_data(response.content)
                    return data.get('token') or data.get('secret_token') or data.get('secretToken')
                elif response.is_html:
                    # Extract tokens from HTML
                    tokens = self.extract_tokens_from_html(response.content)
                    return tokens[0] if tokens else None
                else:
                    # Plain text response
                    text = response.content.strip().strip('"\'')
                    return text if text else None
            
            return None
            
        except Exception as e:
            print(f"Error getting secret token: {e}")
            return None
    
    async def hackrx_get_flight_number(self, landmark_type: str) -> Optional[str]:
        """
        Get flight number based on landmark type
        
        Args:
            landmark_type: Type of landmark (Gateway of India, Taj Mahal, etc.)
            
        Returns:
            Flight number or None if failed
        """
        # Map landmark types to endpoints
        endpoint_map = {
            "Gateway of India": "getFirstCityFlightNumber",
            "Taj Mahal": "getSecondCityFlightNumber", 
            "Eiffel Tower": "getThirdCityFlightNumber",
            "Big Ben": "getFourthCityFlightNumber"
        }
        
        # Default endpoint for other landmarks
        endpoint = endpoint_map.get(landmark_type, "getFifthCityFlightNumber")
        url = f"https://register.hackrx.in/teams/public/flights/{endpoint}"
        
        try:
            print(f"Attempting to fetch flight number for {landmark_type} from: {url}")
            response = await self.fetch_url(url)
            print(f"Flight API response status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"Flight API response content: {response.content[:400]}...")
                if response.is_json:
                    data = self.extract_json_data(response.content)
                    print(f"Parsed flight JSON data: {data}")
                    
                    # Handle HackRX API response structure: {"success": true, "data": {"flightNumber": "aed52f"}}
                    flight_number = None
                    if isinstance(data, dict):
                        # Try nested data structure first (HackRX format)
                        if 'data' in data and isinstance(data['data'], dict):
                            flight_number = data['data'].get('flightNumber') or data['data'].get('flight_number') or data['data'].get('flight')
                        
                        # Try root level as fallback
                        if not flight_number:
                            flight_number = data.get('flight_number') or data.get('flightNumber') or data.get('flight')
                    
                    if flight_number:
                        print(f"Extracted flight number: {flight_number}")
                        return flight_number
                else:
                    # Try to extract flight number from text
                    text = response.content.strip()
                    # Look for flight number patterns (e.g., AA123, FL456, etc.)
                    flight_pattern = r'[A-Z]{2}\d{3,4}|[a-f0-9]{6}'  # Added hex pattern for aed52f format
                    matches = re.findall(flight_pattern, text)
                    if matches:
                        print(f"Extracted flight number from pattern: {matches[0]}")
                        return matches[0]
                    elif text:
                        print(f"Extracted flight number from text: {text}")
                        return text
            
            print(f"Failed to get flight number. Status: {response.status_code}, Error: {response.error}")
            return None
            
        except Exception as e:
            print(f"Error getting flight number for {landmark_type}: {e}")
            return None


# Singleton instance
_web_client = None

async def get_web_client() -> WebClient:
    """Get or create web client instance"""
    global _web_client
    if _web_client is None:
        _web_client = WebClient()
    return _web_client


# Utility functions for easy access
async def fetch_web_content(url: str, headers: Optional[Dict[str, str]] = None) -> WebResponse:
    """Fetch content from a URL"""
    async with WebClient() as client:
        return await client.fetch_url(url, headers)

async def extract_tokens_from_url(url: str) -> List[str]:
    """Extract tokens from a web page"""
    async with WebClient() as client:
        response = await client.fetch_url(url)
        if response.status_code == 200 and response.is_html:
            return client.extract_tokens_from_html(response.content)
        return []

async def solve_hackrx_challenge() -> Dict[str, Any]:
    """
    Solve the complete HackRX challenge workflow
    
    Returns:
        Dictionary with challenge solution steps and final answer
    """
    result = {
        "steps": [],
        "city": None,
        "landmark": None, 
        "flight_number": None,
        "secret_token": None,
        "success": False,
        "error": None
    }
    
    try:
        async with WebClient() as client:
            # Step 1: Get favorite city
            result["steps"].append("Getting favorite city from API...")
            city = await client.hackrx_get_city()
            result["city"] = city
            
            if not city:
                result["error"] = "Failed to get favorite city"
                return result
            
            # Step 2: Map city to landmark (this would need the landmark mapping data)
            # For now, we'll need to implement the lookup logic separately
            result["steps"].append(f"City received: {city}")
            
            # Step 3: Get secret token
            result["steps"].append("Getting secret token...")
            secret_token = await client.hackrx_get_secret_token()
            result["secret_token"] = secret_token
            
            # Step 4: Get flight number (placeholder - needs landmark mapping)
            # This would be called after determining the landmark
            result["steps"].append("Challenge workflow initiated successfully")
            result["success"] = True
            
    except Exception as e:
        result["error"] = str(e)
    
    return result