#!/usr/bin/env python3
"""
Universal Web Client - Pure web scraping tool with zero hardcoding
Works with any URL and extraction requirement
"""
import asyncio
import aiohttp
import time
import json
import re
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs


@dataclass
class WebRequest:
    """Configuration for a web request"""
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    data: Optional[Union[Dict[str, Any], str]] = None
    params: Optional[Dict[str, str]] = None
    timeout: int = 30
    max_retries: int = 3
    follow_redirects: bool = True


@dataclass
class ExtractionRule:
    """Rule for extracting data from web content"""
    name: str
    extraction_type: str  # "regex", "json_path", "css_selector", "xpath", "text_between"
    pattern: str
    optional: bool = True
    multiple: bool = False  # Extract all matches vs first match
    post_process: Optional[str] = None  # "strip", "lower", "upper", "int", "float"


@dataclass
class WebExtractionResult:
    """Result from web content extraction"""
    url: str
    status_code: int
    success: bool
    raw_content: str
    content_type: str
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UniversalWebClient:
    """
    Universal web scraping client with zero hardcoding
    
    Philosophy:
    - Pure utility tool for web scraping
    - No business logic or hardcoded URLs
    - Configurable extraction patterns
    - Works with any website or API
    - Designed for LLM-driven usage
    """
    
    def __init__(self, default_timeout: int = 30):
        """Initialize universal web client"""
        self.default_timeout = default_timeout
        self.session = None
        
        # Default browser-like headers
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
            timeout=aiohttp.ClientTimeout(total=self.default_timeout),
            headers=self.default_headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch(self, request: WebRequest) -> WebExtractionResult:
        """
        Fetch content from any URL with configurable parameters
        
        Args:
            request: WebRequest configuration
            
        Returns:
            WebExtractionResult with content and metadata
        """
        start_time = time.time()
        
        # Prepare request headers
        headers = self.default_headers.copy()
        if request.headers:
            headers.update(request.headers)
        
        # Prepare request parameters
        kwargs = {
            'url': request.url,
            'headers': headers,
            'timeout': aiohttp.ClientTimeout(total=request.timeout),
            'allow_redirects': request.follow_redirects
        }
        
        # Add data for POST/PUT requests
        if request.data:
            if isinstance(request.data, dict):
                if headers.get('Content-Type', '').startswith('application/json'):
                    kwargs['json'] = request.data
                else:
                    kwargs['data'] = request.data
            else:
                kwargs['data'] = request.data
        
        # Add URL parameters
        if request.params:
            kwargs['params'] = request.params
        
        last_error = None
        
        for attempt in range(request.max_retries):
            try:
                async with self.session.request(request.method, **kwargs) as response:
                    content = await response.text()
                    content_type = response.headers.get('content-type', '').lower()
                    
                    processing_time = time.time() - start_time
                    
                    return WebExtractionResult(
                        url=str(response.url),  # Final URL after redirects
                        status_code=response.status,
                        success=response.status < 400,
                        raw_content=content,
                        content_type=content_type,
                        processing_time=processing_time,
                        metadata={
                            'headers': dict(response.headers),
                            'method': request.method,
                            'attempt': attempt + 1,
                            'redirected': str(response.url) != request.url
                        }
                    )
                    
            except asyncio.TimeoutError:
                last_error = f"Request timeout after {request.timeout}s"
            except aiohttp.ClientError as e:
                last_error = f"Client error: {str(e)}"
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
            
            # Exponential backoff for retries
            if attempt < request.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        # Return error result if all attempts failed
        return WebExtractionResult(
            url=request.url,
            status_code=0,
            success=False,
            raw_content="",
            content_type="",
            processing_time=time.time() - start_time,
            error=last_error
        )
    
    async def fetch_and_extract(
        self,
        url: str,
        extraction_rules: List[ExtractionRule],
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[Dict[str, Any], str]] = None,
        **kwargs
    ) -> WebExtractionResult:
        """
        Fetch URL and extract data using configurable rules
        
        Args:
            url: Target URL
            extraction_rules: List of extraction rules to apply
            method: HTTP method
            headers: Optional headers
            data: Optional request data
            **kwargs: Additional request parameters
            
        Returns:
            WebExtractionResult with extracted data
        """
        # Create request
        request = WebRequest(
            url=url,
            method=method,
            headers=headers,
            data=data,
            **kwargs
        )
        
        # Fetch content
        result = await self.fetch(request)
        
        # Extract data if fetch was successful
        if result.success:
            result.extracted_data = await self.extract_data(
                content=result.raw_content,
                content_type=result.content_type,
                extraction_rules=extraction_rules
            )
        
        return result
    
    async def extract_data(
        self,
        content: str,
        content_type: str,
        extraction_rules: List[ExtractionRule]
    ) -> Dict[str, Any]:
        """
        Extract data from content using configurable rules
        
        Args:
            content: Raw content to extract from
            content_type: Content type (for context)
            extraction_rules: Rules defining what to extract
            
        Returns:
            Dictionary with extracted data
        """
        extracted = {}
        
        for rule in extraction_rules:
            try:
                value = None
                
                if rule.extraction_type == "regex":
                    value = self._extract_regex(content, rule)
                elif rule.extraction_type == "json_path":
                    value = self._extract_json_path(content, rule)
                elif rule.extraction_type == "css_selector":
                    value = self._extract_css_selector(content, rule)
                elif rule.extraction_type == "xpath":
                    value = self._extract_xpath(content, rule)
                elif rule.extraction_type == "text_between":
                    value = self._extract_text_between(content, rule)
                elif rule.extraction_type == "auto_detect":
                    value = self._extract_auto_detect(content, rule)
                
                # Post-process value if specified
                if value is not None and rule.post_process:
                    value = self._post_process_value(value, rule.post_process)
                
                # Store result
                if value is not None:
                    extracted[rule.name] = value
                elif not rule.optional:
                    extracted[rule.name] = None
                    
            except Exception as e:
                if not rule.optional:
                    extracted[rule.name] = f"Extraction error: {str(e)}"
        
        return extracted
    
    def _extract_regex(self, content: str, rule: ExtractionRule) -> Optional[Union[str, List[str]]]:
        """Extract using regular expressions"""
        matches = re.findall(rule.pattern, content, re.IGNORECASE | re.MULTILINE)
        
        if not matches:
            return None
        
        if rule.multiple:
            return matches
        else:
            return matches[0] if matches else None
    
    def _extract_json_path(self, content: str, rule: ExtractionRule) -> Optional[Any]:
        """Extract from JSON using dot notation path (e.g., 'data.user.name')"""
        try:
            data = json.loads(content)
            
            # Navigate the JSON path
            path_parts = rule.pattern.split('.')
            current = data
            
            for part in path_parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif isinstance(current, list) and part.isdigit():
                    index = int(part)
                    current = current[index] if 0 <= index < len(current) else None
                else:
                    return None
            
            return current
            
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return None
    
    def _extract_css_selector(self, content: str, rule: ExtractionRule) -> Optional[Union[str, List[str]]]:
        """Extract using CSS selectors"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            elements = soup.select(rule.pattern)
            
            if not elements:
                return None
            
            if rule.multiple:
                return [elem.get_text().strip() for elem in elements]
            else:
                return elements[0].get_text().strip() if elements else None
                
        except Exception:
            return None
    
    def _extract_xpath(self, content: str, rule: ExtractionRule) -> Optional[Union[str, List[str]]]:
        """Extract using XPath (requires lxml)"""
        try:
            from lxml import html
            
            tree = html.fromstring(content)
            elements = tree.xpath(rule.pattern)
            
            if not elements:
                return None
            
            if rule.multiple:
                return [elem.text.strip() if hasattr(elem, 'text') and elem.text else str(elem) for elem in elements]
            else:
                elem = elements[0]
                return elem.text.strip() if hasattr(elem, 'text') and elem.text else str(elem)
                
        except ImportError:
            # lxml not available, fallback to CSS selector if possible
            return None
        except Exception:
            return None
    
    def _extract_text_between(self, content: str, rule: ExtractionRule) -> Optional[Union[str, List[str]]]:
        """Extract text between two markers"""
        try:
            start_marker, end_marker = rule.pattern.split('|||', 1)  # Use ||| as separator
            
            if rule.multiple:
                # Find all occurrences
                results = []
                start_index = 0
                
                while True:
                    start_pos = content.find(start_marker, start_index)
                    if start_pos == -1:
                        break
                    
                    start_pos += len(start_marker)
                    end_pos = content.find(end_marker, start_pos)
                    
                    if end_pos == -1:
                        break
                    
                    results.append(content[start_pos:end_pos].strip())
                    start_index = end_pos + len(end_marker)
                
                return results if results else None
            else:
                # Find first occurrence
                start_pos = content.find(start_marker)
                if start_pos == -1:
                    return None
                
                start_pos += len(start_marker)
                end_pos = content.find(end_marker, start_pos)
                
                if end_pos == -1:
                    return None
                
                return content[start_pos:end_pos].strip()
                
        except ValueError:
            # Invalid pattern format
            return None
    
    def _extract_auto_detect(self, content: str, rule: ExtractionRule) -> Optional[Any]:
        """Auto-detect and extract common patterns"""
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://[^\s<>"{}|\\^`[\]]+',
            'phone': r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'token': r'[A-Za-z0-9+/]{20,}={0,2}',
            'hex_id': r'[a-fA-F0-9]{6,}',
            'uuid': r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
            'number': r'\b\d+\.?\d*\b',
            'date': r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}'
        }
        
        pattern_name = rule.pattern.lower()
        if pattern_name in patterns:
            matches = re.findall(patterns[pattern_name], content, re.IGNORECASE)
            
            if rule.multiple:
                return matches if matches else None
            else:
                return matches[0] if matches else None
        
        return None
    
    def _post_process_value(self, value: Any, post_process: str) -> Any:
        """Apply post-processing to extracted value"""
        if value is None:
            return None
        
        try:
            if post_process == "strip" and isinstance(value, str):
                return value.strip()
            elif post_process == "lower" and isinstance(value, str):
                return value.lower()
            elif post_process == "upper" and isinstance(value, str):
                return value.upper()
            elif post_process == "int":
                return int(float(str(value)))
            elif post_process == "float":
                return float(str(value))
            elif post_process == "clean_whitespace" and isinstance(value, str):
                return ' '.join(value.split())
            elif post_process == "remove_quotes" and isinstance(value, str):
                return value.strip('"\'')
        except (ValueError, TypeError):
            pass
        
        return value
    
    async def quick_fetch(self, url: str, **kwargs) -> WebExtractionResult:
        """Quick fetch for simple use cases"""
        request = WebRequest(url=url, **kwargs)
        return await self.fetch(request)
    
    def create_extraction_rules_from_patterns(self, patterns: Dict[str, str]) -> List[ExtractionRule]:
        """Helper to create extraction rules from simple pattern dictionary"""
        rules = []
        
        for name, pattern in patterns.items():
            # Auto-detect extraction type based on pattern
            extraction_type = "regex"
            
            if pattern.startswith("$.") or "." in pattern and not re.search(r'[.*+?^${}()|[\]\\]', pattern):
                extraction_type = "json_path"
            elif pattern.startswith("#") or pattern.startswith(".") or " " in pattern:
                extraction_type = "css_selector"
            elif "|||" in pattern:
                extraction_type = "text_between"
            elif pattern in ["email", "url", "phone", "token", "hex_id", "uuid", "number", "date"]:
                extraction_type = "auto_detect"
            
            rules.append(ExtractionRule(
                name=name,
                extraction_type=extraction_type,
                pattern=pattern,
                optional=True
            ))
        
        return rules


# Utility functions for easy usage
async def fetch_url(url: str, **kwargs) -> WebExtractionResult:
    """Quick utility to fetch a URL"""
    async with UniversalWebClient() as client:
        return await client.quick_fetch(url, **kwargs)


async def extract_from_url(url: str, patterns: Dict[str, str], **kwargs) -> Dict[str, Any]:
    """Quick utility to fetch and extract data from URL"""
    async with UniversalWebClient() as client:
        rules = client.create_extraction_rules_from_patterns(patterns)
        result = await client.fetch_and_extract(url, rules, **kwargs)
        return result.extracted_data if result.success else {}


async def scrape_json_api(url: str, field_paths: List[str], **kwargs) -> Dict[str, Any]:
    """Quick utility for JSON API scraping"""
    async with UniversalWebClient() as client:
        rules = [
            ExtractionRule(
                name=path.replace(".", "_"),
                extraction_type="json_path",
                pattern=path,
                optional=True
            ) for path in field_paths
        ]
        
        result = await client.fetch_and_extract(url, rules, **kwargs)
        return result.extracted_data if result.success else {}


# Singleton instance for shared usage
_universal_client = None

async def get_universal_web_client() -> UniversalWebClient:
    """Get singleton universal web client"""
    global _universal_client
    if _universal_client is None:
        _universal_client = UniversalWebClient()
    return _universal_client