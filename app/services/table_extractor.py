#!/usr/bin/env python3
"""
Table Extractor Service
Enhanced table and structured data extraction from PDFs for HackRX challenges
"""
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pymupdf as fitz  # PyMuPDF
from app.utils.debug import conditional_print
# Optional imports for advanced table extraction
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    conditional_print("Warning: camelot-py not available. Advanced table extraction disabled.")

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    conditional_print("Warning: tabula-py not available. Java-based table extraction disabled.")
from io import StringIO
import tempfile
import os

@dataclass
class TableData:
    """Structured table data"""
    headers: List[str]
    rows: List[List[str]]
    table_index: int
    page_number: int
    extraction_method: str
    confidence_score: float
    
@dataclass 
class LandmarkMapping:
    """Landmark-city mapping data structure"""
    landmark: str
    current_city: str
    original_city: Optional[str] = None
    category: str = "unknown"  # indian, international
    emoji: Optional[str] = None

class TableExtractor:
    """
    Enhanced table extraction for geographic mapping challenges
    Handles landmark-city relationships and structured data from PDFs
    """
    
    def __init__(self):
        """Initialize table extractor"""
        self.landmark_keywords = [
            "Gateway of India", "India Gate", "Charminar", "Marina Beach", 
            "Howrah Bridge", "Golconda Fort", "Qutub Minar", "Taj Mahal",
            "Meenakshi Temple", "Lotus Temple", "Mysore Palace", "Rock Garden",
            "Victoria Memorial", "Vidhana Soudha", "Sun Temple", "Golden Temple",
            "Eiffel Tower", "Statue of Liberty", "Big Ben", "Colosseum",
            "Sydney Opera House", "Christ the Redeemer", "Burj Khalifa",
            "CN Tower", "Petronas Towers", "Leaning Tower of Pisa", 
            "Mount Fuji", "Niagara Falls", "Louvre Museum", "Stonehenge",
            "Sagrada Familia", "Acropolis", "Machu Picchu", "Moai Statues",
            "Christchurch Cathedral", "The Shard", "Blue Mosque",
            "Neuschwanstein Castle", "Buckingham Palace", "Space Needle", "Times Square"
        ]
        
        self.indian_cities = [
            "Delhi", "Mumbai", "Chennai", "Hyderabad", "Ahmedabad", "Mysuru",
            "Kochi", "Pune", "Nagpur", "Chandigarh", "Kerala", "Bhopal",
            "Varanasi", "Jaisalmer"
        ]
        
        self.international_cities = [
            "New York", "London", "Tokyo", "Beijing", "Bangkok", "Toronto",
            "Dubai", "Amsterdam", "Cairo", "San Francisco", "Berlin", "Barcelona", 
            "Moscow", "Seoul", "Cape Town", "Istanbul", "Riyadh", "Paris",
            "Singapore", "Jakarta", "Vienna", "Kathmandu", "Los Angeles"
        ]
    
    def extract_tables_pymupdf(self, pdf_path: str) -> List[TableData]:
        """
        Extract tables using PyMuPDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted table data
        """
        tables = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Find tables on the page
                page_tables = page.find_tables()
                
                for i, table in enumerate(page_tables):
                    try:
                        # Extract table data
                        table_data = table.extract()
                        
                        if table_data and len(table_data) > 0:
                            # Process headers and rows
                            headers = table_data[0] if table_data[0] else []
                            rows = table_data[1:] if len(table_data) > 1 else []
                            
                            # Clean empty cells
                            headers = [str(cell).strip() if cell else "" for cell in headers]
                            rows = [[str(cell).strip() if cell else "" for cell in row] for row in rows]
                            
                            tables.append(TableData(
                                headers=headers,
                                rows=rows,
                                table_index=i,
                                page_number=page_num + 1,
                                extraction_method="pymupdf",
                                confidence_score=0.8  # PyMuPDF generally reliable
                            ))
                    
                    except Exception as e:
                        print(f"Error extracting table {i} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing PDF with PyMuPDF: {e}")
        
        return tables
    
    def extract_tables_camelot(self, pdf_path: str) -> List[TableData]:
        """
        Extract tables using Camelot
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted table data
        """
        if not CAMELOT_AVAILABLE:
            print("Camelot not available, skipping advanced table extraction")
            return []
            
        tables = []
        
        try:
            # Try lattice method first (for tables with borders)
            camelot_tables = camelot.read_pdf(pdf_path, flavor='lattice', pages='all')
            
            # If lattice fails, try stream method
            if len(camelot_tables) == 0:
                camelot_tables = camelot.read_pdf(pdf_path, flavor='stream', pages='all')
            
            for i, table in enumerate(camelot_tables):
                try:
                    df = table.df
                    
                    # Convert DataFrame to our format
                    headers = df.columns.tolist()
                    rows = df.values.tolist()
                    
                    # Clean data
                    headers = [str(col).strip() for col in headers]
                    rows = [[str(cell).strip() if pd.notna(cell) else "" for cell in row] for row in rows]
                    
                    tables.append(TableData(
                        headers=headers,
                        rows=rows,
                        table_index=i,
                        page_number=table.parsing_report['page'],
                        extraction_method="camelot",
                        confidence_score=table.accuracy / 100.0  # Convert to 0-1 scale
                    ))
                
                except Exception as e:
                    print(f"Error processing Camelot table {i}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error processing PDF with Camelot: {e}")
        
        return tables
    
    def extract_landmark_mappings_from_text(self, text: str) -> List[LandmarkMapping]:
        """
        Extract landmark mappings from plain text using pattern matching
        
        Args:
            text: Document text content
            
        Returns:
            List of landmark mappings
        """
        mappings = []
        
        print("Extracting landmark mappings from text...")
        print(f"Text sample (first 500 chars): {text[:500]}")
        
        # Pattern 1: Handle emoji + landmark on one line, city on next line
        # Example:
        # 🏖️ Marina Beach
        # Hyderabad  
        pattern_newline = r'([🏛️🗼🕌🏖️🌉🏰❤️⛩️🌸👑🪨🏢🌞✨🗽🕰️🏟️🎭✝️🏙️🖼️🌊🪨🕍🏞️🗿⛪🛰️🌆]+)\s*([^\n🏛️🗼🕌🏖️🌉🏰❤️⛩️🌸👑🪨🏢🌞✨🗽🕰️🏟️🎭✝️🏙️🖼️🌊🪨🕍🏞️🗿⛪🛰️🌆]+)\n\s*([A-Za-z\s]+?)(?=\n|$)'
        
        newline_matches = re.findall(pattern_newline, text, re.MULTILINE)
        print(f"Found {len(newline_matches)} newline pattern matches")
        
        for emoji, landmark, city in newline_matches:
            landmark = landmark.strip()
            city = city.strip()
            print(f"Processing: {emoji} {landmark} -> {city}")
            
            # Validate landmark and city
            if self._is_valid_landmark(landmark) and self._is_valid_city(city):
                category = "indian" if city in self.indian_cities else "international"
                mapping = LandmarkMapping(
                    landmark=landmark,
                    current_city=city,
                    category=category,
                    emoji=emoji
                )
                mappings.append(mapping)
                print(f"Added mapping: {landmark} -> {city} ({category})")
            else:
                print(f"Skipped invalid mapping: {landmark} -> {city}")
        
        # Pattern 2: Fallback - Look for emoji + landmark name + city patterns on same line
        # Example: 🏛️ Gateway of India Delhi
        pattern_sameline = r'([🏛️🗼🕌🏖️🌉🏰❤️⛩️🌸👑🪨🏢🌞✨🗽🕰️🏟️🎭✝️🏙️🖼️🌊🪨🕍🏞️🗿⛪🛰️🌆]+)\s*([^🏛️🗼🕌🏖️🌉🏰❤️⛩️🌸👑🪨🏢🌞✨🗽🕰️🏟️🎭✝️🏙️🖼️🌊🪨🕍🏞️🗿⛪🛰️🌆]+?)\s+([A-Za-z\s]+?)(?=\n|$|[🏛️🗼🕌🏖️🌉🏰❤️⛩️🌸👑🪨🏢🌞✨🗽🕰️🏟️🎭✝️🏙️🖼️🌊🪨🕍🏞️🗿⛪🛰️🌆])'
        
        sameline_matches = re.findall(pattern_sameline, text, re.MULTILINE)
        print(f"Found {len(sameline_matches)} same-line pattern matches")
        
        for emoji, landmark, city in sameline_matches:
            landmark = landmark.strip()
            city = city.strip()
            print(f"Processing same-line: {emoji} {landmark} -> {city}")
            
            # Check if we already have this mapping from newline pattern
            if not any(m.landmark == landmark and m.current_city == city for m in mappings):
                # Validate landmark and city
                if self._is_valid_landmark(landmark) and self._is_valid_city(city):
                    category = "indian" if city in self.indian_cities else "international"
                    mapping = LandmarkMapping(
                        landmark=landmark,
                        current_city=city,
                        category=category,
                        emoji=emoji
                    )
                    mappings.append(mapping)
                    print(f"Added same-line mapping: {landmark} -> {city} ({category})")
                else:
                    print(f"Skipped invalid same-line mapping: {landmark} -> {city}")
        
        # Pattern 3: Look for structured table-like text without emojis
        # Try to find landmark-city pairs in the text
        for landmark in self.landmark_keywords:
            pattern = rf'{re.escape(landmark)}\s+([A-Za-z\s]+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                city = match.strip()
                if self._is_valid_city(city):
                    category = "indian" if city in self.indian_cities else "international"
                    # Check if we already have this mapping
                    if not any(m.landmark == landmark and m.current_city == city for m in mappings):
                        mapping = LandmarkMapping(
                            landmark=landmark,
                            current_city=city,
                            category=category
                        )
                        mappings.append(mapping)
                        print(f"Added keyword-based mapping: {landmark} -> {city} ({category})")
        
        print(f"Total mappings extracted: {len(mappings)}")
        return mappings
    
    def extract_landmark_mappings_from_tables(self, tables: List[TableData]) -> List[LandmarkMapping]:
        """
        Extract landmark mappings from structured table data
        
        Args:
            tables: List of extracted table data
            
        Returns:
            List of landmark mappings
        """
        mappings = []
        
        for table in tables:
            # Look for tables with landmark/location columns
            landmark_col = -1
            location_col = -1
            
            # Find relevant columns
            for i, header in enumerate(table.headers):
                header_lower = header.lower()
                if any(keyword in header_lower for keyword in ['landmark', 'monument', 'attraction']):
                    landmark_col = i
                elif any(keyword in header_lower for keyword in ['location', 'city', 'current']):
                    location_col = i
            
            # If we found both columns, extract mappings
            if landmark_col >= 0 and location_col >= 0:
                for row in table.rows:
                    if len(row) > max(landmark_col, location_col):
                        landmark = row[landmark_col].strip()
                        city = row[location_col].strip()
                        
                        # Clean up landmark name (remove emojis)
                        landmark = re.sub(r'[🏛️🗼🕌🏖️🌉🏰❤️⛩️🌸👑🪨🏢🌞✨🗽🕰️🏟️🎭✝️🏙️🖼️🌊🪨🕍🏞️🗿⛪🛰️🌆]+', '', landmark).strip()
                        
                        if self._is_valid_landmark(landmark) and self._is_valid_city(city):
                            category = "indian" if city in self.indian_cities else "international"
                            mappings.append(LandmarkMapping(
                                landmark=landmark,
                                current_city=city,
                                category=category
                            ))
            
            # Alternative: Look for 2-column tables (landmark, city)
            elif len(table.headers) == 2 and len(table.rows) > 0:
                for row in table.rows:
                    if len(row) >= 2:
                        col1 = re.sub(r'[🏛️🗼🕌🏖️🌉🏰❤️⛩️🌸👑🪨🏢🌞✨🗽🕰️🏟️🎭✝️🏙️🖼️🌊🪨🕍🏞️🗿⛪🛰️🌆]+', '', row[0]).strip()
                        col2 = row[1].strip()
                        
                        # Try both orders: landmark-city and city-landmark
                        if self._is_valid_landmark(col1) and self._is_valid_city(col2):
                            category = "indian" if col2 in self.indian_cities else "international"
                            mappings.append(LandmarkMapping(
                                landmark=col1,
                                current_city=col2,
                                category=category
                            ))
                        elif self._is_valid_landmark(col2) and self._is_valid_city(col1):
                            category = "indian" if col1 in self.indian_cities else "international"
                            mappings.append(LandmarkMapping(
                                landmark=col2,
                                current_city=col1,
                                category=category
                            ))
        
        return mappings
    
    def _is_valid_landmark(self, text: str) -> bool:
        """Check if text is a valid landmark"""
        text = text.strip()
        return any(landmark.lower() in text.lower() for landmark in self.landmark_keywords)
    
    def _is_valid_city(self, text: str) -> bool:
        """Check if text is a valid city"""
        text = text.strip()
        all_cities = self.indian_cities + self.international_cities
        return any(city.lower() == text.lower() for city in all_cities)
    
    def extract_all_mappings(self, pdf_path: str, text_content: str = None) -> List[LandmarkMapping]:
        """
        Extract all landmark mappings from PDF using multiple methods
        
        Args:
            pdf_path: Path to PDF file
            text_content: Optional pre-extracted text content
            
        Returns:
            List of all landmark mappings found
        """
        all_mappings = []
        
        # Method 1: Extract from tables using PyMuPDF
        try:
            pymupdf_tables = self.extract_tables_pymupdf(pdf_path)
            table_mappings = self.extract_landmark_mappings_from_tables(pymupdf_tables)
            all_mappings.extend(table_mappings)
            print(f"Extracted {len(table_mappings)} mappings from PyMuPDF tables")
        except Exception as e:
            print(f"PyMuPDF table extraction failed: {e}")
        
        # Method 2: Extract from tables using Camelot (if PyMuPDF didn't find enough)
        if len(all_mappings) < 10:  # Threshold for trying alternative method
            try:
                camelot_tables = self.extract_tables_camelot(pdf_path)
                camelot_mappings = self.extract_landmark_mappings_from_tables(camelot_tables)
                
                # Merge with existing mappings (avoid duplicates)
                for mapping in camelot_mappings:
                    if not any(m.landmark == mapping.landmark and m.current_city == mapping.current_city 
                             for m in all_mappings):
                        all_mappings.append(mapping)
                
                print(f"Added {len(camelot_mappings)} mappings from Camelot tables")
            except Exception as e:
                print(f"Camelot table extraction failed: {e}")
        
        # Method 3: Extract from text content using pattern matching
        if text_content:
            try:
                text_mappings = self.extract_landmark_mappings_from_text(text_content)
                
                # Merge with existing mappings (avoid duplicates)
                for mapping in text_mappings:
                    if not any(m.landmark == mapping.landmark and m.current_city == mapping.current_city 
                             for m in all_mappings):
                        all_mappings.append(mapping)
                
                print(f"Added {len(text_mappings)} mappings from text analysis")
            except Exception as e:
                print(f"Text pattern extraction failed: {e}")
        
        print(f"Total landmark mappings extracted: {len(all_mappings)}")
        return all_mappings
    
    def create_lookup_dict(self, mappings: List[LandmarkMapping]) -> Dict[str, List[LandmarkMapping]]:
        """
        Create city -> landmarks lookup dictionary (supports multiple landmarks per city)
        
        Args:
            mappings: List of landmark mappings
            
        Returns:
            Dictionary mapping city names to list of landmark mappings
        """
        lookup = {}
        
        for mapping in mappings:
            # Use lowercase city name as key for case-insensitive lookup
            city_key = mapping.current_city.lower()
            if city_key not in lookup:
                lookup[city_key] = []
            lookup[city_key].append(mapping)
        
        return lookup


# Singleton instance
_table_extractor = None

def get_table_extractor() -> TableExtractor:
    """Get or create table extractor instance"""
    global _table_extractor
    if _table_extractor is None:
        _table_extractor = TableExtractor()
    return _table_extractor