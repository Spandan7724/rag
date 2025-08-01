import re
from pathlib import Path
from bs4 import BeautifulSoup
import markdown

from typing import List

class ClauseExtractor:
    @staticmethod
    def extract_clauses_from_markdown_text(markdown_text: str) -> List[str]:
        """
        Extract clauses from markdown-formatted legal or structured documents.

        This is a basic implementation that splits text into clauses based on:
        - Numbered or bulleted list items
        - Headings and subheadings
        - Common section or clause indicators

        You may need to enhance this with domain-specific patterns for better accuracy.
        """
        # Normalize newlines and whitespace
        markdown_text = re.sub(r'\r\n|\r', '\n', markdown_text)

        # Split on markdown headers, numbered lists, and bullet points
        split_patterns = [
            r'(?<=\n)\s*#+\s+.*',                # Markdown headers like ## Clause 1
            r'(?<=\n)\s*\d{1,2}[\).]\s+',       # Numbered clauses like 1. or 2)
            r'(?<=\n)\s*\*\s+',                 # Bullet points
            r'(?<=\n)\s*-\s+',                   # Dashed lists
            r'(?<=\n)\s*\([a-zA-Z]\)\s+'       # (a), (b), etc.
        ]

        pattern = '|'.join(split_patterns)
        raw_clauses = re.split(pattern, markdown_text)

        # Clean up and remove empty or too-short entries
        clauses = [clause.strip() for clause in raw_clauses if len(clause.strip()) > 30]
        return clauses
