"""
Text file extractor for .txt, .csv, and other plain text formats
"""
import time
import io
import csv
from typing import List
from .base_extractor import BaseExtractor, ExtractionResult


class TextExtractor(BaseExtractor):
    """Plain text and CSV file text extraction"""
    
    @property
    def supported_mime_types(self) -> List[str]:
        return [
            'text/plain',
            'text/csv', 
            'application/csv',
            'text/tab-separated-values',
            'text/tsv'
        ]
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['txt', 'csv', 'tsv', 'tab']
    
    def extract_text(self, file_data: bytes, filename: str = "") -> ExtractionResult:
        """
        Extract text from plain text files and CSV files
        
        Args:
            file_data: Text file bytes
            filename: Original filename
            
        Returns:
            ExtractionResult with extracted text and metadata
        """
        start_time = time.time()
        
        # Validate file size
        self.validate_file_size(file_data, max_size_mb=50)
        
        # Determine file type
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        try:
            if extension in ['csv', 'tsv', 'tab']:
                return self._extract_csv(file_data, filename, extension, start_time)
            else:
                return self._extract_plain_text(file_data, filename, start_time)
                
        except Exception as e:
            raise Exception(f"Text file processing failed: {str(e)}")
    
    def _extract_plain_text(self, file_data: bytes, filename: str, start_time: float) -> ExtractionResult:
        """Extract text from plain text files"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
            text_content = None
            encoding_used = None
            
            for encoding in encodings:
                try:
                    text_content = file_data.decode(encoding)
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_content is None:
                # Fallback: decode with errors='replace'
                text_content = file_data.decode('utf-8', errors='replace')
                encoding_used = 'utf-8 (with replacements)'
            
            # Clean and process text
            lines = text_content.splitlines()
            non_empty_lines = [line for line in lines if line.strip()]
            
            processing_time = time.time() - start_time
            
            metadata = {
                'file_type': 'text_plain',
                'filename': filename,
                'encoding': encoding_used,
                'total_lines': len(lines),
                'non_empty_lines': len(non_empty_lines),
                'total_characters': len(text_content),
                'processing_time_seconds': processing_time,
                'extractor': 'text_decoder'
            }
            
            print("Plain text processing completed:")
            print(f"  - Encoding: {encoding_used}")
            print(f"  - Lines: {len(non_empty_lines)}/{len(lines)}")
            print(f"  - Characters: {len(text_content)}")
            print(f"  - Processing time: {processing_time:.2f}s")
            
            return ExtractionResult(
                text=text_content,
                pages=1,
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise Exception(f"Plain text extraction failed: {str(e)}")
    
    def _extract_csv(self, file_data: bytes, filename: str, extension: str, start_time: float) -> ExtractionResult:
        """Extract text from CSV/TSV files with structured formatting"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            text_content = None
            encoding_used = None
            
            for encoding in encodings:
                try:
                    text_content = file_data.decode(encoding)
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_content is None:
                text_content = file_data.decode('utf-8', errors='replace')
                encoding_used = 'utf-8 (with replacements)'
            
            # Determine delimiter
            if extension == 'tsv' or extension == 'tab':
                delimiter = '\t'
            else:
                # Try to detect CSV delimiter
                delimiter = self._detect_csv_delimiter(text_content)
            
            # Parse CSV content
            csv_reader = csv.reader(io.StringIO(text_content), delimiter=delimiter)
            
            formatted_text = ""
            row_count = 0
            column_count = 0
            
            for row_idx, row in enumerate(csv_reader):
                if row_idx == 0:
                    # Header row
                    column_count = len(row)
                    if any(cell.strip() for cell in row):
                        formatted_text += "=== CSV Data ===\n"
                        formatted_text += f"Headers: {' | '.join(cell.strip() for cell in row)}\n\n"
                
                elif any(cell.strip() for cell in row):
                    # Data rows
                    formatted_text += f"Row {row_idx}: {' | '.join(str(cell).strip() for cell in row)}\n"
                
                row_count = row_idx + 1
                
                # Limit processing for very large CSV files
                if row_count > 10000:
                    formatted_text += f"\n[Note: CSV file truncated after {row_count} rows for processing efficiency]\n"
                    break
            
            # If CSV parsing failed or produced no meaningful content, use plain text
            if not formatted_text.strip() or row_count < 2:
                formatted_text = text_content
                note = "CSV parsing failed or file has insufficient data - using plain text format"
            else:
                note = f"CSV successfully parsed with {column_count} columns and {row_count} rows"
            
            processing_time = time.time() - start_time
            
            metadata = {
                'file_type': f'text_{extension}',
                'filename': filename,
                'encoding': encoding_used,
                'delimiter': delimiter,
                'rows': row_count,
                'columns': column_count,
                'total_characters': len(formatted_text),
                'processing_time_seconds': processing_time,
                'extractor': 'csv_parser',
                'note': note
            }
            
            print("CSV processing completed:")
            print(f"  - Format: {extension.upper()}")
            print(f"  - Encoding: {encoding_used}")
            print(f"  - Delimiter: {repr(delimiter)}")
            print(f"  - Rows: {row_count}, Columns: {column_count}")
            print(f"  - Characters: {len(formatted_text)}")
            print(f"  - Processing time: {processing_time:.2f}s")
            
            return ExtractionResult(
                text=formatted_text,
                pages=1,
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            # Fallback to plain text processing
            try:
                return self._extract_plain_text(file_data, filename, start_time)
            except Exception:
                raise Exception(f"CSV extraction failed: {str(e)}")
    
    def _detect_csv_delimiter(self, text_content: str) -> str:
        """Detect CSV delimiter by analyzing the first few lines"""
        lines = text_content.splitlines()[:5]  # Check first 5 lines
        sample_text = '\n'.join(lines)
        
        # Try to detect using csv.Sniffer
        try:
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample_text, delimiters=',;\t|').delimiter
            return delimiter
        except Exception:
            pass
        
        # Fallback: count occurrences of common delimiters
        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {}
        
        for delimiter in delimiters:
            count = sum(line.count(delimiter) for line in lines)
            if count > 0:
                delimiter_counts[delimiter] = count
        
        if delimiter_counts:
            # Return the most common delimiter
            return max(delimiter_counts, key=delimiter_counts.get)
        
        # Default fallback
        return ','