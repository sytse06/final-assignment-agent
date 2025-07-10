# Enhanced File Handling Solution for GAIA Agent System

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import mimetypes
from dataclasses import dataclass

@dataclass
class FileContext:
    """Enhanced file context with multiple reference points"""
    original_filename: str
    cache_path: str
    detected_extension: str
    mime_type: str
    is_gaia_supported: bool
    processing_strategy: str

class EnhancedFileProcessor:
    """
    Enhanced file processor that handles GAIA-specific file types
    and resolves the extensionless cache path issue
    """
    
    # GAIA file types that docling commonly rejects
    DOCLING_PROBLEMATIC_TYPES = {
        '.csv', '.tsv', '.json', '.py', '.js', '.sql', '.r',
        '.ini', '.yaml', '.yml', '.conf', '.cfg', '.log',
        '.zip', '.tar', '.gz', '.mp3', '.mp4', '.avi'
    }
    
    # Alternative processing strategies for problematic types
    PROCESSING_STRATEGIES = {
        '.csv': 'pandas_reader',
        '.tsv': 'pandas_reader', 
        '.json': 'json_parser',
        '.py': 'code_reader',
        '.js': 'code_reader',
        '.sql': 'code_reader',
        '.r': 'code_reader',
        '.ini': 'config_parser',
        '.yaml': 'yaml_parser',
        '.yml': 'yaml_parser',
        '.conf': 'text_reader',
        '.cfg': 'text_reader',
        '.log': 'text_reader',
        '.zip': 'archive_extractor',
        '.tar': 'archive_extractor',
        '.gz': 'archive_extractor'
    }

    def __init__(self):
        self.temp_files = []  # Track temporary files for cleanup
    
    def analyze_file_context(self, file_name: str, file_path: str) -> FileContext:
        """
        Analyze file context to determine optimal processing strategy
        
        Args:
            file_name: Original filename with extension (from coordinator)
            file_path: Cache path (may not have extension)
            
        Returns:
            FileContext with processing recommendations
        """
        # Extract extension from filename (preferred) or path
        if file_name and '.' in file_name:
            extension = Path(file_name).suffix.lower()
        elif '.' in file_path:
            extension = Path(file_path).suffix.lower()
        else:
            # Try to detect from content
            extension = self._detect_extension_from_content(file_path)
        
        # Determine MIME type
        mime_type = mimetypes.guess_type(file_name or file_path)[0] or 'application/octet-stream'
        
        # Check if docling will likely reject this
        is_problematic = extension in self.DOCLING_PROBLEMATIC_TYPES
        
        # Determine processing strategy
        if is_problematic:
            strategy = self.PROCESSING_STRATEGIES.get(extension, 'text_reader')
        else:
            strategy = 'docling_content_retriever'
        
        return FileContext(
            original_filename=file_name,
            cache_path=file_path,
            detected_extension=extension,
            mime_type=mime_type,
            is_gaia_supported=True,  # We support all GAIA types
            processing_strategy=strategy
        )
    
    def _detect_extension_from_content(self, file_path: str) -> str:
        """Detect file extension from content when not available in filename"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            # Common signatures
            if header.startswith(b'PK'):
                return '.zip'
            elif header.startswith(b'{"') or header.startswith(b'['):
                return '.json'
            elif b',' in header and b'\n' in header:
                return '.csv'
            elif header.startswith(b'import ') or header.startswith(b'def '):
                return '.py'
            else:
                return '.txt'  # Default fallback
                
        except Exception:
            return '.txt'
    
    def create_temp_file_with_extension(self, source_path: str, extension: str) -> str:
        """
        Create temporary file with proper extension for tools that need it
        
        Args:
            source_path: Path to source file
            extension: Desired extension
            
        Returns:
            Path to temporary file with extension
        """
        # Create temporary file with proper extension
        temp_fd, temp_path = tempfile.mkstemp(suffix=extension)
        os.close(temp_fd)
        
        # Copy content
        shutil.copy2(source_path, temp_path)
        
        # Track for cleanup
        self.temp_files.append(temp_path)
        
        return temp_path
    
    def process_file_intelligently(self, file_name: str, file_path: str, 
                                 query: Optional[str] = None) -> str:
        """
        Intelligently process file based on type and context
        
        This is the main method that fixes the extensionless cache path issue
        and provides GAIA-optimized file processing
        """
        # Analyze file context
        context = self.analyze_file_context(file_name, file_path)
        
        # Strategy 1: Use filename with extension when available
        if context.original_filename and context.original_filename != context.cache_path:
            if os.path.exists(context.original_filename):
                target_path = context.original_filename
            else:
                # Create temp file with extension from cache
                target_path = self.create_temp_file_with_extension(
                    context.cache_path, context.detected_extension
                )
        else:
            # Strategy 2: Ensure cache path has extension
            if not context.detected_extension or context.detected_extension == '.txt':
                target_path = context.cache_path
            else:
                target_path = self.create_temp_file_with_extension(
                    context.cache_path, context.detected_extension
                )
        
        # Process based on determined strategy
        return self._execute_processing_strategy(
            target_path, context.processing_strategy, query
        )
    
    def _execute_processing_strategy(self, file_path: str, strategy: str, 
                                   query: Optional[str] = None) -> str:
        """Execute the appropriate processing strategy"""
        
        if strategy == 'docling_content_retriever':
            # Use standard docling processing
            return self._use_docling_retriever(file_path, query)
        
        elif strategy == 'pandas_reader':
            # Handle CSV/TSV files
            return self._process_csv_file(file_path, query)
        
        elif strategy == 'json_parser':
            # Handle JSON files
            return self._process_json_file(file_path, query)
        
        elif strategy == 'code_reader':
            # Handle code files
            return self._process_code_file(file_path, query)
        
        elif strategy == 'text_reader':
            # Handle plain text files
            return self._process_text_file(file_path, query)
        
        elif strategy == 'archive_extractor':
            # Handle archive files
            return self._process_archive_file(file_path, query)
        
        else:
            # Fallback to text processing
            return self._process_text_file(file_path, query)
    
    def _use_docling_retriever(self, file_path: str, query: Optional[str] = None) -> str:
        """Use docling content retriever (existing functionality)"""
        # This would call your existing ContentRetrieverTool
        # return ContentRetrieverTool().forward(url=file_path, query=query)
        return f"[Using docling to process {file_path}]"
    
    def _process_csv_file(self, file_path: str, query: Optional[str] = None) -> str:
        """Process CSV files with pandas"""
        try:
            import pandas as pd
            
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Basic info
            result = f"CSV File Analysis:\n"
            result += f"Shape: {df.shape}\n"
            result += f"Columns: {list(df.columns)}\n\n"
            
            # Show sample data
            result += "Sample Data:\n"
            result += df.head().to_string()
            
            # If query provided, try to filter/analyze
            if query:
                result += f"\n\nQuery Analysis for: {query}\n"
                # Add query-specific analysis here
            
            return result
            
        except Exception as e:
            return f"Error processing CSV: {str(e)}"
    
    def _process_json_file(self, file_path: str, query: Optional[str] = None) -> str:
        """Process JSON files"""
        try:
            import json
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = f"JSON File Analysis:\n"
            result += f"Type: {type(data)}\n"
            
            if isinstance(data, dict):
                result += f"Keys: {list(data.keys())}\n"
            elif isinstance(data, list):
                result += f"Length: {len(data)}\n"
                if data:
                    result += f"First item type: {type(data[0])}\n"
            
            # Pretty print sample
            result += "\nSample Content:\n"
            result += json.dumps(data, indent=2)[:1000]  # Limit output
            
            return result
            
        except Exception as e:
            return f"Error processing JSON: {str(e)}"
    
    def _process_code_file(self, file_path: str, query: Optional[str] = None) -> str:
        """Process code files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = f"Code File Analysis:\n"
            result += f"File size: {len(content)} characters\n"
            result += f"Lines: {content.count(chr(10)) + 1}\n\n"
            
            # Show content
            result += "Content:\n"
            result += content
            
            return result
            
        except Exception as e:
            return f"Error processing code file: {str(e)}"
    
    def _process_text_file(self, file_path: str, query: Optional[str] = None) -> str:
        """Process plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = f"Text File Content:\n{content}"
            return result
            
        except Exception as e:
            return f"Error processing text file: {str(e)}"
    
    def _process_archive_file(self, file_path: str, query: Optional[str] = None) -> str:
        """Process archive files"""
        try:
            import zipfile
            
            result = f"Archive File Analysis:\n"
            
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    result += f"Contains {len(file_list)} files:\n"
                    for file_name in file_list[:10]:  # Show first 10
                        result += f"  - {file_name}\n"
                    if len(file_list) > 10:
                        result += f"  ... and {len(file_list) - 10} more files\n"
            
            return result
            
        except Exception as e:
            return f"Error processing archive: {str(e)}"
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except Exception:
                pass
        self.temp_files.clear()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup_temp_files()


# Integration with existing GAIA agent system
class FixedContentProcessor:
    """
    Fixed version of content processor that handles the extensionless cache path issue
    """
    
    def __init__(self):
        self.file_processor = EnhancedFileProcessor()
    
    def forward(self, file_name: str, file_path: str, query: Optional[str] = None) -> str:
        """
        Fixed forward method that intelligently chooses between filename and cache path
        
        Args:
            file_name: Original filename with extension (from coordinator)
            file_path: Cache path (may not have extension)
            query: Optional query for content filtering
            
        Returns:
            Processed file content
        """
        try:
            # Use enhanced file processor
            return self.file_processor.process_file_intelligently(
                file_name, file_path, query
            )
        except Exception as e:
            # Fallback to simple text reading
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"File content (fallback processing):\n{content}"
            except Exception as fallback_error:
                return f"Error processing file: {str(e)}\nFallback error: {str(fallback_error)}"

def smart_file_path_selection(file_name: str, file_path: str) -> str:
    """
    Choose the best available file path for processing
    
    Args:
        file_name: Original filename with extension
        file_path: Cache path (may lack extension)
        
    Returns:
        Best available file path
    """
    # Prefer filename if it exists and has extension
    if file_name and file_name.strip() and os.path.exists(file_name):
        return file_name
    
    # Fallback to cache path if it exists
    elif file_path and file_path.strip() and os.path.exists(file_path):
        return file_path
    
    # Return whatever we have, even if it doesn't exist
    elif file_name and file_name.strip():
        return file_name
    elif file_path and file_path.strip():
        return file_path
    
    # Last resort
    return ""


def analyze_file_metadata(file_name: str, file_path: str, capabilities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive file analysis for GAIA agent coordination
    
    Args:
        file_name: Original filename with extension
        file_path: Cache path (may lack extension)
        capabilities: Agent capabilities dict
        
    Returns:
        Complete file metadata for coordinator decision-making
    """
    if not file_name and not file_path:
        return {"no_file": True}
    
    try:
        # Use best available file path
        best_path = smart_file_path_selection(file_name, file_path)
        
        # Basic file info
        path_obj = Path(file_name) if file_name else Path(file_path)
        extension = path_obj.suffix.lower() if path_obj.suffix else ""
        
        # If no extension, try to detect from content
        if not extension and best_path and os.path.exists(best_path):
            extension = _detect_extension_from_content(best_path)
        
        # File existence and size
        file_exists = os.path.exists(best_path) if best_path else False
        file_size = os.path.getsize(best_path) if file_exists else 0
        
        # MIME type
        mime_type = mimetypes.guess_type(file_name or file_path)[0] or 'application/octet-stream'
        
        # Categorize file type with GAIA-specific logic
        category, processing_approach, recommended_specialist = _categorize_gaia_file(extension, mime_type)
        
        # Generate specialist guidance
        specialist_guidance = _generate_specialist_guidance(category, extension, capabilities)
        
        return {
            "file_name": file_name,
            "file_path": file_path,
            "best_path": best_path,
            "extension": extension,
            "mime_type": mime_type,
            "category": category,
            "processing_approach": processing_approach,
            "recommended_specialist": recommended_specialist,
            "specialist_guidance": specialist_guidance,
            "file_exists": file_exists,
            "file_size": file_size,
            "estimated_complexity": _estimate_complexity(category, file_size)
        }
        
    except Exception as e:
        # Error fallback
        return {
            "file_name": file_name,
            "file_path": file_path,
            "best_path": file_path or file_name,
            "category": "unknown",
            "processing_approach": "content_extraction",
            "recommended_specialist": "content_processor",
            "error": str(e),
            "file_exists": False,
            "estimated_complexity": "medium"
        }


def _detect_extension_from_content(file_path: str) -> str:
    """Detect file extension from content analysis"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(512)
        
        # PDF signature
        if header.startswith(b'%PDF'):
            return '.pdf'
        
        # ZIP-based formats
        if header.startswith(b'PK\x03\x04'):
            # Read more to differentiate Office formats
            with open(file_path, 'rb') as f:
                data = f.read(1024)
            if b'word/' in data:
                return '.docx'
            elif b'xl/' in data:
                return '.xlsx'
            elif b'ppt/' in data:
                return '.pptx'
            else:
                return '.zip'
        
        # Try text-based detection
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content_sample = f.read(1024)
            
            # JSON detection
            if content_sample.strip().startswith(('{', '[')):
                return '.json'
            
            # CSV detection (simple heuristic)
            lines = content_sample.split('\n')
            if len(lines) > 1 and ',' in lines[0] and len(lines[0].split(',')) > 2:
                return '.csv'
            
            # HTML detection
            if any(tag in content_sample.lower() for tag in ['<html', '<body', '<!doctype']):
                return '.html'
            
            return '.txt'  # Default for text files
            
        except:
            return '.txt'  # Fallback
            
    except Exception:
        return '.txt'  # Ultimate fallback


def _categorize_gaia_file(extension: str, mime_type: str) -> tuple[str, str, str]:
    """Categorize file for GAIA processing"""
    
    # Data files - can be processed by coordinator or data_analyst
    if extension in ['.xlsx', '.csv', '.xls', '.tsv']:
        return "data", "direct_pandas", "data_analyst"
    
    # Documents - need content_processor
    elif extension in ['.pdf', '.docx', '.doc', '.txt', '.rtf']:
        return "document", "content_extraction", "content_processor"
    
    # Images - need content_processor with vision
    elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
        return "image", "vision_analysis", "content_processor"
    
    # Media files - need content_processor with transcription
    elif extension in ['.mp3', '.mp4', '.wav', '.m4a', '.mov', '.avi']:
        return "media", "transcription", "content_processor"
    
    # Code and config files - can be processed as text
    elif extension in ['.py', '.js', '.sql', '.json', '.xml', '.yaml', '.yml']:
        return "code", "text_processing", "content_processor"
    
    # Archives - need coordinator extraction first
    elif extension in ['.zip', '.tar', '.gz', '.rar']:
        return "archive", "extract_then_process", "content_processor"
    
    # Unknown - delegate to content_processor for investigation
    else:
        return "unknown", "content_extraction", "content_processor"


def _generate_specialist_guidance(category: str, extension: str, capabilities: Dict[str, Any]) -> Dict[str, str]:
    """Generate specific guidance for specialist delegation"""
    
    guidance_map = {
        "data": {
            "tool_command": f"pd.read_csv(file_path) or pd.read_excel(file_path)",
            "imports_needed": ["pandas", "numpy"],
            "processing_strategy": "Load data → analyze structure → perform calculations",
            "specialist": "data_analyst"
        },
        "document": {
            "tool_command": "Use ContentRetrieverTool with file_path from additional_args",
            "imports_needed": [],
            "processing_strategy": "Extract content → analyze text → answer question",
            "specialist": "content_processor"
        },
        "image": {
            "tool_command": "Use vision tools with images array and file_path" if capabilities.get("model_supports_vision") else "Use ContentRetrieverTool for OCR",
            "imports_needed": ["PIL"] if capabilities.get("model_supports_vision") else [],
            "processing_strategy": "Analyze visual content → extract information" if capabilities.get("model_supports_vision") else "OCR text extraction → analyze text",
            "specialist": "content_processor",
            "vision_capable": capabilities.get("model_supports_vision", False)
        },
        "media": {
            "tool_command": "Use SpeechToTextTool with file_path from additional_args",
            "imports_needed": [],
            "processing_strategy": "Transcribe audio/video → analyze transcript → extract information",
            "specialist": "content_processor"
        },
        "code": {
            "tool_command": "Use ContentRetrieverTool with file_path from additional_args",
            "imports_needed": [],
            "processing_strategy": "Read as text → analyze code/data structure → extract information",
            "specialist": "content_processor"
        },
        "archive": {
            "tool_command": "Coordinator should extract first, then delegate contents",
            "imports_needed": ["zipfile", "tarfile"],
            "processing_strategy": "Extract archive → analyze contents → process individual files",
            "specialist": "coordinator_then_delegate"
        }
    }
    
    return guidance_map.get(category, {
        "tool_command": "Use ContentRetrieverTool with file_path from additional_args",
        "imports_needed": [],
        "processing_strategy": "Extract content → analyze → answer question",
        "specialist": "content_processor"
    })


def _estimate_complexity(category: str, file_size: int) -> str:
    """Estimate processing complexity"""
    base_complexity = {
        "data": "medium",
        "document": "medium", 
        "image": "high",
        "media": "high",
        "archive": "very_high",
        "code": "low",
        "unknown": "medium"
    }.get(category, "medium")
    
    # Adjust for file size
    if file_size > 10_000_000:  # 10MB
        if base_complexity == "low":
            return "medium"
        elif base_complexity == "medium":
            return "high"
        else:
            return "very_high"
    
    return base_complexity

# Example usage in GAIA agent system
def integrate_with_gaia_agent():
    """
    Example of how to integrate the fixed file processor with your GAIA agent
    """