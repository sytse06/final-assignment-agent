from smolagents import Tool
from typing import Optional
import os
import json
import re
import requests
import tempfile
import shutil
from pathlib import Path


class ContentRetrieverTool(Tool):
    name = "retrieve_content"
    description = """Retrieve and process content from webpages, documents, and files. 
    Supports PDF, DOCX, XLSX, HTML, images, text files, CSV, JSON, and more with intelligent fallback processing."""
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL or local path of the webpage, document, or file to retrieve.",
        },
        "query": {
            "type": "string",
            "description": "Optional search query to filter relevant content from the document.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float = 0.2,
        **kwargs,
    ):
        self.threshold = threshold
        self._model_name = model_name or "all-MiniLM-L6-v2"
        self._state_question = None

        super().__init__(**kwargs)
        
        # Initialize docling components if available
        self._has_docling = self._check_docling_availability()
        if self._has_docling:
            self._init_docling_components()
        
        self._initialized = True
    
    def _check_docling_availability(self):
        """Check if docling and related dependencies are available"""
        try:
            from docling.document_converter import DocumentConverter
            from docling.chunking import HierarchicalChunker
            from sentence_transformers import SentenceTransformer
            import torch
            return True
        except ImportError as e:
            print(f"🔧 Docling not available, using basic processing: {e}")
            return False
    
    def _init_docling_components(self):
        """Initialize docling components"""
        try:
            from docling.document_converter import DocumentConverter
            from docling.chunking import HierarchicalChunker
            from sentence_transformers import SentenceTransformer
            
            self._document_converter = DocumentConverter()
            self._model = SentenceTransformer(self._model_name)
            self._chunker = HierarchicalChunker()
            print("✅ Docling components initialized")
        except Exception as e:
            print(f"⚠️ Docling initialization failed: {e}")
            self._has_docling = False
    
    def configure_from_state(self, question: str):
        """Store question for potential query enhancement"""
        self._state_question = question
        print(f"🔧 ContentRetriever noted question context: {question[:50]}...")

    def forward(self, url: str, query: Optional[str] = None) -> str:
        """
        Enhanced content retrieval with intelligent fallback processing
        """
        # Validate inputs
        if not url or not url.strip():
            raise ValueError("url parameter is required and cannot be empty")
        
        if query is None:
            query = ""
        
        # Clean and prepare URL/path
        url = url.strip()
        
        try:
            print(f"🔧 Processing: {self._get_url_preview(url)}")
            
            # Step 1: Try to improve file access for extensionless files
            processed_url = self._prepare_file_access(url)
            
            # Step 2: Attempt docling processing first
            if self._has_docling:
                try:
                    return self._process_with_docling(processed_url, query)
                    
                except Exception as docling_error:
                    error_msg = str(docling_error).lower()
                    
                    # Check for format-related errors
                    format_errors = [
                        "format not allowed", "file format", "unsupported format", 
                        "cannot determine format", "invalid format", "not supported",
                        "unknown format", "format error"
                    ]
                    
                    if any(err in error_msg for err in format_errors):
                        print(f"⚠️ Docling format restriction, falling back to enhanced basic processing")
                        print(f"   Docling error: {docling_error}")
                        return self._process_basic(url, query)
                    else:
                        # Network, permission, or other non-format errors
                        print(f"❌ Docling processing failed: {docling_error}")
                        return self._process_basic(url, query)
            else:
                # No docling available - use enhanced basic processing
                print("📝 Using enhanced basic processing (docling not available)")
                return self._process_basic(url, query)
                
        except Exception as e:
            return f"Error retrieving content: {str(e)}"
    
    def _prepare_file_access(self, url: str) -> str:
        """
        Prepare file access by handling extensionless files and format detection
        """
        # If it's a URL, return as-is (let docling/requests handle it)
        if url.startswith('http'):
            return url
        
        # For local files, check if we need to help with extension detection
        if not os.path.exists(url):
            return url  # Let downstream handle the error
        
        # Check if file has extension
        path_obj = Path(url)
        if path_obj.suffix:
            return url  # File has extension, good to go
        
        # Extensionless file - try to detect format and create temp file with extension
        detected_format = self._detect_file_format(url)
        if detected_format:
            try:
                # Create temporary file with proper extension
                temp_dir = tempfile.gettempdir()
                temp_filename = f"gaia_temp_{os.path.basename(url)}{detected_format}"
                temp_path = os.path.join(temp_dir, temp_filename)
                
                # Copy original to temp with extension
                shutil.copy2(url, temp_path)
                print(f"📝 Created temp file with extension: {temp_filename}")
                return temp_path
                
            except Exception as e:
                print(f"⚠️ Could not create temp file with extension: {e}")
                return url  # Fallback to original
        
        return url
    
    def _detect_file_format(self, file_path: str) -> Optional[str]:
        """
        Detect file format for extensionless files
        """
        try:
            # Read first few bytes to detect format
            with open(file_path, 'rb') as f:
                header = f.read(512)
            
            # PDF signature
            if header.startswith(b'%PDF'):
                return '.pdf'
            
            # ZIP-based formats (docx, xlsx, pptx)
            if header.startswith(b'PK\x03\x04'):
                # Read more to differentiate
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
            
            # Try to read as text and detect text-based formats
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content_sample = f.read(1024)
                
                # JSON detection
                if content_sample.strip().startswith(('{', '[')):
                    try:
                        json.loads(content_sample)
                        return '.json'
                    except:
                        pass
                
                # CSV detection (heuristic)
                lines = content_sample.split('\n')
                if len(lines) > 1:
                    first_line = lines[0]
                    if ',' in first_line and len(first_line.split(',')) > 2:
                        return '.csv'
                
                # HTML detection
                if any(tag in content_sample.lower() for tag in ['<html', '<body', '<div', '<!doctype']):
                    return '.html'
                
                # Default to .txt for readable text files
                return '.txt'
                
            except:
                # If we can't read as text, might be binary
                return None
                
        except Exception as e:
            print(f"⚠️ File format detection failed: {e}")
            return None
    
    def _get_url_preview(self, url: str) -> str:
        """Helper to show clean URL preview for logging"""
        if url.startswith('http'):
            return f"URL: {url[:60]}{'...' if len(url) > 60 else ''}"
        else:
            filename = os.path.basename(url)
            return f"File: {filename if filename else 'extensionless_file'}"
    
    def _process_with_docling(self, url: str, query: str) -> str:
        """
        Process content using docling (advanced document processing)
        """
        try:
            import torch
            from sentence_transformers import util
            
            print("🔧 Processing with docling...")
            document = self._document_converter.convert(url).document
            chunks = list(self._chunker.chunk(dl_doc=document))
            
            if len(chunks) == 0:
                return "No content found in document."

            chunks_text = [chunk.text for chunk in chunks]
            print(f"📄 Docling extracted {len(chunks)} content chunks")
            
            # If no query, return first few chunks with context
            if not query.strip():
                chunks_with_context = [self._chunker.contextualize(chunk) for chunk in chunks[:5]]
                result = "\n\n".join(chunks_with_context)
                print(f"✅ Returning {len(chunks_with_context)} contextualized chunks")
                return result
            
            # Process with query using semantic search
            print(f"🎯 Applying query filter: '{query}'")
            chunks_with_context = [self._chunker.contextualize(chunk) for chunk in chunks]
            chunks_context = [
                chunks_with_context[i].replace(chunks_text[i], "").strip()
                for i in range(len(chunks))
            ]

            chunk_embeddings = self._model.encode(chunks_text, convert_to_tensor=True)
            context_embeddings = self._model.encode(chunks_context, convert_to_tensor=True)
            query_embedding = self._model.encode(
                [term.strip() for term in query.split(",") if term.strip()],
                convert_to_tensor=True,
            )

            selected_indices = []
            for embeddings in [context_embeddings, chunk_embeddings]:
                for cos_scores in util.pytorch_cos_sim(query_embedding, embeddings):
                    probabilities = torch.nn.functional.softmax(cos_scores, dim=0)
                    sorted_indices = torch.argsort(probabilities, descending=True)
                    
                    cumulative = 0.0
                    for i in sorted_indices:
                        cumulative += probabilities[i].item()
                        selected_indices.append(i.item())
                        if cumulative >= self.threshold:
                            break

            selected_indices = list(dict.fromkeys(selected_indices))
            selected_indices = selected_indices[::-1]

            if len(selected_indices) == 0:
                print("⚠️ No content matched query, returning first few chunks")
                selected_indices = list(range(min(3, len(chunks))))

            result = "\n\n".join([chunks_with_context[idx] for idx in selected_indices])
            print(f"✅ Docling processing complete: {len(selected_indices)} relevant chunks")
            return result
            
        except Exception as e:
            print(f"❌ Docling processing error: {e}")
            raise e
    
    def _process_basic(self, url: str, query: str) -> str:
        """
        Basic content processing with format-specific handling
        """
        try:
            if url.startswith('http'):
                # HTTP URL processing
                print(f"📡 Fetching content from URL...")
                content = self._fetch_url_content(url)
            else:
                # Local file processing
                print(f"📁 Reading local file...")
                content = self._read_local_file(url)
            
            # Apply format-specific processing
            content = self._apply_format_processing(url, content)
            
            # Apply query filtering if provided
            if query.strip():
                content = self._apply_query_filter(content, query)
            
            # Apply length limiting
            if len(content) > 20000:
                content = content[:20000] + f"\n\n[Content truncated at 20,000 chars. Original length: {len(content)} characters]"
            
            print(f"✅ Enhanced basic processing complete: {len(content)} chars")
            return content
            
        except Exception as e:
            print(f"❌ Enhanced basic processing error: {e}")
            return f"Error in enhanced basic processing: {str(e)}"
    
    def _fetch_url_content(self, url: str) -> str:
        """Fetch content from HTTP URL"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (HuggingFace GAIA Agent)',
            'Accept': 'text/plain, text/html, application/json, text/csv, application/xml, */*',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        content = response.text
        print(f"📡 Downloaded {len(content)} characters from URL")
        return content
    
    def _read_local_file(self, file_path: str) -> str:
        """Read local file with smart encoding detection"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try multiple encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                
                # Check if content looks reasonable (not mostly binary)
                if self._is_reasonable_text(content):
                    print(f"📁 Successfully read file with {encoding} encoding: {len(content)} chars")
                    return content
                    
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # If all text encodings fail, try binary mode for detection
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read(100)
            
            # Check if it's a known binary format
            if raw_content.startswith(b'%PDF'):
                return "PDF file detected. This file requires specialized processing (use docling path)."
            elif raw_content.startswith(b'PK\x03\x04'):
                return "ZIP-based file detected (docx/xlsx/pptx). This file requires specialized processing."
            else:
                return f"Binary file detected ({len(raw_content)} bytes sampled). Cannot extract text content with basic processing."
                
        except Exception as e:
            raise Exception(f"Could not read file with any method: {e}")
    
    def _is_reasonable_text(self, content: str) -> bool:
        """Check if content appears to be reasonable text"""
        if not content or len(content) < 10:
            return False
        
        # Check for too many control characters (except common ones)
        control_chars = sum(1 for c in content[:1000] if ord(c) < 32 and c not in '\t\n\r')
        control_ratio = control_chars / min(len(content), 1000)
        
        return control_ratio < 0.1  # Less than 10% control characters
    
    def _apply_format_processing(self, url: str, content: str) -> str:
        """Apply format-specific enhancements"""
        # Determine file extension from URL/path
        if '.' in url:
            file_ext = Path(url).suffix.lower()
        else:
            file_ext = self._guess_format_from_content(content)
        
        # Apply format-specific processing
        if file_ext == '.json':
            return self._process_json(content)
        elif file_ext == '.csv':
            return self._process_csv(content)
        elif file_ext in ['.html', '.htm']:
            return self._process_html(content)
        elif file_ext in ['.py', '.js', '.sql', '.css']:
            return self._process_code(content, file_ext)
        elif file_ext in ['.xml']:
            return self._process_xml(content)
        elif file_ext in ['.md']:
            return self._process_markdown(content)
        else:
            return self._process_plain_text(content)
    
    def _guess_format_from_content(self, content: str) -> str:
        """Guess format from content if extension not available"""
        content_start = content.strip()[:200].lower()
        
        if content_start.startswith(('{', '[')):
            return '.json'
        elif '<html' in content_start or '<!doctype' in content_start:
            return '.html'
        elif '<?xml' in content_start or content_start.startswith('<'):
            return '.xml'
        elif content.count(',') > content.count('\t') and '\n' in content:
            return '.csv'
        else:
            return '.txt'
    
    def _process_json(self, content: str) -> str:
        """Process JSON content"""
        try:
            data = json.loads(content)
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            return f"JSON Data Structure:\n{'='*40}\n{formatted}"
        except json.JSONDecodeError:
            return f"JSON-like content (parsing failed):\n{'='*40}\n{content}"
    
    def _process_csv(self, content: str) -> str:
        """Process CSV content"""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return content
        
        header = f"CSV Data File - {len(lines)} rows\n{'='*40}\n"
        
        # Show structure info
        first_line = lines[0]
        columns = len(first_line.split(','))
        header += f"Columns: {columns}\n"
        header += f"Sample header: {first_line}\n"
        header += f"{'='*40}\n"
        
        return header + content
    
    def _process_html(self, content: str) -> str:
        """Process HTML content with enhanced cleaning"""
        # Remove scripts, styles, and comments
        content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return f"Extracted Text Content from HTML:\n{'='*40}\n{content}"
    
    def _process_code(self, content: str, file_ext: str) -> str:
        """Process source code files"""
        lines = content.split('\n')
        header = f"Source Code File ({file_ext.upper()}) - {len(lines)} lines\n{'='*40}\n"
        return header + content
    
    def _process_xml(self, content: str) -> str:
        """Process XML content"""
        return f"XML Document Content:\n{'='*40}\n{content}"
    
    def _process_markdown(self, content: str) -> str:
        """Process Markdown content"""
        return f"Markdown Document:\n{'='*40}\n{content}"
    
    def _process_plain_text(self, content: str) -> str:
        """Process plain text content"""
        lines = content.split('\n')
        if len(lines) > 5:
            header = f"Text Document - {len(lines)} lines\n{'='*40}\n"
            return header + content
        return content
    
    def _apply_query_filter(self, content: str, query: str) -> str:
        """Apply query-based filtering to content"""
        if not query.strip():
            return content
        
        lines = content.split('\n')
        query_terms = [term.strip().lower() for term in query.split(',') if term.strip()]
        
        if not query_terms:
            return content
        
        # Find lines containing query terms
        relevant_lines = []
        for line_num, line in enumerate(lines):
            line_lower = line.lower()
            if any(term in line_lower for term in query_terms):
                # Include some context around matching lines
                start = max(0, line_num - 1)
                end = min(len(lines), line_num + 2)
                for i in range(start, end):
                    if i not in [r[0] for r in relevant_lines]:  # Avoid duplicates
                        relevant_lines.append((i, lines[i]))
        
        if relevant_lines:
            # Sort by line number and format
            relevant_lines.sort(key=lambda x: x[0])
            filtered_content = '\n'.join([line[1] for line in relevant_lines])
            
            filter_info = f"Query Filter Applied: '{query}'\n"
            filter_info += f"Found {len(relevant_lines)} relevant lines from {len(lines)} total\n"
            filter_info += f"{'='*50}\n"
            
            print(f"🎯 Query filter applied: {len(relevant_lines)}/{len(lines)} lines matched")
            return filter_info + filtered_content
        else:
            print(f"⚠️ No lines matched query '{query}', returning full content")
            return f"No content matched query '{query}'. Full content:\n{'='*50}\n{content}"