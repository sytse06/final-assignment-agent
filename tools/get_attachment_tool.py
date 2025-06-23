# tools/get_attachment_tool.py
from smolagents import Tool
import requests
from urllib.parse import urljoin
import base64
import tempfile
import os
from typing import Optional
from pathlib import Path


class GetAttachmentTool(Tool):
    """
    GAIA attachment tool that uses metadata.json for file type detection.
    Much more reliable than file signature detection.
    """
    
    name = "get_attachment"
    description = """Retrieves the attachment file for the current GAIA task with smart content processing.

Format options:
- 'URL': Returns download URL
- 'LOCAL_FILE_PATH': Downloads and returns local file path  
- 'CONTENT': Downloads and returns processed content (recommended for data analysis)
- 'DATA_URL': Returns base64 data URL
- 'TEXT': Returns raw text (text files only)

For data analysis tasks, use 'CONTENT' format to get structured data directly.
Excel/CSV files are processed as tabular data, documents use semantic retrieval.

Returns empty string if no file is attached to the current task."""

    inputs = {
        "fmt": {
            "type": "string", 
            "description": "Format to retrieve attachment. Options: 'URL', 'DATA_URL', 'LOCAL_FILE_PATH', 'TEXT', 'CONTENT'.",
            "nullable": True,
            "default": "CONTENT"
        }
    }
    output_type = "string"

    def __init__(
        self, 
        agent_evaluation_api: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs
    ):
        self.agent_evaluation_api = (
            agent_evaluation_api or 
            "https://agents-course-unit4-scoring.hf.space/"
        )
        self.task_id = task_id
        self._metadata_cache = {}  # Cache for metadata lookups
        super().__init__(**kwargs)

    def attachment_for(self, task_id: Optional[str]):
        """Set the task ID for file retrieval"""
        self.task_id = task_id
        if task_id:
            print(f"🔗 GetAttachmentTool configured for task: {task_id}")

    def forward(self, fmt: str = "CONTENT") -> str:
        """
        Retrieve attachment using metadata.json for file type detection.
        """
        # Handle task_id passed as format (agent error correction)
        if fmt and len(fmt) > 10 and '-' in fmt:
            print(f"⚠️  Detected task_id passed as format: {fmt}")
            print(f"🔧 Auto-correcting: setting task_id and using CONTENT format")
            self.task_id = fmt
            fmt = "CONTENT"
        
        fmt = fmt.upper().strip()
        
        # Validate format
        valid_formats = ["URL", "DATA_URL", "LOCAL_FILE_PATH", "TEXT", "CONTENT"]
        if fmt not in valid_formats:
            return f"Error: Invalid format '{fmt}'. Use: {', '.join(valid_formats)}"

        if not self.task_id:
            print("⚠️  No task_id set - cannot retrieve attachment")
            return ""

        # Get file info from metadata
        file_info = self._get_file_info_from_metadata()
        
        if not file_info:
            return "Error: No file attachment found for this task"
        
        # Try to find the actual file
        local_file_path = self._find_file_with_metadata(file_info)
        
        if local_file_path:
            print(f"✅ Found file: {local_file_path}")
            return self._process_local_file(local_file_path, fmt, file_info)
        
        # Fallback: Try API
        print(f"📥 File not found locally, trying API...")
        return self._try_api_fallback(fmt)

    def _get_file_info_from_metadata(self) -> Optional[dict]:
        """Get file information from metadata.json"""
        
        if self.task_id in self._metadata_cache:
            return self._metadata_cache[self.task_id]
        
        # Try to load metadata.json
        metadata_paths = [
            "./tests/gaia_data/metadata.json",
            "../tests/gaia_data/metadata.json",
            "./gaia_data/metadata.json",
            "./metadata.json"
        ]
        
        for metadata_path in metadata_paths:
            if os.path.exists(metadata_path):
                try:
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Handle different metadata structures
                    questions = []
                    if isinstance(metadata, dict) and 'validation' in metadata:
                        questions = metadata['validation']
                    elif isinstance(metadata, list):
                        questions = metadata
                    
                    # Find our task
                    for question in questions:
                        if question.get('task_id') == self.task_id:
                            file_info = {
                                'file_name': question.get('file_name'),
                                'file_path': question.get('file_path'),
                                'task_id': self.task_id
                            }
                            self._metadata_cache[self.task_id] = file_info
                            print(f"📋 Found metadata: {file_info['file_name']}")
                            return file_info
                    
                    print(f"❌ Task {self.task_id} not found in metadata")
                    return None
                    
                except Exception as e:
                    print(f"❌ Error reading metadata: {e}")
                    continue
        
        print(f"❌ No metadata.json found")
        return None

    def _find_file_with_metadata(self, file_info: dict) -> Optional[str]:
        """Find file using metadata information"""
        
        # Strategy 1: Use file_path from metadata if it exists
        file_path = file_info.get('file_path')
        if file_path and os.path.exists(file_path):
            print(f"✅ Using metadata file_path: {file_path}")
            return file_path
        
        # Strategy 2: Search HF cache for files matching task_id
        cache_bases = [
            os.path.expanduser("~/.cache/huggingface/datasets/downloads"),
            os.path.expanduser("~/.cache/huggingface/hub/datasets"),
            "/tmp/huggingface_cache/datasets",
        ]
        
        print(f"🔍 Searching HF cache for task_id: {self.task_id}")
        
        for cache_base in cache_bases:
            cache_path = Path(cache_base)
            if not cache_path.exists():
                continue
                
            try:
                # Look for files containing the task_id
                for file_path in cache_path.rglob("*"):
                    if not file_path.is_file():
                        continue
                    
                    # Check if filename contains task_id (prioritize non-JSON)
                    if self.task_id in file_path.name and not file_path.name.endswith('.json'):
                        print(f"✅ Found file by task_id: {file_path}")
                        return str(file_path)
                
                # Also check JSON metadata files and look for corresponding data files
                for file_path in cache_path.rglob("*.json"):
                    if self.task_id in file_path.name:
                        data_file = file_path.with_suffix('')
                        if data_file.exists():
                            print(f"✅ Found data file via JSON: {data_file}")
                            return str(data_file)
                        
            except (PermissionError, OSError):
                continue
        
        print(f"❌ Could not find file for task: {self.task_id}")
        return None

    def _process_local_file(self, file_path: str, fmt: str, file_info: dict) -> str:
        """Process local file using metadata for type detection"""
        
        if fmt == "LOCAL_FILE_PATH":
            return file_path
        
        elif fmt == "TEXT":
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return content
            except UnicodeDecodeError:
                return f"Error: File cannot be read as text. Binary file at: {file_path}"
        
        elif fmt == "DATA_URL":
            try:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                mime_type = self._get_mime_type_from_metadata(file_info)
                encoded_content = base64.b64encode(file_content).decode('utf-8')
                return f"data:{mime_type};base64,{encoded_content}"
            except Exception as e:
                return f"Error creating data URL: {str(e)}"
        
        elif fmt == "CONTENT":
            return self._smart_content_processing(file_path, file_info)
        
        else:
            return f"Error: Unsupported format: {fmt}"

    def _smart_content_processing(self, file_path: str, file_info: dict) -> str:
        """Smart content processing using metadata for file type"""
        
        try:
            # Get file extension from metadata
            file_name = file_info.get('file_name', '')
            if not file_name:
                return f"Error: No file_name in metadata for {file_path}"
            
            # Extract extension from metadata file_name
            file_extension = Path(file_name).suffix.lower()
            
            print(f"📄 Processing as {file_extension} based on metadata: {file_name}")
            
            # Process based on metadata extension
            if file_extension == '.csv':
                return self._process_csv(file_path)
            
            elif file_extension in ['.xlsx', '.xls']:
                return self._process_excel(file_path)
            
            elif file_extension == '.txt':
                return self._process_text(file_path)
            
            elif file_extension == '.json':
                return self._process_json(file_path)
            
            elif file_extension in ['.pdf', '.docx', '.pptx']:
                return self._process_document(file_path)
            
            elif file_extension in ['.png', '.jpg', '.jpeg', '.gif']:
                return self._process_image(file_path)
            
            elif file_extension == '.pdb':
                return self._process_pdb(file_path)
            
            else:
                return f"Unknown file type from metadata: {file_extension}\nFile: {file_path}\nTreat as binary file."
                
        except Exception as e:
            return f"Error processing file content: {str(e)}\nFile path: {file_path}"

    def _get_mime_type_from_metadata(self, file_info: dict) -> str:
        """Get MIME type from metadata file extension"""
        file_name = file_info.get('file_name', '')
        extension = Path(file_name).suffix.lower()
        
        ext_to_mime = {
            ".pdf": "application/pdf",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".pdb": "chemical/x-pdb",
        }
        
        return ext_to_mime.get(extension, "application/octet-stream")

    def _process_csv(self, file_path: str) -> str:
        """Process CSV files with pandas"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            info = f"CSV Data Summary:\n"
            info += f"- Rows: {len(df)}\n"
            info += f"- Columns: {len(df.columns)}\n"
            info += f"- Column names: {list(df.columns)}\n\n"
            info += f"Data Types:\n{df.dtypes}\n\n"
            info += f"First 10 rows:\n{df.head(10).to_string()}\n"
            
            if len(df) > 15:
                info += f"\nLast 5 rows:\n{df.tail(5).to_string()}\n"
            
            return info
            
        except Exception as e:
            return f"Error processing CSV: {str(e)}\nFile path: {file_path}"

    def _process_excel(self, file_path: str) -> str:
        """Process Excel files with pandas"""
        try:
            import pandas as pd
            
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            info = f"Excel File Summary:\n"
            info += f"- Sheets: {sheet_names}\n\n"
            
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                info += f"Sheet '{sheet_name}':\n"
                info += f"- Rows: {len(df)}\n"
                info += f"- Columns: {len(df.columns)}\n"
                info += f"- Column names: {list(df.columns)}\n\n"
                info += f"Data Types:\n{df.dtypes}\n\n"
                info += f"First 10 rows:\n{df.head(10).to_string()}\n\n"
                
                # Show unique values in key columns for analysis
                for col in df.columns:
                    if col.lower() in ['format', 'platform', 'type', 'category']:
                        unique_vals = df[col].dropna().unique()
                        if len(unique_vals) <= 20:  # Only show if not too many
                            info += f"Unique values in '{col}': {list(unique_vals)}\n"
                
                info += "\n" + "="*50 + "\n\n"
            
            return info
            
        except Exception as e:
            return f"Error processing Excel: {str(e)}\nFile path: {file_path}"

    def _process_text(self, file_path: str) -> str:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            info = f"Text File Content:\n"
            info += f"- Length: {len(content)} characters\n"
            info += f"- Lines: {len(content.splitlines())}\n\n"
            info += f"Content:\n{content}"
            
            return info
            
        except Exception as e:
            return f"Error processing text file: {str(e)}\nFile path: {file_path}"

    def _process_json(self, file_path: str) -> str:
        """Process JSON files"""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            info = f"JSON File Content:\n"
            info += f"- Type: {type(data).__name__}\n"
            
            if isinstance(data, dict):
                info += f"- Keys: {list(data.keys())}\n"
            elif isinstance(data, list):
                info += f"- Items: {len(data)}\n"
            
            info += f"\nContent:\n{json.dumps(data, indent=2)[:2000]}..."
            
            return info
            
        except Exception as e:
            return f"Error processing JSON: {str(e)}\nFile path: {file_path}"

    def _process_pdb(self, file_path: str) -> str:
        """Process PDB files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            atom_lines = [line for line in lines if line.startswith('ATOM')]
            
            info = f"PDB File Content:\n"
            info += f"- Total lines: {len(lines)}\n"
            info += f"- Atom records: {len(atom_lines)}\n"
            info += f"- File path: {file_path}\n\n"
            
            if atom_lines:
                info += f"First 5 atom records:\n"
                for line in atom_lines[:5]:
                    info += f"{line}\n"
            
            info += f"\nFull content:\n{content[:2000]}..."
            
            return info
            
        except Exception as e:
            return f"Error processing PDB: {str(e)}\nFile path: {file_path}"

    def _process_document(self, file_path: str) -> str:
        """Process documents using ContentRetrieverTool"""
        try:
            from content_retriever_tool import ContentRetrieverTool
            retriever = ContentRetrieverTool()
            content = retriever.forward(url=file_path, query="all content data information text")
            return f"Document Content (processed by ContentRetrieverTool):\n{content}"
        except ImportError:
            return f"Document file at: {file_path}\n(ContentRetrieverTool not available)"
        except Exception as e:
            return f"Error processing document: {str(e)}\nFile path: {file_path}"

    def _process_image(self, file_path: str) -> str:
        """Process image files"""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
            
            info = f"Image File:\n"
            info += f"- Dimensions: {width}x{height}\n"
            info += f"- Mode: {mode}\n"
            info += f"- Format: {format_name}\n"
            info += f"- File path: {file_path}\n\n"
            info += "Note: Use vision-capable models to analyze image content."
            
            return info
            
        except ImportError:
            return f"Image at: {file_path}\n(PIL not available)"
        except Exception as e:
            return f"Error processing image: {str(e)}\nFile path: {file_path}"

    def _try_api_fallback(self, fmt: str) -> str:
        """Fallback API method"""
        try:
            file_url = urljoin(self.agent_evaluation_api, f"files/{self.task_id}")
            
            if fmt == "URL":
                return file_url

            response = requests.get(file_url, timeout=30)
            
            if 400 <= response.status_code < 500:
                return "Error: No file attachment found and not available locally"
            
            response.raise_for_status()
            
            # Save file and process
            mime_type = response.headers.get("content-type", "application/octet-stream")
            file_extension = self._get_file_extension(mime_type)
            
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=file_extension,
                prefix=f"gaia_task_{self.task_id}_"
            ) as tmp_file:
                tmp_file.write(response.content)
                file_path = tmp_file.name
            
            # Create fake file_info for processing
            fake_file_info = {'file_name': f"{self.task_id}{file_extension}"}
            return self._process_local_file(file_path, fmt, fake_file_info)
            
        except Exception as e:
            return f"Error: Could not retrieve file - {str(e)}"

    def _get_file_extension(self, mime_type: str) -> str:
        """Get file extension from MIME type"""
        mime_to_ext = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "text/csv": ".csv",
            "text/plain": ".txt",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "application/json": ".json",
        }
        return mime_to_ext.get(mime_type, ".bin")

    def __repr__(self):
        return f"GetAttachmentTool(task_id='{self.task_id}')"