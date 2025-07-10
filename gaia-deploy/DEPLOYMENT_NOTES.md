# GAIA Agent Deployment Notes

## Required Environment Variables
Set these in HF Spaces settings for full functionality:

### Essential (for basic operation)
- `GROQ_API_KEY` - Primary model provider (recommended)
- `GOOGLE_API_KEY` - Google Gemini access
- `HF_TOKEN` - HuggingFace access token

### Optional (for enhanced capabilities)  
- `SERPER_API_KEY` - Enhanced web search (langchain_tools.py)
- `OPENAI_API_KEY` - OpenAI model access
- `ANTHROPIC_API_KEY` - Claude model access

## Tool Dependencies Verification

Your agent system uses these specialized tools:

### 1. ContentRetrieverTool (content_retriever_tool.py)
**Dependencies:** docling, pymupdf, sentence-transformers, pillow, requests
**Purpose:** Advanced document processing (PDF, DOCX, XLSX, etc.)
**Fallback:** Basic text processing if docling unavailable

### 2. VisionWebBrowserTool (vision_browser_tool.py)  
**Dependencies:** selenium, helium, webdriver-manager, pillow
**Purpose:** Browser automation with screenshots
**Note:** May need Chrome/Chromium in HF Spaces

### 3. YouTubeContentTool (youtube_content_tool.py)
**Dependencies:** yt-dlp, requests
**Purpose:** YouTube transcript and metadata extraction
**Fallback:** Basic metadata if transcript unavailable

### 4. LangChain Tools (langchain_tools.py)
**Dependencies:** langchain-community, arxiv, pymupdf, wikipedia
**Purpose:** Research tools (web search, Wikipedia, ArXiv)
**Fallback:** Graceful degradation if APIs unavailable

## Agent System Configuration

### CodeAgent Additional Imports (from agent_logic.py)
Your data_analyst agent uses these imports:
```python
additional_authorized_imports = [
    "pandas", "numpy", "openpyxl", "xlrd", "csv",
    "scipy", "matplotlib", "seaborn", 
    "sklearn", "scikit-learn", "statistics", "math",
    "pathlib", "mimetypes", "re", "json", "os"
]
```

### Browser Tool Compatibility
VisionWebBrowserTool may face restrictions in HF Spaces:
- Chrome/Chromium availability
- Selenium WebDriver setup  
- File download permissions

**Recommendation:** Test browser functionality and implement graceful fallback.

## File Processing Capabilities

Your system supports all 17 GAIA file types:
- **Spreadsheets:** .xlsx, .xls, .csv, .tsv
- **Documents:** .pdf, .docx, .doc, .txt  
- **Images:** .png, .jpg, .jpeg, .gif, .bmp
- **Audio:** .mp3, .wav, .m4a, .mov
- **Data:** .json, .xml, .py, .pdb
- **Archives:** .zip

## Testing Checklist

Before final submission:
1. ✅ Test file processing with sample attachments
2. ✅ Verify web search functionality  
3. ✅ Check YouTube content extraction
4. ✅ Test browser automation (if available)
5. ✅ Validate model provider fallbacks

## Performance Optimization

Your smart routing system:
- Simple questions → one_shot_llm (faster, cheaper)
- Complex questions → manager_coordination (full agent system)
- File attachments → automatic specialist routing

Expected performance: 50-60% GAIA accuracy
