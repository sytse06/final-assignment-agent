# 📦 GAIA Agent Deployment - Keep Existing System

## Simple Integration: Use Your Current System

### Phase 1: Copy Existing Files (15 minutes)

#### Step 1: Gather Your Files
```bash
# Create deployment directory
mkdir gaia-deploy
cd gaia-deploy

# Copy your existing agent files
cp /path/to/examples/agent_logic.py .
cp /path/to/examples/agent_interface.py .
cp /path/to/examples/agent_logging.py .
cp /path/to/examples/agent_testing.py .
cp /path/to/examples/dev_retriever.py .
cp /path/to/examples/gaia_embeddings.csv .
cp /path/to/examples/metadata.json* .
cp /path/to/examples/gaia_dataset_utils.py .
cp /path/to/examples/content_retriever_tool.py .
cp /path/to/examples/youtube_content_tool.py .
cp /path/to/examples/*_tool.py . 2>/dev/null || true
```

#### Step 2: Add Course Interface
```python
# app.py - Simple adapter for course submission
# Use the minimal adapter from previous artifact
```

#### Step 3: Dependencies
```txt
# requirements.txt - From your pyproject.toml
gradio>=5.13.2
requests>=2.32.3
pandas>=2.0.0
python-dotenv>=1.0.0

transformers>=4.40.0
huggingface-hub>=0.33.0
torch>=2.1.0

smolagents[transformers]>=1.19.0

langchain-community>=0.3.24
langchain-huggingface>=0.2.0
langgraph>=0.4.8
langchain-groq>=0.3.2
langchain-google-genai>=2.1.5

groq>=0.4.0
litellm>=1.72.1

beautifulsoup4>=4.12.0
pillow>=11.0.0
openpyxl>=3.1.0
pypdf2>=3.0.0
python-docx>=1.1.0

numpy>=2.0.0,<3.0.0
scipy>=1.11.0

docling>=2.36.1
pymupdf>=1.26.1

duckduckgo-search>=8.0.2
lxml>=5.3.0

sentence-transformers>=4.1.0
weaviate-client>=4.14.4
langchain-weaviate>=0.0.5

backoff>=2.2.1
```

#### Step 4: Simple README
```markdown
---
title: GAIA Agent
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
---

# GAIA Agent

Multi-agent system for GAIA benchmark.

Features multi-agent coordination and file processing capabilities.
```

### Phase 2: Deploy (10 minutes)

#### Step 5: Upload to HF Spaces
```bash
# Clone your HF Space
git clone https://huggingface.co/spaces/your-username/your-space-name
cd your-space-name

# Copy files
cp ../gaia-deploy/* .

# Deploy
git add .
git commit -m "Deploy GAIA agent"
git push origin main
```

#### Step 6: Environment Variables
Add to HF Spaces settings if needed:
- `GROQ_API_KEY`
- `GOOGLE_API_KEY` 
- `OPENAI_API_KEY`

### Phase 3: Test (5 minutes)

#### Step 7: Verify
1. Check build completes without errors
2. Test with simple question
3. Login and run course evaluation

## File Structure

```
your-space/
├── agent_logic.py              # Your existing system
├── agent_interface.py          # Your configuration  
├── agent_logging.py            # Your logging
├── agent_testing.py            # Your testing
├── dev_retriever.py            # Your retriever
├── gaia_embeddings.csv         # Your embeddings
├── metadata.json               # Your dataset
├── gaia_dataset_utils.py       # Your utilities
├── content_retriever_tool.py   # Your tools
├── youtube_content_tool.py     # Your tools
├── *_tool.py                   # Other tools
├── app.py                      # Course interface adapter
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Integration Approach

### Simple Adapter Pattern
- Keeps your existing code unchanged
- Translates between course format and your system
- Provides fallback if imports fail

### Benefits
- No modifications to existing work
- Uses your current capabilities
- Meets course requirements
- Quick to deploy

## If Issues Arise

### Build Problems
- Remove heavy dependencies from requirements.txt
- Check file sizes are under limits
- Verify all imports work

### Runtime Issues
- Check API keys are set
- Verify your system initializes correctly
- Check logs for import errors

### Submission Problems
- Ensure HF login works
- Check network connectivity
- Retry if needed

## Expected Timeline: 30 minutes

This approach uses your existing system with a simple interface layer for course compatibility.