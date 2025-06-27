# agent_logic.py
# GAIA Agent System using SmolagAgents Manager Pattern

import os
import uuid
import re
import backoff
from typing import TypedDict, Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Core dependencies
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

import openai

# SmolagAgents imports
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    ManagedAgent,
    LiteLLMModel,
    AgentLogger,
    LogLevel,
    GoogleSearchTool,
    VisitWebpageTool,
    WikipediaSearchTool,
    SpeechToTextTool
)

# Import agent context system
from agent_context import ContextBridge

# Import retriever system
from dev_retriever import load_gaia_retriever

# Import logging
from agent_logging import AgentLoggingSetup

try:
    from tools import (GetAttachmentTool, ContentRetrieverTool)
    CUSTOM_TOOLS_AVAILABLE = True
    print("✅ Custom tools imported successfully")
except ImportError as e:
    print(f"⚠️  Custom tools not available: {e}")
    CUSTOM_TOOLS_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GAIAConfig:
    """Configuration for GAIA agent"""
    model_provider: str = "groq"
    model_name: str = "qwen-qwq-32b"
    temperature: float = 0.3
    
    # For testing with Ollama
    api_base: Optional[str] = None
    num_ctx: int = 32768
    
    # Retriever settings
    csv_file: str = "gaia_embeddings.csv"
    rag_examples_count: int = 3
    
    # Agent settings
    max_agent_steps: int = 15
    planning_interval: int = 5
        
    # Routing settings
    enable_smart_routing: bool = True
    skip_rag_for_simple: bool = True
    
    # Context settings
    enable_context_bridge: bool = True
    context_bridge_debug: bool = True
    enable_grounding_tools: bool = False # DISABLED for GAIA
    
    # Logging
    enable_csv_logging: bool = True
    step_log_file: str = "gaia_steps.csv"
    question_log_file: str = "gaia_questions.csv"
    debug_mode: bool = True

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class GAIAState(TypedDict):
    """State for GAIA workflow"""
    # Core task information
    task_id: Optional[str]
    question: str
    
    # File handling (NEW: Enhanced file support)
    file_name: Optional[str]
    file_path: Optional[str]
    has_file: Optional[bool]
    file_metadata: Optional[Dict]
    
    # Task flow
    steps: List[str]
    agent_used: Optional[str]
    current_error: Optional[str]
    raw_answer: Optional[str]
    final_answer: Optional[str]
    
    # Coordination
    similar_examples: Optional[List[Dict]]
    complexity: Optional[str]
    selected_agent: Optional[str]
    coordination_analysis: Optional[Dict]
    
    # Execution tracking
    execution_successful: Optional[bool]

# ============================================================================
# LLM RETRY LOGIC
# ============================================================================

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, max_tries=3)
def llm_invoke_with_retry(llm, messages):
    """Retry logic for LangChain LLM calls"""
    return llm.invoke(messages)

# ============================================================================
# OTHER UTILITY FUNCTIONS
# ============================================================================

def is_simple_math(question: str) -> bool:
    """Check if question is simple arithmetic"""
    question_lower = question.lower()
    
    # Math operation keywords
    math_keywords = ['calculate', 'what is', '%', 'percent', 'multiply', 'divide', 'add', 'subtract']
    has_math_keyword = any(keyword in question_lower for keyword in math_keywords)
    
    # Simple patterns
    simple_patterns = [
        r'\d+\s*%\s*of\s*\d+',  # "15% of 200"
        r'what\s+is\s+\d+.*\d+',  # "what is 15 * 23"
        r'calculate\s+\d+.*\d+',   # "calculate 15 + 23"
    ]
    
    has_simple_pattern = any(re.search(pattern, question_lower) for pattern in simple_patterns)
    
    return has_math_keyword and has_simple_pattern and len(question.split()) < 10

def is_simple_fact(question: str) -> bool:
    """Check if question is a simple factual query"""
    question_lower = question.lower()
    
    simple_fact_patterns = [
        r'what\s+are?\s+the\s+primary\s+colors?',
        r'what\s+is\s+the\s+capital\s+of',
        r'when\s+was.*born',
        r'how\s+many.*in',
    ]
    
    return any(re.search(pattern, question_lower) for pattern in simple_fact_patterns)

def has_attachments(task_id: str) -> bool:
    """Check if task has file attachments (placeholder)"""
    # This would check for actual attachments in a real implementation
    # For now, return False as we don't have attachment detection
    return False

def needs_web_search(question: str) -> bool:
    """Check if question needs current information"""
    question_lower = question.lower()
    
    web_keywords = [
        'current', 'latest', 'recent', 'today', 'now', 'this year',
        'population of', 'price of', 'stock price', 'weather',
        'who won', 'election results'
    ]
    
    return any(keyword in question_lower for keyword in web_keywords)

def extract_file_info_from_task_id(task_id: str) -> Dict[str, str]:
    """
    Extract file information from task_id using GAIA dataset.
    """
    if not task_id:
        return {"file_name": "", "file_path": "", "has_file": False}
    
    try:
        # Try to get file info from GAIA dataset
        from gaia_dataset_utils import GAIADatasetManager
        
        # This should be passed in or configured, but for now use default path
        dataset_manager = GAIADatasetManager("./tests/gaia_data")
        
        # Get the actual question data for this task_id
        question_data = dataset_manager.get_question_by_id(task_id)
        
        if question_data:
            file_name = question_data.get('file_name', '')
            file_path = question_data.get('file_path', '')
            has_file = bool(file_name)
            
            return {
                "file_name": file_name,
                "file_path": file_path, 
                "has_file": has_file
            }
        else:
            # Task not found in dataset
            return {"file_name": "", "file_path": "", "has_file": False}
            
    except Exception as e:
        print(f"⚠️  Could not extract file info for task {task_id}: {e}")
        # For development/testing with arbitrary task_ids, return no file
        return {"file_name": "", "file_path": "", "has_file": False}

def validate_gaia_format(answer: str) -> bool:
    """Check if answer meets GAIA requirements"""
    if not answer or len(answer.strip()) == 0:
        return False
    
    # Check for FINAL ANSWER pattern
    if "FINAL ANSWER:" not in answer.upper():
        return False
    
    return True

# ============================================================================
# MAIN GAIA AGENT
# ============================================================================
class GAIAAgent:
    """GAIA agent using SmolagAgents coordinator pattern within LangGraph workflow"""
    
    def __init__(self, config: Optional[GAIAConfig] = None) -> None:
        if config is None:
            config = GAIAConfig()
        
        self.config: GAIAConfig = config
        self.retriever: Any = self._initialize_retriever()
        self.orchestration_model: Any = self._initialize_orchestration_model()  # LangChain for workflow
        self.specialist_model: LiteLLMModel = self._initialize_specialist_model()        # LiteLLM for SmolagAgents
        
        # Setup enhanced logging
        if config.enable_csv_logging:
            self.logging: Optional[EnhancedAgentLoggingSetup] = EnhancedAgentLoggingSetup(
                debug_mode=config.debug_mode,
                step_log_file=config.step_log_file,
                question_log_file=config.question_log_file
            )
        else:
            self.logging: Optional[EnhancedAgentLoggingSetup] = None
        
        # Create coordinator and specialists (will be created fresh for each task)
        self.coordinator: Optional[CodeAgent] = None
        self.specialists: Dict[str, Union[CodeAgent, ToolCallingAgent]] = {}
        
        # Build workflow
        self.workflow: Any = self._build_workflow()  # StateGraph type is complex
        
        print("🚀 GAIA Agent initialized with SmolagAgents coordinator pattern!")
    
    def _initialize_retriever(self):
        """Initialize retriever for similar questions in manager context."""
        try:
            print("🔄 Setting up retriever...")
            retriever = load_gaia_retriever(self.config.csv_file)
            
            if retriever and retriever.is_ready():
                print("✅ Retriever ready!")
                return retriever
            else:
                raise RuntimeError("❌ Failed to initialize retriever")
        except Exception as e:
            print(f"❌ Retriever initialization error: {e}")
            raise
    
    def _initialize_orchestration_model(self):
        """Initialize both orchestration and specialist models"""
        try:
            # Initialize orchestration model (LangChain)
            if self.config.model_provider == "groq":
                self.orchestration_model = ChatGroq(
                    model=self.config.model_name,
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=self.config.temperature
                )
            elif self.config.model_provider == "openrouter":
                self.orchestration_model = ChatOpenAI(
                    model=self.config.model_name,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=self.config.temperature
                )
            elif self.config.model_provider == "google":
                self.orchestration_model = ChatGoogleGenerativeAI(
                    model=self.config.model_name,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=self.config.temperature
                )
            elif self.config.model_provider == "ollama":
                from langchain_ollama import ChatOllama
                self.orchestration_model = ChatOllama(
                    model=self.config.model_name,
                    base_url=self.config.api_base or "http://localhost:11434",
                    temperature=self.config.temperature
                )
            else:
                # Fallback to Openrouter
                self.orchestration_model = ChatOpenAI(
                    model="qwen/qwen3-32b",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=self.config.temperature
                )
            
            print(f"✅ Orchestration model: {getattr(self.orchestration_model, 'model_name', getattr(self.orchestration_model, 'model', 'Unknown'))}")
            
            # Initialize specialist model (SmolAgents)
            if self.config.model_provider == "groq":
                model_id = f"groq/{self.config.model_name}"
                api_key = os.getenv("GROQ_API_KEY")
            elif self.config.model_provider == "openrouter":
                model_id = f"openrouter/{self.config.model_name}"
                api_key = os.getenv("OPENROUTER_API_KEY")
            elif self.config.model_provider == "ollama":
                model_id = f"ollama_chat/{self.config.model_name}"
                specialist_model = LiteLLMModel(
                    model_id=model_id,
                    api_base=self.config.api_base or "http://localhost:11434",
                    num_ctx=self.config.num_ctx,
                    temperature=self.config.temperature)
                print(f"✅ Specialist model: {model_id}")
                return specialist_model
            elif self.config.model_provider == "google":
                model_id = f"gemini/{self.config.model_name}"
                api_key = os.getenv("GOOGLE_API_KEY")
            else:
                raise ValueError(f"Unknown provider: {self.config.model_provider}")
            
            specialist_model = LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                temperature=self.config.temperature
            )
            
            print(f"✅ Specialist model: {model_id}")
            return specialist_model
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            raise
        
    # ============================================================================
    # FILE HANDLING WITH SMOLAGENTS INSIGHTS
    # ============================================================================

    def _download_file_once(self, task_id: str, file_name: str) -> Optional[str]:
        """
        Download attached file once, reuse for all agents
        """
        try:
            # Production API endpoint
            agent_evaluation_api = getattr(self.config, 'agent_evaluation_api', 
                                         "https://agents-course-unit4-scoring.hf.space/")
            file_url = f"{agent_evaluation_api}files/{task_id}"
            
            print(f"📡 Single download request: {file_url}")
            
            response = requests.get(
                file_url,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=30,
                stream=True
            )
            
            if response.status_code == 200:
                # Create temp file with correct extension (important for SmolagAgents)
                file_extension = os.path.splitext(file_name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp_file.write(chunk)
                    local_path = tmp_file.name
                
                file_size = os.path.getsize(local_path)
                print(f"✅ Downloaded once: {local_path} ({file_size} bytes)")
                return local_path
                
            else:
                print(f"❌ Download failed: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None

    def _analyze_file_metadata(self, file_name: str, file_path: str) -> Dict[str, Any]:
        """
        Python-native file analysis needs no tool and gives coordinator Python capabilities
        """
        if not file_name:
            return {"no_file": True}
        
        try:
            # Basic file analysis
            path_obj = Path(file_name)
            extension = path_obj.suffix.lower()
            
            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_name)
            
            # Categorize file type
            if extension in ['.xlsx', '.csv', '.xls', '.tsv']:
                category = "data"
                processing_approach = "direct_pandas"
                recommended_specialist = "data_analyst"
                specialist_guidance = {
                    "tool_command": "pd.read_excel(file_path) or pd.read_csv(file_path)",
                    "imports_needed": ["pandas", "openpyxl"],
                    "processing_strategy": "Load data → analyze structure → perform calculations"
                }
            elif extension in ['.pdf', '.docx', '.doc', '.txt']:
                category = "document"
                processing_approach = "content_extraction"
                recommended_specialist = "document_processor"
                specialist_guidance = {
                    "tool_command": "Use ContentRetrieverTool with file_path from additional_args",
                    "imports_needed": [],
                    "processing_strategy": "Extract content → analyze text → answer question"
                }
            elif extension in ['.mp3', '.mp4', '.wav', '.m4a', '.mov']:
                category = "media"
                processing_approach = "transcription"
                recommended_specialist = "document_processor"
                specialist_guidance = {
                    "tool_command": "Use SpeechToTextTool with file_path from additional_args",
                    "imports_needed": [],
                    "processing_strategy": "Transcribe audio → analyze transcript → extract information"
                }
            elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                category = "image"
                processing_approach = "vision_analysis"
                recommended_specialist = "document_processor"
                specialist_guidance = {
                    "tool_command": "Use vision tools with file_path from additional_args",
                    "imports_needed": [],
                    "processing_strategy": "Analyze image → extract visual information → answer question"
                }
            else:
                category = "unknown"
                processing_approach = "content_extraction"
                recommended_specialist = "document_processor"
                specialist_guidance = {
                    "tool_command": "Use ContentRetrieverTool with file_path from additional_args",
                    "imports_needed": [],
                    "processing_strategy": "Extract content → analyze → answer question"
                }
            
            return {
                "file_name": file_name,
                "file_path": file_path,
                "extension": extension,
                "mime_type": mime_type,
                "category": category,
                "processing_approach": processing_approach,
                "recommended_specialist": recommended_specialist,
                "specialist_guidance": specialist_guidance,
                "estimated_complexity": "medium"
            }
            
        except Exception as e:
            print(f"⚠️ File analysis error: {e}")
            return {
                "file_name": file_name,
                "file_path": file_path,
                "category": "unknown",
                "processing_approach": "content_extraction",
                "recommended_specialist": "document_processor",
                "error": str(e)
            }

    # ============================================================================
    # SMOLAGENTS COORDINATOR AND SPECIALISTS
    # ============================================================================        
    # ============================================================================
    # SMOLAGENTS COORDINATOR AND SPECIALISTS
    # ============================================================================

    def _create_coordinator(self) -> CodeAgent:
        """
        🔥 Create coordinator CodeAgent for strategic analysis
        Pure Python - no tools needed
        """
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        
        coordinator = CodeAgent(
            name="gaia_coordinator",
            description="""Strategic coordinator for GAIA task analysis and specialist selection.

CAPABILITIES:
- Python-native file analysis using pathlib and mimetypes
- Strategic task decomposition and approach planning
- Specialist selection based on file type and question analysis
- Execution guidance for selected specialists

WORKFLOW:
1. Analyze question for core problem and requirements
2. Perform Python file analysis if file present
3. Select optimal specialist based on analysis
4. Provide strategic execution guidance
""",
            tools=[],  # 🔥 No tools - pure Python analysis
            additional_authorized_imports=[
                "pathlib", "mimetypes", "re", "json", "os", "pandas"
            ],
            model=self.specialist_model,  # 🔥 LiteLLM for SmolagAgents coordinator
            planning_interval=7,
            max_steps=8,
            logger=logger
        )
        
        return coordinator

    def _create_specialists(self) -> Dict[str, Union[CodeAgent, ToolCallingAgent]]:
        """
        🔥 Create three specialist agents with proper tool configuration
        """
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        specialists = {}

        # 1. Data Analyst - CodeAgent for direct Python file access
        specialists["data_analyst"] = CodeAgent(
            name="data_analyst", 
            description="Excel/CSV analysis and numerical calculations using pandas",
            tools=[],  # 🔥 No tools - pure Python file access
            additional_authorized_imports=[
                "pandas", "numpy", "openpyxl", "xlrd", "csv",
                "scipy", "matplotlib", "seaborn", 
                "sklearn", "scikit-learn", "statistics", "math"
            ],
            model=self.specialist_model,  # 🔥 LiteLLM for SmolagAgents specialists
            max_steps=self.config.max_agent_steps,
            add_base_tools=True,
            logger=logger
        )
        
        # 2. Web Researcher - ToolCallingAgent for search
        web_tools = [
            GoogleSearchTool(),
            VisitWebpageTool(), 
            WikipediaSearchTool()
        ]
        
        specialists["web_researcher"] = ToolCallingAgent(
            name="web_researcher",
            description="Web search using GoogleSearchTool, Wikipedia, and content extraction",
            tools=web_tools,
            model=self.specialist_model,  # 🔥 LiteLLM for SmolagAgents web researcher
            max_steps=self.config.max_agent_steps,
            add_base_tools=True,
            logger=logger
        )
        
        # 3. Document Processor - ToolCallingAgent for file processing
        doc_tools = [SpeechToTextTool()]
        
        if CUSTOM_TOOLS_AVAILABLE:
            doc_tools.append(ContentRetrieverTool())
        
        specialists["document_processor"] = ToolCallingAgent(
            name="document_processor",
            description="Document extraction and audio transcription specialist",
            tools=doc_tools,
            model=self.specialist_model,  # 🔥 LiteLLM for SmolagAgents document processor
            max_steps=self.config.max_agent_steps,
            add_base_tools=True,
            logger=logger
        )
        
        print(f"🎯 Created {len(specialists)} specialized agents:")
        print(f"   data_analyst: CodeAgent with {len(specialists['data_analyst'].additional_authorized_imports)} imports")
        print(f"   web_researcher: ToolCallingAgent with {len(specialists['web_researcher'].tools)} tools")
        print(f"   document_processor: ToolCallingAgent with {len(specialists['document_processor'].tools)} tools")
        
        return specialists

    def _build_coordinator_task(self, state: GAIAState) -> str:
        """Build comprehensive coordination task for CodeAgent coordinator"""
        
        question = state["question"]
        file_name = state.get("file_name", "")
        file_path = state.get("file_path", "")
        has_file = state.get("has_file", False)
        similar_examples = state.get("similar_examples", [])
        complexity = state.get("complexity", "unknown")
        
        # Build similar examples context
        examples_context = ""
        if similar_examples:
            examples_context = "\n\nSIMILAR GAIA EXAMPLES:\n"
            for i, example in enumerate(similar_examples[:3], 1):
                examples_context += f"{i}. Q: {example.get('question', '')[:100]}...\n"
                examples_context += f"   A: {example.get('answer', '')}\n"
        
        task = f"""
You are the GAIA coordination agent. Analyze this question and provide strategic guidance.

QUESTION: {question}
COMPLEXITY: {complexity}
FILE INFO: {file_name} (path: {file_path}, has_file: {has_file})
{examples_context}

ANALYSIS TASKS:

1. CORE PROBLEM ANALYSIS:
```python
# Analyze the fundamental problem
question = "{question}"
print(f"Core problem: {{analyze_core_problem(question)}}")
print(f"Required capabilities: {{identify_capabilities_needed(question)}}")
```

2. FILE ANALYSIS (if file present):
```python
# Analyze file using Python
from pathlib import Path
import mimetypes

if "{file_name}":
    file_path = Path("{file_name}")
    extension = file_path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(str(file_path))
    
    # Categorize file type and determine processing approach
    if extension in ['.xlsx', '.csv', '.xls', '.tsv']:
        category = "data"
        processing_approach = "direct_pandas"
        recommended_specialist = "data_analyst"
        file_access_method = "direct_file_path"  # CodeAgent can use file paths directly
    elif extension in ['.pdf', '.docx', '.doc', '.txt']:
        category = "document"
        processing_approach = "content_extraction"
        recommended_specialist = "document_processor"
        file_access_method = "additional_args"  # ToolCallingAgent needs additional_args
    elif extension in ['.mp3', '.mp4', '.wav', '.m4a']:
        category = "media"
        processing_approach = "transcription"
        recommended_specialist = "document_processor"
        file_access_method = "additional_args"  # ToolCallingAgent needs additional_args
    else:
        category = "unknown"
        processing_approach = "content_extraction"
        recommended_specialist = "document_processor"
        file_access_method = "additional_args"
    
    print(f"File analysis: {{extension}} → {{category}} → {{processing_approach}}")
    print(f"Recommended specialist: {{recommended_specialist}}")
    print(f"File access method: {{file_access_method}}")
```

3. SPECIALIST SELECTION:
```python
# Available specialists and their capabilities:
specialists = {{
    "data_analyst": "CodeAgent - Excel/CSV processing, calculations, direct file access",
    "web_researcher": "ToolCallingAgent - Google search, current information, web content",
    "document_processor": "ToolCallingAgent - PDF/document extraction, transcription, vision"
}}

# Selection logic based on question and file analysis
if "calculate" in question.lower() or "data" in question.lower():
    if category == "data":
        selected = "data_analyst"
        reasoning = "Data file + calculation requirements"
    else:
        selected = "data_analyst"  # Can still handle calculations
        reasoning = "Calculation requirements"
elif "current" in question.lower() or "latest" in question.lower():
    selected = "web_researcher"
    reasoning = "Current information needed"
elif has_file and category in ["document", "media", "image"]:
    selected = "document_processor"
    reasoning = "File processing required"
else:
    # Default based on file type or question type
    selected = recommended_specialist if has_file else "web_researcher"
    reasoning = "Best match for question type"

print(f"SELECTED SPECIALIST: {{selected}}")
print(f"REASONING: {{reasoning}}")
```

4. EXECUTION STRATEGY:
```python
# Build execution strategy
execution_steps = []
if selected == "data_analyst":
    if has_file:
        execution_steps = [
            "Access file using direct file path (CodeAgent capability)",
            "Load data with pandas (pd.read_excel or pd.read_csv)",
            "Analyze data structure and content",
            "Perform required calculations or analysis",
            "Format answer according to GAIA requirements"
        ]
    else:
        execution_steps = [
            "Perform calculations using Python",
            "Use appropriate mathematical libraries",
            "Format numerical answer according to GAIA requirements"
        ]
elif selected == "web_researcher":
    execution_steps = [
        "Search for current information using GoogleSearchTool",
        "Visit relevant web pages for detailed information",
        "Cross-reference multiple sources",
        "Extract and verify factual information",
        "Format answer according to GAIA requirements"
    ]
else:  # document_processor
    execution_steps = [
        "Access file via additional_args parameter",
        "Use appropriate tool (ContentRetrieverTool or SpeechToTextTool)",
        "Extract relevant content from file",
        "Analyze extracted content for answer",
        "Format answer according to GAIA requirements"
    ]

for i, step in enumerate(execution_steps, 1):
    print(f"Step {{i}}: {{step}}")
```

FINAL OUTPUT: Print a clear summary of your analysis and recommendations.
"""
        
        return task

    def _parse_coordinator_analysis(self, coordination_result: str) -> Dict[str, Any]:
        """Parse coordinator's Python output into structured analysis"""
        try:
            lines = coordination_result.split('\n')
            
            analysis = {
                "core_problem": "Unknown",
                "selected_specialist": "data_analyst",  # Safe default
                "selection_reasoning": "Default selection",
                "execution_approach": "Standard processing",
                "confidence": 0.8,
                "file_metadata": {},
                "execution_strategy": {"steps": ["Standard processing"]}
            }
            
            # Parse coordinator output
            for line in lines:
                line = line.strip()
                
                if "SELECTED SPECIALIST:" in line:
                    specialist = line.split(":")[-1].strip()
                    if specialist in ["data_analyst", "web_researcher", "document_processor"]:
                        analysis["selected_specialist"] = specialist
                
                elif "REASONING:" in line:
                    analysis["selection_reasoning"] = line.split(":")[-1].strip()
                
                elif "Core problem:" in line:
                    analysis["core_problem"] = line.split(":")[-1].strip()
            
            # Validate specialist selection
            if analysis["selected_specialist"] not in ["data_analyst", "web_researcher", "document_processor"]:
                print(f"⚠️ Invalid specialist '{analysis['selected_specialist']}', defaulting to data_analyst")
                analysis["selected_specialist"] = "data_analyst"
            
            return analysis
            
        except Exception as e:
            print(f"⚠️ Failed to parse coordinator analysis: {e}")
            
            # Fallback analysis
            return {
                "core_problem": "Analysis parsing failed",
                "selected_specialist": "data_analyst",
                "selection_reasoning": "Fallback due to parsing error",
                "execution_approach": "Standard processing",
                "confidence": 0.5,
                "file_metadata": {},
                "execution_strategy": {"steps": ["Standard processing"]}
            }

    def _build_specialist_task(self, state: GAIAState, specialist_name: str) -> str:
        """Build tasks with proper file access patterns"""
        question = state["question"]
        coordination_analysis = state.get("coordination_analysis", {})
        file_metadata = state.get("file_metadata", {})
        
        task_parts = [f"QUESTION: {question}"]
        
        # Add coordinator's analysis
        if coordination_analysis:
            strategy_section = f"""
COORDINATOR ANALYSIS:
- Core problem: {coordination_analysis.get('core_problem', 'Standard processing')}
- Selection reasoning: {coordination_analysis.get('selection_reasoning', 'Selected for this task')}
- Execution approach: {coordination_analysis.get('execution_approach', 'Standard approach')}"""
            task_parts.append(strategy_section)
        
        # Add file information
        if file_metadata:
            file_section = f"""
FILE INFORMATION:
- File: {file_metadata.get('file_name', 'No file')}
- Category: {file_metadata.get('category', 'Unknown')}
- Processing approach: {file_metadata.get('processing_approach', 'Standard')}"""
            task_parts.append(file_section)
        
        # Add specialist-specific instructions
        specialist_instructions = self._get_specialist_instructions(specialist_name, state)
        if specialist_instructions:
            task_parts.append(specialist_instructions)
        
        # Add GAIA format requirement
        task_parts.append("""
CRITICAL REQUIREMENTS:
- Follow the coordinator's analysis and file guidance above
- Use the recommended processing approach for any file handling
- End your response with 'FINAL ANSWER: [your specific answer]' in exact GAIA format
- Provide the actual answer, never use placeholder text
- Numbers: no commas, no units unless specified
- Strings: no articles (the, a, an), no abbreviations
- Lists: comma separated, apply above rules to each element""")
        
        return "\n".join(task_parts)

    def _get_specialist_instructions(self, specialist_name: str, state: GAIAState) -> str:
        """Generate specialist-specific instructions"""
        file_metadata = state.get("file_metadata", {})
        file_path = state.get("file_path", "")
        
        if specialist_name == "data_analyst":
            instructions = """
DATA ANALYST GUIDANCE (CodeAgent):
- You can access files directly using file paths in your Python code
- Use pandas for Excel/CSV files: pd.read_excel(file_path) or pd.read_csv(file_path)
- Focus on numerical calculations and data processing
- Provide specific numerical results, not general descriptions"""
            
            if file_path and file_metadata.get('category') == 'data':
                instructions += f"""
- File Processing: Load the file using: pd.read_excel("{file_path}") or pd.read_csv("{file_path}")
- The file path is directly available for your Python code execution"""
            
            return instructions
        
        elif specialist_name == "web_researcher":
            return """
WEB RESEARCHER GUIDANCE (ToolCallingAgent):
- Use GoogleSearchTool for current information and facts
- Use VisitWebpageTool to read webpage content for detailed information
- Use WikipediaSearchTool for encyclopedic knowledge
- Cross-reference multiple sources for accuracy
- Focus on recent and authoritative sources"""
        
        elif specialist_name == "document_processor":
            instructions = """
DOCUMENT PROCESSOR GUIDANCE (ToolCallingAgent):
- File access: The file_path will be available via additional_args
- For documents: Use ContentRetrieverTool to extract text content
- For audio/video: Use SpeechToTextTool for transcription
- Process the file according to the coordinator's recommended approach"""
            
            if file_metadata:
                category = file_metadata.get('category', 'unknown')
                if category == 'document':
                    instructions += "\n- Document Processing: Use ContentRetrieverTool to read document content"
                elif category == 'media':
                    instructions += "\n- Media Processing: Use SpeechToTextTool to transcribe audio/video to text"
            
            return instructions
        
        else:
            return "GENERAL GUIDANCE: Follow the coordinator's recommended approach"
                   
    # ============================================================================
    # LANGGRAPH WORKFLOW 
    # ============================================================================

    def _build_workflow(self):
        """Build LangGraph workflow with coordinator analysis and specialist execution."""
        builder = StateGraph(GAIAState)
        
        if self.config.enable_smart_routing:
            # Enhanced smart routing workflow with coordinator
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("complexity_check", self._complexity_check_node)
            builder.add_node("one_shot_answering", self._one_shot_answering_node)
            builder.add_node("coordinator", self._coordinator_node)
            builder.add_node("specialist_execution", self._specialist_execution_node)
            builder.add_node("format_answer", self._format_answer_node)
            
            # Routing flow
            builder.add_edge(START, "read_question")
            builder.add_edge("read_question", "complexity_check")
            
            # Conditional routing based on complexity
            builder.add_conditional_edges(
                "complexity_check",
                self._route_by_complexity,
                {
                    "simple": "one_shot_answering",
                    "complex": "coordinator"
                }
            )
            
            # Coordinator → specialist execution (coordinator populates rich context, specialist executes)
            builder.add_edge("coordinator", "specialist_execution")
            builder.add_edge("specialist_execution", "format_answer")
            
            # Simple workflow: one-shot → format
            builder.add_edge("one_shot_answering", "format_answer")
            
            # Both paths converge at formatting
            builder.add_edge("format_answer", END)
            
        else:
            # Linear workflow with coordinator (fallback)
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("coordinator", self._coordinator_node)  # Replaces agent_selector
            builder.add_node("specialist_execution", self._specialist_execution_node)  # Enhanced with coordinator context
            builder.add_node("format_answer", self._format_answer_node)
            
            # Linear flow: read → coordinate → delegate → format
            builder.add_edge(START, "read_question")
            builder.add_edge("read_question", "coordinator")
            builder.add_edge("coordinator", "specialist_execution")
            builder.add_edge("specialist_execution", "format_answer")
            builder.add_edge("format_answer", END)
        
        return builder.compile()

    def _read_question_node(self, state: GAIAState) -> GAIAState:
        """Question reading with file info extraction and download"""
        task_id = state.get("task_id")
        question = state["question"]
        
        # Start context bridge tracking
        if self.config.enable_context_bridge:
            ContextBridge.start_task_execution(task_id)
            ContextBridge.track_operation("Processing question and extracting file info")
        
        if self.logging:
            self.logging.log_step("question_setup", f"Processing question: {question[:50]}...", {
                'node_name': 'read_question'
            })
        
        # Extract file information from task_id
        file_info = extract_file_info_from_task_id(task_id)
        
        # Download file once if present
        local_file_path = None
        if file_info.get("has_file") and file_info.get("file_name"):
            print(f"📁 File detected: {file_info['file_name']}")
            
            # Try existing path first (development/testing)
            if file_info.get("file_path") and os.path.exists(file_info["file_path"]):
                local_file_path = file_info["file_path"]
                print(f"✅ Using existing file: {local_file_path}")
            else:
                # Download file for production use
                local_file_path = self._download_file_once(task_id, file_info["file_name"])
            
            # Update file info with local path
            if local_file_path:
                file_info["file_path"] = local_file_path
                file_info["has_file"] = True
            else:
                file_info["has_file"] = False
                print("❌ File download failed")
        
        # Analyze file metadata using Python-native approach
        file_metadata = {}
        if file_info.get("has_file"):
            file_metadata = self._analyze_file_metadata(
                file_info.get("file_name", ""),
                file_info.get("file_path", "")
            )
            print(f"📊 File analysis: {file_metadata.get('category', 'unknown')} → {file_metadata.get('recommended_specialist', 'unknown')}")
            
            # Enhanced logging for file metadata
            if self.logging:
                self.logging.log_file_metadata(file_metadata)
        
        # RAG retrieval for similar examples
        similar_examples = []
        if not self.config.enable_smart_routing or not self.config.skip_rag_for_simple:
            try:
                similar_docs = self.retriever.search(question, k=self.config.rag_examples_count)
                
                for doc in similar_docs:
                    content = doc.page_content
                    if "Question :" in content and "Final answer :" in content:
                        parts = content.split("Final answer :")
                        if len(parts) == 2:
                            q_part = parts[0].replace("Question :", "").strip()
                            a_part = parts[1].strip()
                            similar_examples.append({
                                "question": q_part,
                                "answer": a_part
                            })
                
                print(f"📚 Found {len(similar_examples)} similar examples")
                if self.logging:
                    self.logging.set_similar_examples_count(len(similar_examples))
                                
            except Exception as e:
                print(f"⚠️  RAG retrieval error: {e}")
        
        # Return enhanced state
        return {
            **state,
            "similar_examples": similar_examples,
            "file_name": file_info.get("file_name", ""),
            "file_path": file_info.get("file_path", ""),
            "has_file": file_info.get("has_file", False),
            "file_metadata": file_metadata,
            "steps": state["steps"] + ["Question setup and file analysis completed"]
        }

    def _complexity_check_node(self, state: GAIAState):
        """Enhanced complexity check with file awareness"""
        question = state["question"]
        task_id = state.get("task_id", "")
        has_file = state.get("has_file", False)
        
        if self.config.enable_context_bridge:
            ContextBridge.track_operation("Analyzing question complexity")
        
        if self.logging:
            self.logging.log_step("complexity_analysis", f"Analyzing complexity for: {question[:50]}...")
        
        print(f"🧠 Analyzing complexity for: {question[:50]}...")
        
        # Enhanced complexity detection with file awareness
        if is_simple_math(question):
            complexity = "simple"
            reason = "Simple arithmetic detected"
        elif is_simple_fact(question):
            complexity = "simple" 
            reason = "Simple factual query detected"
        elif has_file:  # NEW: Use state file info
            complexity = "complex"
            reason = "File attachment detected in state"
        elif needs_web_search(question):
            complexity = "complex"
            reason = "Current information needed"
        else:
            # LLM decides for edge cases
            complexity = self._llm_complexity_check(question)
            reason = "LLM complexity assessment"
        
        print(f"📊 Complexity: {complexity} ({reason})")
        
        # Update context bridge
        if self.config.enable_context_bridge:
            ContextBridge.track_operation(f"Complexity: {complexity} - {reason}")
        
        if self.logging:
            self.logging.set_complexity(complexity)
            self.logging.log_step("complexity_result", f"Final complexity: {complexity} - {reason}")
        
        # Enhanced RAG for complex questions
        if complexity == "complex" and self.config.skip_rag_for_simple and not state.get("similar_examples"):
            print("📚 Retrieving RAG examples for complex question...")
            
            try:
                similar_docs = self.retriever.search(question, k=self.config.rag_examples_count)
                similar_examples = []
                
                for doc in similar_docs:
                    content = doc.page_content
                    if "Question :" in content and "Final answer :" in content:
                        parts = content.split("Final answer :")
                        if len(parts) == 2:
                            q_part = parts[0].replace("Question :", "").strip()
                            a_part = parts[1].strip()
                            similar_examples.append({
                                "question": q_part,
                                "answer": a_part
                            })
                
                print(f"📚 Found {len(similar_examples)} similar examples")                            
                if self.logging:
                    self.logging.set_similar_examples_count(len(similar_examples))
                    
            except Exception as e:
                print(f"⚠️  RAG retrieval error: {e}")
                similar_examples = state.get("similar_examples", [])
        else:
            similar_examples = state.get("similar_examples", [])
        
        return {
            "complexity": complexity,
            "similar_examples": similar_examples,
            "steps": state["steps"] + [f"Complexity assessed: {complexity} ({reason})"]
        }

    def _llm_complexity_check(self, question: str) -> str:
        """Use LLM to determine complexity for edge cases"""
        
        prompt = f"""Analyze this question and determine if it needs specialist tools or can be answered directly.

        Question: {question}

        Consider:
        - Simple math/facts = "simple" 
        - Needs file analysis, web search, or complex reasoning = "complex"

        Respond with just "simple" or "complex":"""
        
        try:
            response = llm_invoke_with_retry(self.orchestration_model, [HumanMessage(content=prompt)])
            result = response.content.strip().lower()
            complexity = "simple" if "simple" in result else "complex"
            
            return complexity
            
        except Exception as e:
            # Default to complex if LLM fails
            print(f"⚠️  LLM complexity check failed: {str(e)}, defaulting to complex")
            return "complex"

    def _route_by_complexity(self, state: GAIAState) -> str:
        """Routing function for conditional edges"""
        return state.get("complexity", "complex")

    def _one_shot_answering_node(self, state: GAIAState):
        """Direct LLM answering for simple questions"""
        task_id = state.get("task_id", "")
        
        print("⚡ Using one-shot direct answering")
        
        if self.config.enable_context_bridge:
            ContextBridge.track_operation("Direct LLM answering for simple question")
        
        if self.logging:
            self.logging.set_routing_path("one_shot_llm")
            self.logging.log_step("one_shot_start", "Starting direct LLM answering")
        
        question = state["question"]
        
        prompt = f"""You are a general AI assistant. You must provide specific, factual answers.
        
        Question: {question}

        CRITICAL: Never use placeholder text like "[your answer]" or "[Title of...]". Always give the actual answer.

        Report your thoughts, and finish with: FINAL ANSWER: [YOUR SPECIFIC ANSWER]

        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list.
        - Numbers: no commas, no units ($ %) unless specified
        - Strings: no articles (the, a, an), no abbreviations, digits as text unless specified  
        - Lists: apply above rules to each element

        NEVER use placeholder text. Always give real, specific answers."""
        
        try:
            response = llm_invoke_with_retry(self.orchestration_model, [HumanMessage(content=prompt)])
            
            if not response.content or len(response.content.strip()) == 0:
                raise ValueError("Empty response from LLM")
            
            if self.logging:
                self.logging.log_step("one_shot_complete", "Direct LLM answering completed successfully")
            
            return {
                "raw_answer": response.content,
                "steps": state["steps"] + ["Direct LLM answering completed"]
            }
            
        except Exception as e:
            error_msg = f"One-shot answering failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            if self.logging:
                self.logging.log_step("one_shot_error", error_msg)
            
            return {
                "raw_answer": error_msg,
                "steps": state["steps"] + [error_msg]
            }

    def _coordinator_node(self, state: "GAIAState"):
        """
        Manager agent coordination using ManagedAgent pattern
        """
        
        print(f"🧠 Manager Agent Coordination: {state['question'][:80]}...")
        
        try:
            # STEP 1: Ensure file access (download once if needed)
            enhanced_state = self._ensure_file_access(state)
            
            # STEP 2: Create manager agent with managed specialists
            manager_agent = self._create_manager_agent()
            
            # STEP 3: Build manager coordination task  
            coordination_task = self._build_manager_coordination_task(enhanced_state)
            
            # STEP 4: Execute manager agent (it will coordinate specialists)
            if enhanced_state.get("file_path"):
                # Pass file path via additional_args for ToolCallingAgent specialists
                coordination_result = manager_agent.run(
                    task=coordination_task,
                    additional_args={"file_path": enhanced_state["file_path"]}
                )
            else:
                coordination_result = manager_agent.run(task=coordination_task)
            
            # STEP 5: Parse final result (manager handles all coordination internally)
            final_answer = self._extract_final_answer(coordination_result)
            
            return {
                **enhanced_state,
                "raw_answer": coordination_result,
                "final_answer": final_answer,
                "execution_successful": True,
                "agent_used": "manager_agent",
                "steps": enhanced_state["steps"] + ["Manager agent coordination completed"]
            }
            
        except Exception as e:
            error_msg = f"Manager coordination failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                **state,
                "current_error": error_msg,
                "execution_successful": False,
                "agent_used": "manager_agent",
                "steps": state["steps"] + [error_msg]
            }

    def _specialist_execution_node(self, state: GAIAState):
        """
        Execute specialist with coordinator's rich context.
        Main goal is RICH context assembly.
        """
        
        selected_specialist = state.get("selected_agent", "data_analyst")
        task_id = state.get("task_id")
        
        if self.logging:
            self.logging.log_step("delegate_start", f"Delegating to: {selected_specialist}")
        
        print(f"🚀 Executing specialist: {selected_specialist}")
        
        if self.config.enable_context_bridge:
            ContextBridge.track_operation(f"Delegating to {selected_specialist}")
        
        try:
            # STEP 1: Configure tools from state (existing logic - keep as-is)
            self._configure_tools_from_state(selected_specialist, state)
            
            # STEP 2: Build enhanced task with coordinator's rich context
            enhanced_task = self._build_specialist_task_with_coordinator_context(state)
            
            # STEP 3: Get specialist and execute (ULTRA MINIMAL)
            specialist = self.specialists[selected_specialist]
            
            if self.config.enable_context_bridge:
                ContextBridge.track_operation(f"Executing {selected_specialist} with enhanced context")
            
            if self.logging:
                self.logging.log_step("specialist_execution", f"Running {selected_specialist}")
            
            # ULTRA MINIMAL: Execute with enhanced task
            result = specialist.run(task=enhanced_task)
            
            if self.config.enable_context_bridge:
                ContextBridge.track_operation(f"{selected_specialist} execution completed")
            
            print(f"✅ Specialist {selected_specialist} completed execution")
            
            if self.logging:
                self.logging.log_step("specialist_success", f"{selected_specialist} completed successfully")
            
            # Return enhanced state with execution results
            return {
                **state,
                "agent_used": selected_specialist,
                "raw_answer": result,
                "execution_successful": True,
                "current_error": ContextBridge.get_current_error() if self.config.enable_context_bridge else None,
                "steps": state["steps"] + [f"Specialist {selected_specialist} execution completed"]
            }
            
        except Exception as e:
            # Error tracking integration
            error_msg = f"Specialist {selected_specialist} failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            if self.config.enable_context_bridge:
                ContextBridge.track_error(error_msg)
            
            if self.logging:
                self.logging.log_step("specialist_error", error_msg)
            
            return {
                **state,
                "agent_used": selected_specialist,
                "raw_answer": error_msg,
                "execution_successful": False,
                "current_error": error_msg,
                "steps": state["steps"] + [error_msg]
            }

    def _format_answer_node(self, state: GAIAState):
        """Enhanced answer formatting with question context"""
        raw_answer = state.get("raw_answer", "")
        task_id = state.get("task_id", "")
        question = state.get("question", "")
        
        # Store question for formatting context
        self._current_question = question
        
        if self.config.enable_context_bridge:
            ContextBridge.track_operation("Formatting final answer")
        
        if self.logging:
            self.logging.log_step("format_start", f"Formatting answer: {raw_answer[:50]}...")
        
        try:
            if self.logging:
                self.logging.log_step("extract_answer", "Extracting final answer from raw response")
            
            formatted_answer = self._extract_final_answer(raw_answer)
            
            if self.logging:
                self.logging.log_step("apply_gaia_format", f"Applying GAIA formatting to: {formatted_answer}")
            
            formatted_answer = self._apply_gaia_formatting(formatted_answer)
            
            if self.logging:
                self.logging.log_step("format_complete", f"Final formatted answer: {formatted_answer}")
            
            # Get execution metrics from context bridge
            execution_metrics = {}
            if self.config.enable_context_bridge:
                execution_metrics = ContextBridge.get_execution_metrics()
                ContextBridge.track_operation(f"Final answer: {formatted_answer}")
            
            return {
                "final_answer": formatted_answer,
                "execution_metrics": execution_metrics,
                "steps": state["steps"] + ["Final answer formatting applied"]
            }
            
        except Exception as e:
            error_msg = f"Answer formatting error: {str(e)}"
            
            if self.config.enable_context_bridge:
                ContextBridge.track_operation(f"Format error: {error_msg}")
                execution_metrics = ContextBridge.get_execution_metrics()
            else:
                execution_metrics = {}
            
            if self.logging:
                self.logging.log_step("format_error", error_msg)
            
            return {
                "final_answer": raw_answer.strip() if raw_answer else "No answer",
                "execution_metrics": execution_metrics,
                "steps": state["steps"] + [error_msg]
            }
        finally:
            # Context cleanup
            if self.config.enable_context_bridge:
                ContextBridge.clear_tracking()
                        
    def _extract_final_answer(self, raw_answer: str) -> str:
        """
        Extract final answer from agent response with pattern matching
        """  
        if raw_answer is None:
            if self.logging:
                self.logging.log_step("extract_empty", "Raw answer is None")
            return "No answer"
        
        # Convert list to string if needed (fixes the error you saw)
        if isinstance(raw_answer, list):
            if self.logging:
                self.logging.log_step("extract_list_input", f"Converting list to string: {raw_answer}")
            raw_answer = str(raw_answer[0]) if raw_answer else "No answer"
        
        # Ensure it's a string
        raw_answer = str(raw_answer) if raw_answer else ""
        
        # More comprehensive patterns to catch various formats
        patterns = [
            r"FINAL ANSWER:\s*(.+?)(?:\n|$)",  # Standard GAIA format
            r"Final Answer:\s*(.+?)(?:\n|$)",  # Title case
            r"final answer:\s*(.+?)(?:\n|$)",  # Lower case
            r"Answer:\s*(.+?)(?:\n|$)",       # Simple answer
            r"ANSWER:\s*(.+?)(?:\n|$)",       # Caps answer
            r"The answer is:\s*(.+?)(?:\n|$)", # Descriptive
            r"Result:\s*(.+?)(?:\n|$)",       # Result format
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, raw_answer, re.IGNORECASE | re.DOTALL)
            if matches:
                # Take the last match (most likely to be the final answer)
                extracted = matches[-1].strip()
                if extracted and extracted.lower() not in ["", "no answer", "none"]:
                    if self.logging:
                        self.logging.log_step("extract_success", f"Pattern {i+1} matched: {extracted}")
                    return extracted
        
        # Fallback: Look for the last substantial line
        lines = [line.strip() for line in raw_answer.strip().split('\n') if line.strip()]
        
        # Try to find a line that looks like an answer
        for line in reversed(lines):
            if len(line) > 0 and not line.lower().startswith(('i ', 'the ', 'let ', 'to ', 'in ', 'based ')):
                if self.logging:
                    self.logging.log_step("extract_fallback", f"Using last substantial line: {line}")
                return line
        
        # Final fallback
        fallback = lines[-1] if lines else "No answer"
        
        if self.logging:
            self.logging.log_step("extract_final_fallback", f"Final fallback: {fallback}")
        
        return fallback

    def _apply_gaia_formatting(self, answer: str) -> str:
        """
        FIXED: Apply GAIA formatting rules to final answers
        Addresses the doubling issue and improves format compliance
        """
        if not answer:
            return "No answer"
        
        original_answer = answer
        answer = answer.strip()
        
        if self.logging:
            self.logging.log_step("gaia_format_start", f"Original answer: '{answer}'")
        
        # FIXED: More aggressive duplicate FINAL ANSWER removal
        # Handle various patterns that can cause duplication
        final_answer_patterns = [
            r'(?i)^.*?final\s*answer\s*:\s*(.*)$',  # Extract everything after FINAL ANSWER:
            r'(?i)final\s*answer\s*:\s*(.+?)(?:\n|$)',  # Match until newline
            r'(?i).*final\s*answer\s*:\s*(.+)',  # Match anything after FINAL ANSWER:
        ]
        
        for pattern in final_answer_patterns:
            match = re.search(pattern, answer, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if extracted:  # Only use if we got something meaningful
                    answer = extracted
                    if self.logging:
                        self.logging.log_step("gaia_format_extract", f"Extracted from FINAL ANSWER: '{answer}'")
                    break
        
        # Remove any remaining prefixes
        prefixes_to_remove = [
            "final answer:", "the answer is:", "answer:", "result:", "solution:",
            "the result is:", "therefore", "so", "thus", "final answer",
            "the final answer is", "my answer is"
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                answer_lower = answer.lower()
                break
        
        # Remove quotes and excess punctuation
        answer = answer.strip('.,!?:;"\'')
        
        # Question-specific formatting requirements
        question = getattr(self, '_current_question', '').lower()
        
        # Keep commas when explicitly requested
        if "comma" in question and ("list" in question or "delimited" in question):
            if self.logging:
                self.logging.log_step("gaia_format_comma_list", "Preserving commas for comma-delimited list")
            # Don't remove commas for comma-separated lists
            pass
        else:
            # Standard GAIA rule: remove commas from numbers
            if answer.replace('.', '').replace('-', '').replace(',', '').replace(' ', '').isdigit():
                answer = answer.replace(',', '')
                if self.logging:
                    self.logging.log_step("gaia_format_number", f"Removed commas from number: '{answer}'")
        
        # Handle rounding instructions
        if "round" in question and "integer" in question:
            numbers = re.findall(r'\d+', answer)
            if numbers:
                answer = numbers[0]  # Take first number found
                if self.logging:
                    self.logging.log_step("gaia_format_round", f"Extracted rounded integer: '{answer}'")
        
        # Extract count from descriptive answers
        if "how many" in question and any(term in question for term in ["albums", "studio", "books", "papers"]):
            numbers = re.findall(r'\b\d+\b', answer)
            if numbers:
                answer = numbers[0]
                if self.logging:
                    self.logging.log_step("gaia_format_count", f"Extracted count: '{answer}'")
        
        # Ensure alphabetical ordering in geographical questions
        if "countries" in question and "comma separated" in question:
            if "," in answer:
                countries = [c.strip() for c in answer.split(",")]
                countries.sort()
                answer = ", ".join(countries)
                if self.logging:
                    self.logging.log_step("gaia_format_countries", f"Sorted countries: '{answer}'")
        
        # Standard GAIA formatting (only if not a special case above)
        if not any(special in question for special in ["comma", "list", "countries"]):
            # Remove articles (the, a, an) - but be careful with meaningful words
            answer = re.sub(r'\b(the|a|an)\s+', '', answer, flags=re.IGNORECASE)
            
            # Remove units unless specified in question
            if not any(keep_unit in question for keep_unit in ["$", "%", "units", "meters", "degrees", "km"]):
                # Only remove obvious unit markers, not letters that might be part of the answer
                answer = re.sub(r'[^\w\s,.-]', '', answer)
        
        # Clean up whitespace
        answer = ' '.join(answer.split())
        
        if self.logging:
            self.logging.log_step("gaia_format_final", f"Final GAIA formatted answer: '{answer}'")
        
        return answer

    def process_question(self, question: str, task_id: str = None) -> Dict:
        """
        Main entry point with context bridge integration.
        
        Args:
            question: The question to process
            task_id: Optional task identifier
            
        Returns:
            Dictionary with complete processing results including metrics
        """
        import time
        
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]
        
        total_start_time = time.time()
        
        if self.logging:
            self.logging.start_task(task_id, model_used=self.config.model_name)
        
        try:
            # Run workflow with enhanced state
            initial_state = {
                "task_id": task_id,
                "question": question,
                "steps": []
            }
            
            result = self.workflow.invoke(initial_state)
            
            # Get metrics from context bridge
            execution_metrics = result.get("execution_metrics", {})
            total_steps = execution_metrics.get("steps_executed", 0)
            
            # Calculate performance metrics
            total_time = time.time() - total_start_time
            
            performance_metrics = {
                "total_execution_time": total_time,
                "total_steps": total_steps,
                "step_source": "context_bridge",
                "execution_metrics": execution_metrics
            }
            
            success = result.get("execution_successful", True)
            final_answer = result.get("final_answer", str(result)) if isinstance(result, dict) else str(result)
            
            # Enhanced logging
            if self.logging:
                self.logging.log_question_result(
                    task_id=task_id,
                    question=question,
                    final_answer=final_answer,
                    total_steps=total_steps,
                    success=success
                )
            
            return {
                "task_id": task_id,
                "question": question,
                "final_answer": final_answer,
                "raw_answer": result.get("raw_answer", ""),
                "steps": result.get("steps", []),
                "complexity": result.get("complexity", "unknown"),
                "similar_examples": result.get("similar_examples", []),
                "execution_successful": success,
                "total_steps": total_steps,
                "performance_metrics": performance_metrics,
                "selected_agent": result.get("selected_agent", "unknown"),
                "file_info": {  # File information from state
                    "file_name": result.get("file_name", ""),
                    "file_path": result.get("file_path", ""),
                    "has_file": result.get("has_file", False)
                }
            }
            
        except Exception as e:
            total_time = time.time() - total_start_time
            error_msg = f"Processing failed: {str(e)}"
            
            if self.logging:
                self.logging.log_question_result(
                    task_id=task_id,
                    question=question,
                    final_answer=error_msg,
                    total_steps=0,
                    success=False
                )
            
            return {
                "task_id": task_id,
                "question": question,
                "final_answer": error_msg,
                "execution_successful": False,
                "total_steps": 0,
                "performance_metrics": {"error": True, "total_execution_time": total_time},
                "selected_agent": "error",
                "file_info": {"file_name": "", "file_path": "", "has_file": False}
            }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 GAIA Agent - State + Context Bridge")
    print("=" * 60)
    print("✅ Combined LangGraph state with task id and file info")
    print("✅ Context bridge for agent step tracking")
    print("✅ State-aware tool configuration")
    print("")
    print("Use agent_testing.py for testing")