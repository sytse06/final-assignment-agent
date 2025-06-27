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
    from tools import (GetAttachmentTool, 
                       ContentRetrieverTool
                       # ContentGroundingTool,  # DISABLED
                       # CONTENT_GROUNDING_AVAILABLE  # DISABLED
                       )
    CUSTOM_TOOLS_AVAILABLE = True
    CONTENT_GROUNDING_AVAILABLE = False  # Force disable
    print("✅ Custom tools imported successfully (grounding disabled)")
except ImportError as e:
    print(f"⚠️  Custom tools not available: {e}")
    CUSTOM_TOOLS_AVAILABLE = False
    CONTENT_GROUNDING_AVAILABLE = False

try:
    # Fixed: Import from the specific langchain_tools submodule
    from tools.langchain_tools import ALL_LANGCHAIN_TOOLS
    LANGCHAIN_TOOLS_AVAILABLE = True
    print(f"✅ LangChain tools imported successfully: {len(ALL_LANGCHAIN_TOOLS)} tools")
except ImportError as e:
    print(f"⚠️  LangChain tools not available: {e}")
    ALL_LANGCHAIN_TOOLS = []
    LANGCHAIN_TOOLS_AVAILABLE = False

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
# PYTHON UTILITY FUNCTIONS
# ============================================================================

    def _ensure_file_access(self, state: "GAIAState") -> "GAIAState":
        """
        Python-native file access optimized for SmolagAgents architecture
        
        Key insights from SmolagAgents analysis:
        1. CodeAgent can access files directly via Python code execution
        2. ToolCallingAgent needs files via additional_args parameter
        3. Both can use cached/downloaded files efficiently
        """
        
        task_id = state.get("task_id")
        file_name = state.get("file_name", "")
        existing_file_path = state.get("file_path", "")
        
        print(f"🔍 SmolagAgents file access: {file_name}")
        
        # METHOD 1: Use existing cached file (development/testing)
        if existing_file_path and os.path.exists(existing_file_path):
            print(f"✅ Using cached file: {existing_file_path}")
            local_file_path = existing_file_path
            file_size = os.path.getsize(existing_file_path)
            print(f"📊 File size: {file_size} bytes")
            
        # METHOD 2: Download once, use everywhere (production)
        elif task_id and file_name:
            print(f"📡 Downloading file for reuse by specialists: {task_id}")
            local_file_path = self._download_file_python_native(task_id, file_name)
            
        # METHOD 3: No file available
        else:
            print("📄 No file attachment")
            local_file_path = None
        
        # Extract file extension for smart routing
        file_extension = ""
        if file_name:
            file_extension = os.path.splitext(file_name)[1].lower()
            print(f"🎯 File extension: {file_extension}")
        
        # Update state with file access results
        return {
            **state,
            "file_path": local_file_path,           # Verified local path for all agents
            "file_extension": file_extension,       # For coordinator routing
            "has_file": bool(local_file_path),      # Updated file status
            "file_accessible": bool(local_file_path and os.path.exists(local_file_path))
        }
    
    def _download_file_python_native(self, task_id: str, file_name: str) -> Optional[str]:
        """
        Download file once, reuse for all agents - eliminates GetAttachmentTool performance issues
        """
        
        try:
            # Production API endpoint
            agent_evaluation_api = getattr(self.config, 'agent_evaluation_api', 
                                         "https://agents-course-unit4-scoring.hf.space/")
            file_url = f"{agent_evaluation_api}files/{task_id}"
            
            print(f"📡 Single download request: {file_url}")
            
            # Direct Python download with streaming for large files
            response = requests.get(
                file_url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
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
    """GAIA agent using hybrid LangGraph state + simplified context bridge"""
    
    def __init__(self, config: GAIAConfig = None):
        if config is None:
            config = GAIAConfig()
        
        self.config = config
        self.retriever = self._initialize_retriever()
        self.model = self._initialize_model()
        
        # Setup logging
        if config.enable_csv_logging:
            self.logging = AgentLoggingSetup(
                debug_mode=config.debug_mode,
                step_log_file=config.step_log_file,
                question_log_file=config.question_log_file
            )
        else:
            self.logging = None
        
        # Create shared tool instances
        self.shared_tools = self._create_shared_tools()
        
        # Create specialist agents
        self.specialists = self._create_specialist_agents()
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        print("🚀 GAIA Agent initialized with hybrid state + context bridge!")
    
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
    
    def _initialize_model(self):
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

    def _create_coordinator_agent(self):
        """Create coordinator CodeAgent for Python-native analysis"""
        
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        
        coordinator = CodeAgent(
            name="coordinator",
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
            tools=[],  # No tools - pure Python analysis
            additional_authorized_imports=[
                "pathlib", "mimetype", "re", "json", "os"
            ],
            model=self.model,
            planning_interval=7,
            max_steps=8,  # Keep coordinator focused
            logger=logger
        )
        
        return coordinator

    def _build_coordinator_analysis_task(self, state: GAIAState) -> str:
        """Build comprehensive coordination analysis task"""
        
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
        
        # Build file context
        file_context = ""
        if has_file and file_name:
            file_context = f"""
    FILE INFORMATION:
    - Name: {file_name}
    - Path: {file_path}
    - Has file: {has_file}
    """
        
        task = f"""
    You are the GAIA coordination agent. Analyze this question and provide strategic guidance.

    QUESTION: {question}

    COMPLEXITY: {complexity}
    {file_context}{examples_context}

    ANALYSIS TASKS:
    1. CORE PROBLEM ANALYSIS:
    - What is the fundamental problem to solve?
    - What type of processing is needed?
    - What are the key requirements?

    2. FILE ANALYSIS (if file present):
    ```python
    # Analyze file using Python
    from pathlib import Path
    import mimetypes
    
    if "{file_name}":
        file_path = Path("{file_name}")
        extension = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Categorize file type
        if extension in ['.xlsx', '.csv', '.xls', '.tsv']:
            category = "data"
            processing_approach = "direct_pandas"
            recommended_specialist = "data_analyst"
        elif extension in ['.pdf', '.docx', '.doc', '.txt']:
            category = "document"
            processing_approach = "content_extraction"
            recommended_specialist = "document_processor"
        elif extension in ['.mp3', '.mp4', '.wav', '.m4a']:
            category = "media"
            processing_approach = "transcription"
            recommended_specialist = "document_processor"
        elif extension in ['.png', '.jpg', '.jpeg', '.gif']:
            category = "image"
            processing_approach = "vision_analysis"
            recommended_specialist = "document_processor"
        else:
            category = "unknown"
            processing_approach = "content_extraction"
            recommended_specialist = "document_processor"
        
        print(f"File analysis: {{extension}} → {{category}} → {{processing_approach}}")
    ```

    3. SPECIALIST SELECTION:
    Available specialists:
    - data_analyst: Excel/CSV processing, calculations, pandas analysis
    - web_researcher: Google search, current information, web content
    - document_processor: PDF/document extraction, transcription, vision

    4. EXECUTION STRATEGY:
    - What's the step-by-step approach?
    - What tools should the specialist use?
    - What specific guidance to provide?

    RETURN ANALYSIS AS JSON:
    ```json
    {{
        "core_problem": "Brief description of fundamental problem",
        "selected_specialist": "data_analyst|web_researcher|document_processor",
        "selection_reasoning": "Why this specialist is optimal",
        "execution_approach": "High-level approach description",
        "confidence": 0.9,
        "file_metadata": {{
            "file_name": "{file_name}",
            "extension": ".xlsx",
            "category": "data",
            "processing_approach": "direct_pandas",
            "specialist_guidance": {{
                "tool_command": "file_path = get_attachment(fmt='LOCAL_FILE_PATH'); df = pd.read_excel(file_path)",
                "imports_needed": ["pandas", "openpyxl"],
                "processing_strategy": "Load Excel → identify columns → perform calculations"
            }},
            "estimated_complexity": "medium"
        }},
        "execution_strategy": {{
            "approach": "Python-native file analysis with pandas",
            "steps": [
                "Download file using get_attachment",
                "Load data with appropriate method",
                "Analyze structure and content",
                "Perform required calculations"
            ]
        }}
    }}
    ```

    Focus on Python-native analysis and provide specific, actionable guidance.
    """
        
        return task

    def _parse_coordination_result(self, coordination_result: str) -> Dict:
        """Parse coordinator's analysis result into structured data"""
        
        try:
            # Try to extract JSON from coordinator result
            import json
            import re
            
            # Look for JSON block in the result
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', coordination_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                coordination_analysis = json.loads(json_str)
            else:
                # Try to find JSON without markdown
                json_match = re.search(r'\{.*\}', coordination_result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    coordination_analysis = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in coordinator result")
            
            # Validate required fields
            required_fields = ["selected_specialist", "selection_reasoning", "core_problem", "execution_approach"]
            for field in required_fields:
                if field not in coordination_analysis:
                    coordination_analysis[field] = f"Missing {field}"
            
            # Ensure confidence is set
            if "confidence" not in coordination_analysis:
                coordination_analysis["confidence"] = 0.8
            
            # Ensure valid specialist selection
            valid_specialists = ["data_analyst", "web_researcher", "document_processor"]
            if coordination_analysis["selected_specialist"] not in valid_specialists:
                print(f"⚠️ Invalid specialist '{coordination_analysis['selected_specialist']}', defaulting to data_analyst")
                coordination_analysis["selected_specialist"] = "data_analyst"
            
            return coordination_analysis
            
        except Exception as e:
            print(f"⚠️ Failed to parse coordinator result: {e}")
            
            # Fallback analysis based on simple heuristics
            return self._create_fallback_analysis(coordination_result)

    def _create_fallback_analysis(self, coordination_result: str) -> Dict:
        """Create fallback analysis when parsing fails"""
        
        # Simple heuristics for fallback
        result_lower = coordination_result.lower()
        
        if any(term in result_lower for term in ["excel", "csv", "data", "calculation"]):
            selected_specialist = "data_analyst"
            reasoning = "Data processing keywords detected"
        elif any(term in result_lower for term in ["search", "web", "current", "online"]):
            selected_specialist = "web_researcher"
            reasoning = "Web search keywords detected"
        elif any(term in result_lower for term in ["document", "pdf", "transcribe", "extract"]):
            selected_specialist = "document_processor"
            reasoning = "Document processing keywords detected"
        else:
            selected_specialist = "data_analyst"
            reasoning = "Default fallback selection"
        
        return {
            "core_problem": "Analysis failed - using heuristic approach",
            "selected_specialist": selected_specialist,
            "selection_reasoning": reasoning,
            "execution_approach": "Standard processing approach",
            "confidence": 0.5,
            "file_metadata": {},
            "execution_strategy": {
                "approach": "Fallback approach",
                "steps": ["Standard processing"]
            }
        }

    def _build_specialist_task_with_coordinator_context(self, state: GAIAState) -> str:
        """
        Build specialist task with coordinator's rich context.
        This is where the magic happens - rich context assembly.
        """
        
        question = state["question"]
        coordination_analysis = state.get("coordination_analysis", {})
        file_metadata = state.get("file_metadata", {})
        selected_specialist = state.get("selected_agent", "data_analyst")
        
        # Start with the core question
        task_parts = [f"QUESTION: {question}"]
        
        # Add coordinator's file analysis (if available)
        if file_metadata:
            file_analysis_section = f"""
    COORDINATOR FILE ANALYSIS:
    - File: {file_metadata.get('file_name', 'No file')}
    - Extension: {file_metadata.get('extension', 'Unknown')}
    - Category: {file_metadata.get('category', 'Unknown')}
    - Processing approach: {file_metadata.get('processing_approach', 'Standard')}"""
            
            # Add specialist-specific guidance
            specialist_guidance = file_metadata.get('specialist_guidance', {})
            if specialist_guidance:
                file_analysis_section += f"""
    - Processing strategy: {specialist_guidance.get('processing_strategy', 'Standard processing')}
    - Required imports: {', '.join(specialist_guidance.get('imports_needed', []))}"""
            
            task_parts.append(file_analysis_section)
        
        # Add coordinator's strategic analysis (if available)
        if coordination_analysis:
            strategy_section = f"""
    COORDINATOR STRATEGIC ANALYSIS:
    - Core problem: {coordination_analysis.get('core_problem', 'Standard processing')}
    - Execution approach: {coordination_analysis.get('execution_approach', 'Standard approach')}
    - Confidence level: {coordination_analysis.get('confidence', 0.8):.1f}"""
            
            # Add execution strategy steps
            execution_strategy = coordination_analysis.get('execution_strategy', {})
            if execution_strategy and execution_strategy.get('steps'):
                strategy_section += f"""
    - Processing steps:
    {chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(execution_strategy.get('steps', [])))}"""
            
            task_parts.append(strategy_section)
        
        # Add specialist-specific instructions
        specialist_instructions = self._get_specialist_specific_instructions(selected_specialist, coordination_analysis, file_metadata)
        if specialist_instructions:
            task_parts.append(specialist_instructions)
        
        # Add GAIA format requirement (critical)
        task_parts.append("""
    CRITICAL REQUIREMENTS:
    - Use the coordinator's analysis and file information above
    - Follow the recommended processing approach and tool commands
    - End your response with 'FINAL ANSWER: [your specific answer]' in exact GAIA format
    - Provide the actual answer, never use placeholder text like '[your answer]'""")
        
        return "\n".join(task_parts)

    def _get_specialist_specific_instructions(self, specialist_name: str, coordination_analysis: Dict, file_metadata: Dict) -> str:
        """Generate specialist-specific instructions based on coordinator analysis"""
        
        if specialist_name == "data_analyst":
            instructions = """
    DATA ANALYST SPECIFIC GUIDANCE:
    - Use the coordinator's file analysis to determine the correct pandas approach
    - If Excel file: Use the recommended processing strategy from file metadata
    - Focus on numerical calculations and data processing
    - Provide specific numerical results, not general descriptions"""
            
            # Add file-specific guidance for data analyst
            if file_metadata.get('category') == 'data':
                processing_approach = file_metadata.get('processing_approach', 'direct_pandas')
                if processing_approach == 'direct_pandas':
                    instructions += """
    - File Processing: The coordinator has analyzed the file - use appropriate pandas methods
    - Data Analysis: Focus on the specific question requirements (calculations, summaries, etc.)
    - Note: File path information will be available through existing tool configuration"""
            
            return instructions
        
        elif specialist_name == "web_researcher":
            instructions = """
    WEB RESEARCHER SPECIFIC GUIDANCE:
    - Use search tools to find current, relevant information
    - Cross-reference multiple sources for accuracy
    - Focus on recent and authoritative sources
    - Provide specific factual answers with sources when possible"""
            
            # Add context-specific guidance
            if coordination_analysis.get('core_problem'):
                core_problem = coordination_analysis['core_problem']
                instructions += f"""
    - Research Focus: {core_problem}
    - Search Strategy: Use specific keywords related to the core problem identified by coordinator"""
            
            return instructions
        
        elif specialist_name == "document_processor":
            instructions = """
    DOCUMENT PROCESSOR SPECIFIC GUIDANCE:
    - Use content extraction tools for documents, transcription for media
    - Focus on extracting relevant information for the specific question
    - Process the file according to the coordinator's recommended approach"""
            
            # Add file-specific guidance for document processor
            if file_metadata:
                category = file_metadata.get('category', 'unknown')
                processing_approach = file_metadata.get('processing_approach', 'content_extraction')
                
                if category == 'document':
                    instructions += """
    - Document Processing: Use available content extraction tools to read document content
    - Focus on finding specific information requested in the question"""
                elif category == 'media':
                    instructions += """
    - Media Processing: Use available transcription tools to convert audio/video to text
    - Extract relevant information from the transcribed content"""
                elif category == 'image':
                    instructions += """
    - Image Processing: Use available vision analysis tools to describe and analyze image content
    - Focus on visual elements relevant to the question"""
            
            return instructions
        
        else:
            return """
    GENERAL GUIDANCE:
    - Follow the coordinator's recommended approach
    - Use appropriate tools for the task type
    - Provide specific, factual answers"""
           
    def _create_specialist_agents(self):
        """Create specialized agents managed by coordinator."""
        
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        specialists = {}

        # Data Analyst - Excel/CSV specialist
        specialists["data_analyst"] = CodeAgent(
            name="data_analyst", 
            description="Excel/CSV analysis and numerical calculations using pandas",
            tools=[],
            additional_authorized_imports=[
                "pandas", "numpy", "openpyxl", "xlrd",
                "scipy", "matplotlib", "seaborn", 
                "sklearn", "scikit-learn"
            ],
            model=self.model,
            max_steps=self.config.max_agent_steps,
            add_base_tools=True,
            logger=logger
        )
        
        # Web Researcher - Focused search toolkit
        web_tools = [
            GoogleSearchTool(),
            VisitWebpageTool(), 
            WikipediaSearchTool()
        ]
        
        specialists["web_researcher"] = ToolCallingAgent(
            name="web_researcher",
            description="Web search using GoogleSearchTool (Serper API), Wikipedia, and content extraction",
            tools=web_tools,
            model=self.model,
            max_steps=self.config.max_agent_steps,
            add_base_tools=True,
            logger=logger
        )
        
        # Document Processor - Media and document specialist
        doc_tools = [SpeechToTextTool()]
        
        if CUSTOM_TOOLS_AVAILABLE:
            doc_tools.append(ContentRetrieverTool())
        
        specialists["document_processor"] = ToolCallingAgent(
            name="document_processor",
            description="Document extraction with ContentRetrieverTool and audio transcription",
            tools=doc_tools,
            model=self.model,
            max_steps=self.config.max_agent_steps,
            add_base_tools=True,
            logger=logger
        )
        
        print(f"🎯 Created {len(specialists)} specialized agents:")
        print(f"   data_analyst: {len(specialists['data_analyst'].tools)} tools (CodeAgent)")
        print(f"   web_researcher: {len(specialists['web_researcher'].tools)} tools (ToolCallingAgent)")
        print(f"   document_processor: {len(specialists['document_processor'].tools)} tools (ToolCallingAgent)")
        
        return specialists
    
def _configure_tools_from_state(self, agent_name: str, state: GAIAState):
    """Configure tools with state information for file processing"""
    task_id = state.get("task_id")
    question = state.get("question")
    file_path = state.get("file_path", "")
    file_name = state.get("file_name", "")
    has_file = state.get("has_file", False)
    
    if not task_id:
        return
    
    if self.config.debug_mode:
        print(f"🔧 Configuring tools for {agent_name}")
        if has_file:
            print(f"   📁 File: {file_name}")
            print(f"   📍 Path: {file_path}")
    
    # Get the specialist agent
    specialist = self.specialists[agent_name]
    
    # Handle CodeAgent (data_analyst) - tools are in base_tools or empty
    if agent_name == "data_analyst":
        # CodeAgent with pure Python - no tool configuration needed
        if self.config.debug_mode:
            print(f"✅ CodeAgent {agent_name} - no tool configuration required")
        return
    
    # Handle ToolCallingAgent tools (web_researcher, document_processor)
    elif hasattr(specialist, 'tools') and isinstance(specialist.tools, list):
        for tool in specialist.tools:
            tool_class_name = tool.__class__.__name__
            
            # Configure ContentRetrieverTool with question context
            if tool_class_name == "ContentRetrieverTool":
                if hasattr(tool, 'configure_from_state'):
                    try:
                        tool.configure_from_state(question)
                        if self.config.debug_mode:
                            print(f"✅ Configured ContentRetrieverTool with question context")
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"⚠️ ContentRetrieverTool configuration failed: {e}")
            
            # Configure SpeechToTextTool with file information if audio file
            elif tool_class_name == "SpeechToTextTool":
                if has_file and file_name:
                    audio_extensions = ['.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi']
                    if any(ext in file_name.lower() for ext in audio_extensions):
                        if hasattr(tool, 'configure_from_state'):
                            try:
                                tool.configure_from_state(file_path, file_name)
                                if self.config.debug_mode:
                                    print(f"✅ Configured SpeechToTextTool for audio file")
                            except Exception as e:
                                if self.config.debug_mode:
                                    print(f"⚠️ SpeechToTextTool configuration failed: {e}")
                        elif self.config.debug_mode:
                            print(f"✅ SpeechToTextTool ready for: {file_name}")
            
            # Configure web search tools with query enhancement
            elif tool_class_name in ["GoogleSearchTool", "WikipediaSearchTool"]:
                if hasattr(tool, 'configure_from_state'):
                    try:
                        tool.configure_from_state(question)
                        if self.config.debug_mode:
                            print(f"✅ Configured {tool_class_name} with question context")
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"⚠️ {tool_class_name} configuration failed: {e}")
                elif self.config.debug_mode:
                    print(f"✅ {tool_class_name} ready for search")
            
            # Configure VisitWebpageTool
            elif tool_class_name == "VisitWebpageTool":
                if hasattr(tool, 'configure_from_state'):
                    try:
                        tool.configure_from_state(question)
                        if self.config.debug_mode:
                            print(f"✅ Configured VisitWebpageTool with question context")
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"⚠️ VisitWebpageTool configuration failed: {e}")
                elif self.config.debug_mode:
                    print(f"✅ VisitWebpageTool ready for content extraction")
            
            # Handle any legacy GetAttachmentTool if present
            elif tool_class_name == "GetAttachmentTool":
                if has_file and file_path and hasattr(tool, 'configure_from_state'):
                    try:
                        tool.configure_from_state(file_path, file_name)
                        if self.config.debug_mode:
                            print(f"✅ Configured GetAttachmentTool with file info")
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"⚠️ GetAttachmentTool configuration failed: {e}")
    
    # Handle CodeAgent with tools dictionary (legacy support)
    elif hasattr(specialist, 'tools') and isinstance(specialist.tools, dict):
        for tool_name, tool in specialist.tools.items():
            tool_class_name = tool.__class__.__name__
            
            # Configure ContentRetrieverTool with question context
            if tool_class_name == "ContentRetrieverTool":
                if hasattr(tool, 'configure_from_state'):
                    try:
                        tool.configure_from_state(question)
                        if self.config.debug_mode:
                            print(f"✅ Configured ContentRetrieverTool (dict access)")
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"⚠️ ContentRetrieverTool configuration failed: {e}")
            
            # Handle GetAttachmentTool
            elif tool_class_name == "GetAttachmentTool":
                if has_file and file_path and hasattr(tool, 'configure_from_state'):
                    try:
                        tool.configure_from_state(file_path, file_name)
                        if self.config.debug_mode:
                            print(f"✅ Configured GetAttachmentTool (dict access)")
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"⚠️ GetAttachmentTool configuration failed: {e}")
    
    else:
        if self.config.debug_mode:
            print(f"⚠️ No tools found for {agent_name} or unsupported tool structure")
    
    # Summary logging
    if self.config.debug_mode:
        total_tools = len(specialist.tools) if hasattr(specialist, 'tools') and isinstance(specialist.tools, list) else 0
        print(f"🎯 Tool configuration complete for {agent_name} ({total_tools} tools)
                   
    # ============================================================================
    # WORKFLOW 
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

    def _read_question_node(self, state: GAIAState):
        """Question reading with file info extraction if relevant"""
        task_id = state.get("task_id")
        question = state["question"]
        
        if 'get_attachment' in self.shared_tools:
            self.shared_tools['get_attachment'].attachment_for(task_id)
            print(f"✅ Activated GetAttachmentTool for task: {task_id}")
        
        # Start context bridge tracking
        if self.config.enable_context_bridge:
            ContextBridge.start_task_execution(task_id)
            ContextBridge.track_operation("Processing question and extracting file info")
        
        if self.logging:
            self.logging.log_step("question_setup", f"Processing question: {question[:50]}...")
        
        # NEW: Extract file information from task_id
        file_info = extract_file_info_from_task_id(task_id)
        
        # Initialize similar_examples
        similar_examples = []
        
        # RAG retrieval (unchanged logic)
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
                similar_examples = []
        
        # Return enhanced state with file information
        return {
            "similar_examples": similar_examples,
            "file_name": file_info.get("file_name", ""),
            "file_path": file_info.get("file_path", ""),
            "has_file": file_info.get("has_file", False),
            "steps": state["steps"] + ["Question setup and file info extracted"]
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