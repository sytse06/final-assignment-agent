
# GAIA Agent System using SmolagAgents Manager Pattern with Vision Browser CodeAgent

import os
import uuid
import re
import backoff
import requests
import tempfile
from typing import Any, TypedDict, Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from pathlib import Path
from contextvars import ContextVar
from PIL import Image
from io import BytesIO

# Core dependencies
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

import openai

# SmolagAgents imports
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    FinalAnswerTool,
    AgentLogger,
    LogLevel,
    PythonInterpreterTool,
    SpeechToTextTool
)

# Import retriever system
from dev_retriever import load_gaia_retriever

# Import logging
from agent_logging import AgentLoggingSetup

# Tools import statements
try:
    from tools.vision_browser_tool import VisionWebBrowserTool
    VISION_BROWSER_AVAILABLE = True
    print("✅ VisionWebBrowserTool imported")
except ImportError as e:
    print(f"⚠️ VisionWebBrowserTool not available: {e}")
    print("💡 Install dependencies: pip install helium selenium")
    VisionWebBrowserTool = None
    VISION_BROWSER_AVAILABLE = False

try:
    from tools.content_retriever_tool import (ContentRetrieverTool)
    CUSTOM_TOOLS_AVAILABLE = True
    print("✅ ContentRetrieverTool imported")
except ImportError as e:
    print(f"⚠️  ContentRetrieverTool not available: {e}")
    CUSTOM_TOOLS_AVAILABLE = False

# Add this block for LangChain tools
try:
    from tools.langchain_tools import (
        get_langchain_tools,
        get_tool_status,
        search_web_serper,
        search_wikipedia,
        search_arxiv
    )
    LANGCHAIN_TOOLS_AVAILABLE = True
    print("✅ LangChain research tools imported")
    
    # Test tool availability immediately
    tool_status = get_tool_status()
    print(f"📊 LangChain tools status: {tool_status}")
    
except ImportError as e:
    print(f"⚠️  LangChain tools not available: {e}")
    LANGCHAIN_TOOLS_AVAILABLE = False
except Exception as e:
    print(f"⚠️  LangChain tools error: {e}")
    LANGCHAIN_TOOLS_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GAIAConfig:
    """Configuration for GAIA agent"""
    model_provider: str = "openrouter"
    model_name: str = "google/gemini-2.5-flash"
    temperature: float = 0.3
    agent_evaluation_api: str = "https://agents-course-unit4-scoring.hf.space/"
    enable_url_downloading: bool = True
    download_timeout: int = 30
    
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
# Context bridge for execution tracking
# ============================================================================
class ContextBridge:
    """Simple execution tracking integrated into GAIAAgent"""
    
    current_operation: ContextVar[Optional[str]] = ContextVar('current_operation', default=None)
    step_counter: ContextVar[int] = ContextVar('step_counter', default=0)
    execution_start: ContextVar[Optional[float]] = ContextVar('execution_start', default=None)
    active_task_id: ContextVar[Optional[str]] = ContextVar('active_task_id', default=None)
    last_error: ContextVar[Optional[str]] = ContextVar('last_error', default=None)
    
    @classmethod
    def start_task_execution(cls, task_id: str):
        """Start tracking task execution"""
        cls.active_task_id.set(task_id)
        cls.execution_start.set(time.time())
        cls.step_counter.set(0)
        cls.last_error.set(None)
        print(f"🚀 Started execution tracking: {task_id}")
    
    @classmethod
    def track_operation(cls, operation: str):
        """Track current operation"""
        cls.current_operation.set(operation)
        current_step = cls.step_counter.get(0)
        cls.step_counter.set(current_step + 1)
        print(f"📝 Step {current_step + 1}: {operation}")
    
    @classmethod
    def track_error(cls, error: str):
        """Track error occurrence"""
        cls.last_error.set(error)
        cls.track_operation(f"ERROR: {error}")
        print(f"❌ Error tracked: {error}")
    
    @classmethod
    def get_execution_metrics(cls) -> Dict:
        """Get current execution metrics"""
        start_time = cls.execution_start.get()
        return {
            "execution_time": time.time() - start_time if start_time else 0,
            "steps_executed": cls.step_counter.get(0),
            "current_operation": cls.current_operation.get(),
            "last_error": cls.last_error.get()
        }
    
    @classmethod
    def clear_tracking(cls):
        """Clear execution tracking"""
        task_id = cls.active_task_id.get()
        metrics = cls.get_execution_metrics()
        
        cls.active_task_id.set(None)
        cls.execution_start.set(None)
        cls.step_counter.set(0)
        cls.current_operation.set(None)
        cls.last_error.set(None)
        
        print(f"🏁 Execution complete: {task_id}, {metrics['steps_executed']} steps, {metrics['execution_time']:.2f}s")

def track(operation: str, config):
    """Simple tracking helper"""
    if config and config.enable_context_bridge:
        try:
            ContextBridge.track_operation(operation)
        except Exception as e:
            if hasattr(config, 'debug_mode') and config.debug_mode:
                print(f"⚠️  Tracking error: {e}")
            pass  

# ============================================================================
# LLM RETRY LOGIC
# ============================================================================

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, max_tries=3)
def llm_invoke_with_retry(llm, messages):
    """Retry logic for LangChain LLM calls"""
    return llm.invoke(messages)

# ============================================================================
# FILE HANDLING UTILITY FUNCTIONS
# ============================================================================
    
def extract_file_info_from_task_id(task_id: str) -> Dict[str, str]:
    """
    Extract file information from task_id using GAIA dataset.
    """
    if not task_id:
        return {"file_name": "", "file_path": "", "has_file": False}
    
    try:
        # Try to get file info from GAIA dataset
        try:
            from gaia_dataset_utils import GAIADatasetManager
            dataset_manager = GAIADatasetManager("./tests/gaia_data")
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
        except ImportError:
            print("⚠️  GAIADatasetManager not available, using fallback")
        except Exception as e:
            print(f"⚠️  GAIA dataset error: {e}, using fallback")
        
        # Fallback: No file for arbitrary task_ids
        return {"file_name": "", "file_path": "", "has_file": False}
            
    except Exception as e:
        print(f"⚠️  Could not extract file info for task {task_id}: {e}")
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
    """GAIA Agent with vision capabilities implemented in a SmolagAgents 
    coordinator pattern embedded in LangGraph workflow"""
    
    def __init__(self, config: Optional[GAIAConfig] = None) -> None:
        if config is None:
            config = GAIAConfig()
        
        self.config: GAIAConfig = config
        self.capabilities: Dict[str, Any] = {}  # Store capability assessment
        
        print("🔄 Initializing core components...")
        self.retriever: Any = self._initialize_retriever()
        self.orchestration_model: Any = self._initialize_orchestration_model()  # LangChain for workflow
        self.specialist_model: LiteLLMModel = self._initialize_specialist_model()        # LiteLLM for SmolagAgents
        
        # Setup enhanced logging
        if config.enable_csv_logging:
            self.logging: Optional[AgentLoggingSetup] = AgentLoggingSetup(
                debug_mode=config.debug_mode,
                step_log_file=config.step_log_file,
                question_log_file=config.question_log_file
            )
        else:
            self.logging: Optional[AgentLoggingSetup] = None
        
        # Validating vision capabilities
        print("🔍 Validating SmolagAgent components...")
        
        try:
            # Test specialist creation
            print("  → Testing specialist agents creation...")
            test_specialists = self._create_specialist_agents()
            print(f"  ✅ Created {len(test_specialists)} specialists successfully")
            print(f"     Specialists: {list(test_specialists.keys())}")
            
            # Validate capabilities including vision browser agent
            print("  → Validating capabilities...")
            self.capabilities = self._validate_agent_capabilities(test_specialists)
            
            # Test coordinator creation
            print("  → Testing coordinator creation...")
            test_coordinator = self._create_coordinator()
            print("  ✅ Coordinator created successfully")
            print(f"     Managed agents: {len(test_coordinator.managed_agents) if hasattr(test_coordinator, 'managed_agents') else 'Unknown'}")
            
            # Test coordinator task execution
            print("  → Testing coordinator execution...")
            simple_test_result = test_coordinator.run("What is 2+2? Answer with just the number.")
            print(f"  ✅ Coordinator execution test successful")
            print(f"     Result preview: {str(simple_test_result)[:50]}...")
            
            # Store validated components for later use
            self._validated_specialists = test_specialists
            self._validated_coordinator = test_coordinator
            
        except Exception as e:
            print(f"\n❌ SMOLAGENT VALIDATION FAILED!")
            print(f"   Error: {str(e)}")
            print(f"   Type: {type(e).__name__}")
            print(f"   Module: {getattr(e, '__module__', 'Unknown')}")
            
            # Additional debugging info
            import traceback
            print(f"\n🔍 Full traceback:")
            traceback.print_exc()
            
            print(f"\n💡 This error occurred during SmolagAgent initialization.")
            print(f"   Common causes:")
            print(f"   - SmolagAgent API changes (deprecated parameters)")
            print(f"   - Missing dependencies or version mismatches")
            print(f"   - Configuration parameter conflicts")
            print(f"   - Model initialization issues")
            
            # Re-raise with clear context
            raise RuntimeError(f"SmolagAgent initialization failed during {e.__class__.__name__}: {str(e)}") from e
        
        print("✅ All SmolagAgent components validated successfully!")
        
        # Report final capability summary
        vision_status = "✅ Full Vision" if self.capabilities.get("vision_browser_ready") else "⚠️ Vision Limited"
        print(f"🎯 GAIA Agent Ready - Vision Capability: {vision_status}")
        
        # Initialize coordinator as None (will be created fresh for each task)
        self.coordinator: Optional[CodeAgent] = None
        
        # Build LangGraph workflow
        print("🔄 Building LangGraph workflow...")
        self.workflow: Any = self._build_workflow()
        
        print("🚀 GAIA Agent initialization complete!")
        print("   → SmolagAgent validation: ✅ PASSED")
        print("   → Vision capabilities: " + vision_status)
        print("   → LangGraph workflow: ✅ BUILT") 
        print("   → Ready for question processing")
            
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
        """Initialize orchestration model (LangChain)"""
        try:
            # Initialize orchestration model (LangChain)
            if self.config.model_provider == "openrouter":
                orchestration_model = ChatOpenAI(
                    model=self.config.model_name,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=self.config.temperature
                )
            elif self.config.model_provider == "anthropic":
                orchestration_model = ChatAnthropic(
                    model=self.config.model_name,
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    base_url="https://api.anthropic.com",
                    temperature=self.config.temperature
                )
            elif self.config.model_provider == "groq":
                orchestration_model = ChatGroq(
                    model=self.config.model_name,
                    google_api_key=os.getenv("GROQ_API_KEY"),
                    base_url="https://api.groq.com",
                    temperature=self.config.temperature
                )
            elif self.config.model_provider == "ollama":
                from langchain_ollama import ChatOllama
                orchestration_model = ChatOllama(
                    model=self.config.model_name,
                    base_url=self.config.api_base or "http://localhost:11434",
                    temperature=self.config.temperature
                )
            else:
                # Fallback to Openrouter
                orchestration_model = ChatOpenAI(
                    model="qwen/qwen3-32b",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=self.config.temperature
                )
            
            print(f"✅ Orchestration model: {getattr(orchestration_model, 'model_name', getattr(orchestration_model, 'model', 'Unknown'))}")
            return orchestration_model
            
        except Exception as e:
            print(f"❌ Error initializing orchestration model: {e}")
            raise

    def _initialize_specialist_model(self) -> LiteLLMModel:
        """Initialize LiteLLM model for SmolagAgents"""
        try:
            if self.config.model_provider == "openrouter":
                model_id = f"openrouter/{self.config.model_name}"
                api_key = os.getenv("OPENROUTER_API_KEY")
            elif self.config.model_provider == "anthropic":
                model_id = f"anthropic/{self.config.model_name}"
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.config.model_provider == "ollama":
                model_id = f"ollama_chat/{self.config.model_name}"
                return LiteLLMModel(
                    model_id=model_id,
                    api_base=self.config.api_base or "http://localhost:11434",
                    num_ctx=self.config.num_ctx,
                    temperature=self.config.temperature)
            elif self.config.model_provider == "groq":
                model_id = f"groq/{self.config.model_name}"
                api_key = os.getenv("GROQ_API_KEY")
            else:
                # Fallback to OpenRouter
                model_id = "openrouter/qwen/qwen-2.5-coder-32b-instruct:free"
                api_key = os.getenv("OPENROUTER_API_KEY")
            
            specialist_model = LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                temperature=self.config.temperature
            )
            
            print(f"✅ Specialist model: {model_id}")
            return specialist_model
            
        except Exception as e:
            print(f"❌ Error initializing specialist model: {e}")
            raise
        
    # ============================================================================
    # FILE HANDLING AND VISION CAPABILITIES USING SMOLAGENTS
    # ============================================================================

    def _check_model_vision_capabilities(self) -> Dict[str, bool]:
        """Check if the current model supports vision capabilities"""
        model_info = {
            "has_vision": False,
            "supports_images": False,
            "model_type": "unknown",
            "confidence": "low"
        }
        
        try:
            # Get model identifier from specialist model
            model_id = getattr(self.specialist_model, 'model_id', '').lower()
            
            # Known vision-capable model patterns
            vision_model_patterns = {
                # OpenAI models
                'gpt-4-vision': True,
                'gpt-4o': True,
                'gpt-4-turbo': True,
                
                # Anthropic models
                'claude-sonnet-4': True,  
                'claude-sonnet-3.7': True,
                'claude-sonnet-3.5': True,
                
                # Google models
                'gemini-pro-vision': True,
                'gemini-1.5': True,
                'gemini-2': True,
                'gemini-2.5': True,
                
                # Open source models
                'llava': True,
                'qwen-vl': True,
                'instructblip': True,
                'blip': True,
                
                # Known text-only models
                'gpt-3.5': False,
                'text-davinci': False,
                'qwen-qwq': False,  # Current model in config
                'llama': False,
                'mistral': False,
                'mixtral': False
            }
            
            # Check against known patterns
            for pattern, has_vision in vision_model_patterns.items():
                if pattern in model_id:
                    model_info.update({
                        "has_vision": has_vision,
                        "supports_images": has_vision,
                        "model_type": model_id,
                        "confidence": "high"
                    })
                    break
            else:
                # Unknown model - make conservative assumption
                model_info.update({
                    "has_vision": False,
                    "supports_images": False,
                    "model_type": model_id,
                    "confidence": "low"
                })
                
            # Additional checks for provider-specific patterns
            if 'openrouter/' in model_id:
                # OpenRouter - check the actual model name after the slash
                actual_model = model_id.split('/')[-1]
                for pattern, has_vision in vision_model_patterns.items():
                    if pattern in actual_model:
                        model_info["has_vision"] = has_vision
                        model_info["supports_images"] = has_vision
                        model_info["confidence"] = "high"
                        break
            
        except Exception as e:
            print(f"⚠️ Could not determine model vision capabilities: {e}")
            model_info.update({
                "has_vision": False,
                "supports_images": False,
                "model_type": "error_detecting",
                "confidence": "error"
            })
        
        return model_info

    def _validate_agent_capabilities(self, specialist_agents):
        """Validate capabilities of all specialist agents"""
        
        # Check model capabilities
        model_capabilities = self._check_model_vision_capabilities()
        
        # Check agent availability
        has_vision_browser = 'vision_browser_agent' in specialist_agents
        has_content_processor = 'content_processor' in specialist_agents
        has_data_analyst = 'data_analyst' in specialist_agents
        has_web_researcher = 'web_researcher' in specialist_agents
        
        # Check tool availability
        vision_browser_tool_available = VISION_BROWSER_AVAILABLE
        content_tools_available = CUSTOM_TOOLS_AVAILABLE
        
        capabilities = {
            # Agent availability
            "data_analyst_ready": has_data_analyst,
            "web_researcher_ready": has_web_researcher,
            "content_processor_ready": has_content_processor,
            "vision_browser_ready": has_vision_browser and vision_browser_tool_available,
            
            # Model capabilities
            "model_supports_vision": model_capabilities["has_vision"],
            "model_type": model_capabilities["model_type"],
            
            # Combined capabilities
            "full_vision_capability": (has_vision_browser and 
                                     vision_browser_tool_available and 
                                     model_capabilities["has_vision"]),
            "text_processing_ready": has_content_processor and content_tools_available,
            "data_processing_ready": has_data_analyst,
            "web_search_ready": has_web_researcher,
            
            # Tool status
            "vision_browser_tool_available": vision_browser_tool_available,
            "content_tools_available": content_tools_available,
            
            # Agent count
            "total_agents": len(specialist_agents)
        }
        
        # Status report
        print(f"📊 Agent Capabilities Assessment:")
        print(f"   Data Analyst: {'✅' if capabilities['data_analyst_ready'] else '❌'}")
        print(f"   Web Researcher: {'✅' if capabilities['web_researcher_ready'] else '❌'}")
        print(f"   Content Processor: {'✅' if capabilities['content_processor_ready'] else '❌'}")
        print(f"   Vision Browser: {'✅' if capabilities['vision_browser_ready'] else '❌'}")
        print(f"   Model Vision: {'✅' if capabilities['model_supports_vision'] else '❌'}")
        print(f"   Full Vision: {'✅' if capabilities['full_vision_capability'] else '⚠️'}")
        
        return capabilities

    def _analyze_image_metadata(self, file_path: str) -> Dict[str, Any]:
        """Analyze image file for enhanced vision processing"""
        if not file_path or not os.path.exists(file_path):
            return {"error": "Image file not found"}
        
        try:
            from PIL import Image
            import mimetypes
            
            with Image.open(file_path) as img:
                image_info = {
                    "dimensions": img.size,
                    "format": img.format,
                    "mode": img.mode,
                    "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info,
                    "file_size": os.path.getsize(file_path),
                    "mime_type": mimetypes.guess_type(file_path)[0],
                    "estimated_complexity": "high" if max(img.size) > 2000 else "medium"
                }
                
                # Detect potential content types
                width, height = img.size
                aspect_ratio = width / height
                
                if aspect_ratio > 2.0:
                    image_info["likely_content"] = "panoramic_or_document"
                elif 0.8 <= aspect_ratio <= 1.2:
                    image_info["likely_content"] = "square_diagram_or_chart"
                elif width > 1000 and height > 800:
                    image_info["likely_content"] = "detailed_image_or_screenshot"
                else:
                    image_info["likely_content"] = "standard_image"
                
                return image_info
                
        except Exception as e:
            return {"error": f"Image analysis failed: {str(e)}"}

    def _analyze_file_metadata(self, file_name: str, file_path: str) -> Dict[str, Any]:
        """Simple file analysis without agent_files dependency"""
        if not file_name:
            return {"no_file": True}
        
        try:
            # Choose best available file path
            if file_name and file_name.strip() and os.path.exists(file_name):
                best_path = file_name
            elif file_path and file_path.strip() and os.path.exists(file_path):
                best_path = file_path
            else:
                best_path = file_name or file_path
            
            # Basic file info
            path_obj = Path(file_name) if file_name else Path(file_path)
            extension = path_obj.suffix.lower() if path_obj.suffix else ""
            
            # File existence and size
            file_exists = os.path.exists(best_path) if best_path else False
            file_size = os.path.getsize(best_path) if file_exists else 0
            
            # MIME type
            import mimetypes
            mime_type = mimetypes.guess_type(file_name or file_path)[0] or 'application/octet-stream'
            
            # Enhanced categorization with vision browser routing
            if extension in ['.xlsx', '.csv', '.xls', '.tsv']:
                category, recommended_specialist = "data", "data_analyst"
            elif extension in ['.pdf', '.docx', '.doc', '.txt', '.rtf']:
                category, recommended_specialist = "document", "content_processor"
            elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                # NEW: Route images to vision browser for visual analysis
                category, recommended_specialist = "image", "vision_browser_agent"
            elif extension in ['.mp3', '.mp4', '.wav', '.m4a', '.mov', '.avi']:
                category, recommended_specialist = "media", "content_processor"
            elif extension in ['.py', '.js', '.sql', '.json', '.xml', '.yaml', '.yml']:
                category, recommended_specialist = "code", "content_processor"
            elif extension in ['.zip', '.tar', '.gz', '.rar']:
                category, recommended_specialist = "archive", "content_processor"
            else:
                category, recommended_specialist = "unknown", "content_processor"
            
            return {
                "file_name": file_name,
                "file_path": file_path,
                "best_path": best_path,
                "extension": extension,
                "mime_type": mime_type,
                "category": category,
                "recommended_specialist": recommended_specialist,
                "file_exists": file_exists,
                "file_size": file_size,
                "estimated_complexity": "high" if file_size > 10_000_000 else "medium"
            }
            
        except Exception as e:
            # Error fallback
            return {
                "file_name": file_name,
                "file_path": file_path,
                "best_path": file_path or file_name,
                "category": "unknown",
                "recommended_specialist": "content_processor",
                "error": str(e),
                "file_exists": False,
                "estimated_complexity": "medium"
            }

    def _load_image_for_agent(self, file_path: str) -> Optional[Image.Image]:
        """Load image file for SmolagAgent processing"""
        try:
            if not file_path or not os.path.exists(file_path):
                print(f"❌ Image file not found: {file_path}")
                return None
                
            image = Image.open(file_path).convert("RGB")
            print(f"✅ Image loaded: {file_path}")
            return image
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
        
    def _download_file_once(self, task_id: str, file_name: str) -> str:
        """Download file from GAIA API endpoint"""
        try:
            file_url = f"{self.config.agent_evaluation_api}/files/{task_id}"
            response = requests.get(file_url)
            response.raise_for_status()
            
            # Save with proper extension from file_name
            local_path = f"/tmp/{file_name}"  # Preserves .png extension
            with open(local_path, 'wb') as f:
                f.write(response.content)
                
            return local_path
        except Exception as e:
            print(f"❌ File download failed: {e}")
            return None

    # ============================================================================
    # SMOLAGENTS COORDINATOR AND SPECIALISTS - ENHANCED WITH VISION BROWSER
    # ============================================================================        

    def _create_specialist_agents(self) -> Dict[str, Any]:
        """Create specialist agents using smolagents pattern with separate vision browser agent"""
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        
        specialist_agents = {}
        
        # 1. Data Analyst - CodeAgent for direct Python file access
        try:
            data_analyst = CodeAgent(
                name="data_analyst", 
                description="Excel/CSV analysis and numerical calculations using pandas",
                tools=[],
                additional_authorized_imports=[
                    "pandas", "numpy", "openpyxl", "xlrd", "csv",
                    "scipy", "matplotlib", "seaborn", "statistics", 
                    "math", "json"
                ],
                model=self.specialist_model,
                max_steps=self.config.max_agent_steps,
                add_base_tools=True,
                logger=logger
            )
            specialist_agents["data_analyst"] = data_analyst
            print(f"✅ Data Analyst: {len(data_analyst.additional_authorized_imports)} imports")
        except Exception as e:
            print(f"❌ Failed to create data_analyst: {e}")
        
        # 2. Web Researcher - ToolCallingAgent for text-based search
        try:
            web_tools = []
            try:
                from tools import get_web_researcher_tools
                web_tools = get_web_researcher_tools()
            except ImportError:
                print("⚠️ tools module unavailable - using fallback web tools")
                try:
                    from smolagents import VisitWebpageTool, WikipediaSearchTool
                    web_tools.extend([VisitWebpageTool(), WikipediaSearchTool()])
                except Exception as e:
                    print(f"⚠️ Fallback web tools failed: {e}")

            web_researcher = ToolCallingAgent(
                name="web_researcher",
                description="Search web, find information sources, verify accessibility (text-based)",
                tools=web_tools,
                model=self.specialist_model,
                max_steps=self.config.max_agent_steps,
                add_base_tools=len(web_tools) == 0,
                logger=logger
            )
            specialist_agents["web_researcher"] = web_researcher
            print(f"✅ Web Researcher: {len(web_tools)} tools")
        except Exception as e:
            print(f"❌ Failed to create web_researcher: {e}")
        
        # 3. Content Processor - ToolCallingAgent for document/media processing (NO vision browser)
        try:
            content_tools = []
            try:
                from tools import get_content_processor_tools
                content_tools = get_content_processor_tools()
                # Filter out vision browser tool if present
                content_tools = [tool for tool in content_tools 
                               if not hasattr(tool, '__class__') or 
                               'vision' not in tool.__class__.__name__.lower()]
            except ImportError:
                print("⚠️ tools module unavailable - using base content tools")

            content_processor = ToolCallingAgent(
                name="content_processor", 
                description="Extract and process content from documents, audio, video, and text sources",
                tools=content_tools,
                model=self.specialist_model,
                max_steps=self.config.max_agent_steps,
                add_base_tools=True,
                logger=logger
            )
            specialist_agents["content_processor"] = content_processor
            print(f"✅ Content Processor: {len(content_tools)} tools (vision-free)")
        except Exception as e:
            print(f"❌ Failed to create content_processor: {e}")

        # 4. NEW: Vision Browser Agent - CodeAgent for visual web navigation
        try:
            if VISION_BROWSER_AVAILABLE:
                vision_browser_agent = CodeAgent(
                    name="vision_browser_agent",
                    description="""Visual web navigation and screenshot analysis specialist.
                    
Handles:
- Visual web browsing with screenshots
- Image analysis and OCR
- Interactive web navigation
- Screenshot-based verification
- Visual content extraction

Uses browser automation tools with direct code execution for maximum control.""",
                    tools=[],  # Will use browser tools via imports
                    additional_authorized_imports=[
                        # Browser automation
                        "selenium", "helium", 
                        
                        # Image processing
                        "PIL", "io", "base64", "requests",
                        
                        # Browser setup and configuration
                        "webdriver_manager", "chromedriver_autoinstaller",
                        
                        # File and system operations
                        "os", "sys", "time", "tempfile", "pathlib",
                        
                        # Error handling and retries
                        "contextlib", "functools", "traceback"
                    ],
                    model=self.specialist_model,
                    max_steps=self.config.max_agent_steps,
                    add_base_tools=True,  # Include base tools for completeness
                    logger=logger
                )
                specialist_agents["vision_browser_agent"] = vision_browser_agent
                print(f"✅ Vision Browser Agent: {len(vision_browser_agent.additional_authorized_imports)} imports")
            else:
                print("⚠️ Vision Browser Agent: Not available (missing dependencies)")
        except Exception as e:
            print(f"❌ Failed to create vision_browser_agent: {e}")

        print(f"🎯 Created {len(specialist_agents)} specialist agents total")
        return specialist_agents
        
    def _create_coordinator(self) -> CodeAgent:
        """Create coordinator with enhanced routing for vision browser agent"""
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        
        # Define specialists
        specialist_agents = self._create_specialist_agents()
        
        # Enhanced coordinator with intelligent agent routing
        coordinator = CodeAgent(
            name="gaia_coordinator",
            description="""GAIA task coordinator managing four specialist agents with intelligent routing.

WORKFLOW:
1. ANALYZE TASK: Question analysis and file preprocessing 
2. ROUTE INTELLIGENTLY: Choose appropriate specialist(s) based on task requirements
3. DELEGATE: Assign task to specialist(s) with proper context
4. SYNTHESIZE: Combine results and format GAIA answer

SPECIALIST ROUTING STRATEGY:

DATA ANALYST (CodeAgent):
- Numerical calculations, statistics, math problems
- CSV/Excel file analysis with pandas
- Data processing and aggregation
- Financial calculations, percentages

WEB RESEARCHER (ToolCallingAgent):
- Text-based web searches
- Information discovery and fact-checking
- API-based content retrieval
- Non-visual web content access

CONTENT PROCESSOR (ToolCallingAgent):
- Document processing (PDF, DOCX, TXT)
- Audio/video transcription and analysis
- Text extraction from files
- Archive extraction and processing

VISION BROWSER AGENT (CodeAgent):
- Visual web navigation with screenshots
- Image analysis and OCR tasks
- Interactive web elements requiring visual confirmation
- Screenshot-based verification
- Complex visual content extraction
- Browser automation requiring precise control

ROUTING DECISION MATRIX:
- Image files (.png, .jpg, etc.) → Vision Browser Agent
- Data files (.csv, .xlsx) → Data Analyst
- Documents (.pdf, .docx) → Content Processor
- Web search needs → Web Researcher
- Visual web tasks → Vision Browser Agent
- Mathematical calculations → Data Analyst

The coordinator handles file preparation and intelligent agent selection.""",
            tools=[],  # No tools needed - direct file access via imports
            managed_agents=list(specialist_agents.values()),
            additional_authorized_imports=[
                # File preprocessing (CORE RESPONSIBILITY)
                "zipfile", "tarfile", "gzip", "bz2",  # Archive handling
                
                # File operations and inspection
                "open", "codecs", "chardet", "os", "sys", "io", "pathlib",
                "mimetypes", "tempfile", "shutil",    # File management
                
                # Data preprocessing for specialist preparation
                "pandas", "numpy", "csv", "json", "xml", "base64",
                
                # Binary and media file inspection
                "PIL", "wave", "struct", "binascii",  # Basic media inspection

                # Smart file handling functions
                "smart_file_handler", "is_url",
                
                # Web and networking
                "requests", "urllib", "time", "typing"
            ],
            model=self.specialist_model,
            planning_interval=5,  # More frequent planning for intelligent routing
            max_steps=self.config.max_agent_steps,  # Sufficient steps for coordination + delegation
            logger=logger
        )
        
        return coordinator

    def _build_coordinator_task(self, state: GAIAState) -> str:
        """Build coordination task with enhanced vision browser routing"""
        question = state["question"]
        similar_examples = state.get("similar_examples", [])
        complexity = state.get("complexity", "unknown")
        has_file = state.get("has_file", False)
        file_name = state.get("file_name", "")
        file_path = state.get("file_path", "")
        file_metadata = state.get("file_metadata", {})
        
        # Enhanced file context with vision browser routing
        file_context = ""
        routing_guidance = ""
        
        if has_file:
            # Determine best file reference and provide processing guidance
            if file_name and file_name.strip() and os.path.exists(file_name):
                file_context = f"\nFILE AVAILABLE: {file_name}"
                if file_metadata:
                    file_type = file_metadata.get("category", "unknown")
                    recommended_specialist = file_metadata.get("recommended_specialist", "content_processor")
                    file_context += f"\nFILE TYPE: {file_type}"
                    file_context += f"\nRECOMMENDED PROCESSING: {recommended_specialist}"
                    
                    # Enhanced routing guidance including vision browser
                    if file_type == "data":
                        routing_guidance = "\nROUTING: Use data_analyst for CSV/Excel analysis with pandas"
                    elif file_type == "document":
                        routing_guidance = "\nROUTING: Use content_processor with ContentRetrieverTool"
                    elif file_type == "image":
                        routing_guidance = "\nROUTING: Use vision_browser_agent for visual analysis and OCR"
                    elif file_type == "media":
                        routing_guidance = "\nROUTING: Use content_processor for audio/video transcription"
                    else:
                        routing_guidance = "\nROUTING: Analyze file type and choose appropriate specialist"
                
            elif file_path and file_path.strip() and os.path.exists(file_path):
                file_context = f"\nFILE AVAILABLE: {file_path}"
                if file_name:
                    file_context += f" (original name: {file_name})"
                routing_guidance = "\nROUTING: Inspect file extension and route to appropriate specialist"
                
            else:
                file_context = "\nFILE: Referenced but not accessible - proceed without file"
                routing_guidance = "\nROUTING: Text-based approach, likely web_researcher or direct reasoning"
        
        # Question analysis for routing hints
        question_lower = question.lower()
        additional_routing_hints = ""
        
        if any(term in question_lower for term in ["calculate", "percent", "average", "sum", "count", "data"]):
            additional_routing_hints += "\nHINT: Mathematical/data analysis → Consider data_analyst"
        
        if any(term in question_lower for term in ["search", "find", "look up", "current", "latest"]):
            additional_routing_hints += "\nHINT: Information discovery → Consider web_researcher"
            
        if any(term in question_lower for term in ["image", "picture", "screenshot", "visual", "see", "show"]):
            additional_routing_hints += "\nHINT: Visual analysis → Consider vision_browser_agent"
            
        if any(term in question_lower for term in ["document", "pdf", "text", "extract"]):
            additional_routing_hints += "\nHINT: Document processing → Consider content_processor"
        
        # Build examples context
        examples_context = ""
        if similar_examples:
            examples_context = "\nSIMILAR SUCCESSFUL EXAMPLES:\n"
            for i, example in enumerate(similar_examples[:2], 1):
                examples_context += f"{i}. {example.get('question', '')[:80]}... → {example.get('answer', '')}\n"
        
        # Core coordination task with enhanced routing
        task = f"""GAIA Coordinator Task - Intelligent Agent Routing

QUESTION: {question}
COMPLEXITY: {complexity}{file_context}{routing_guidance}{additional_routing_hints}{examples_context}

AVAILABLE SPECIALISTS:

1. DATA_ANALYST (CodeAgent):
   - Capabilities: pandas, numpy, mathematical calculations
   - Best for: CSV/Excel analysis, numerical computations, statistics
   - File handling: Direct pandas file reading
   
2. WEB_RESEARCHER (ToolCallingAgent):
   - Capabilities: Web search, information discovery
   - Best for: Finding current information, fact-checking
   - File handling: Not applicable
   
3. CONTENT_PROCESSOR (ToolCallingAgent):
   - Capabilities: Document processing, audio/video transcription
   - Best for: PDF/DOCX extraction, media file processing
   - File handling: ContentRetrieverTool with file_path
   
4. VISION_BROWSER_AGENT (CodeAgent):
   - Capabilities: Visual web navigation, image analysis, OCR
   - Best for: Image files, visual web content, screenshot analysis
   - File handling: Direct image loading with PIL

COORDINATOR WORKFLOW:

1. ANALYZE REQUIREMENTS:
   - Question type (calculation, search, visual, document)
   - File type and processing needs
   - Best specialist match

2. INTELLIGENT ROUTING:
   - Single specialist for focused tasks
   - Multiple specialists for complex multi-step tasks
   - Proper delegation with context

3. TASK DELEGATION:
   - Provide clear instructions to chosen specialist(s)
   - Include file_path when applicable
   - Set expectations for output format

4. RESULT SYNTHESIS:
   - Combine specialist outputs if multiple agents used
   - Apply GAIA formatting requirements
   - Ensure answer precision

DELEGATION EXAMPLES:
- For data_analyst: "Analyze the CSV file using pandas and calculate..."
- For web_researcher: "Search for current information about..."
- For content_processor: "Extract content from the file at file_path..."
- For vision_browser_agent: "Analyze the image and identify..."

CRITICAL REQUIREMENTS:
- End with: FINAL ANSWER: [specific answer]
- Format: numbers (no commas), strings (no articles), lists (comma-separated)
- Be precise and factual
- Choose the most appropriate specialist for the task

Execute intelligent routing and delegation now."""

        return task.strip()
                   
    # ============================================================================
    # LANGGRAPH WORKFLOW - UNCHANGED
    # ============================================================================

    def _build_workflow(self):
        """Build LangGraph workflow for question answering"""
        builder = StateGraph(GAIAState)
        
        if self.config.enable_smart_routing:
            # Enhanced smart routing workflow with optimized coordinator
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("complexity_check", self._complexity_check_node)
            builder.add_node("one_shot_answering", self._one_shot_answering_node)
            builder.add_node("coordinator", self._coordinator_node)
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
            
            builder.add_edge("coordinator", "format_answer")
            
            # Simple workflow: one-shot → format
            builder.add_edge("one_shot_answering", "format_answer")
            
            # Both paths converge at formatting
            builder.add_edge("format_answer", END)
            
        else:
            # Linear workflow with optimized coordinator
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("coordinator", self._coordinator_node)
            builder.add_node("format_answer", self._format_answer_node)
            
            # Direct flow - read → coordinate → format
            builder.add_edge(START, "read_question")
            builder.add_edge("read_question", "coordinator")
            builder.add_edge("coordinator", "format_answer")
            builder.add_edge("format_answer", END)
        
        return builder.compile()

    def _read_question_node(self, state: GAIAState) -> GAIAState:
        """Question reading with file info extraction and download"""
        task_id = state.get("task_id")
        question = state["question"]
        
        # Start context bridge tracking
        if self.config.enable_context_bridge:
            ContextBridge.start_task_execution(task_id)
            track("Processing question and extracting file info", self.config)
        
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
        """🎯 SIMPLIFIED: File check + LLM assessment only"""
        question = state["question"]
        has_file = state.get("has_file", False)
        
        if self.logging:
            self.logging.log_step("complexity_analysis", f"Analyzing complexity: {question[:50]}...")
        
        print(f"🧠 Analyzing complexity: {question[:50]}...")
        
        # Only two rules:
        if has_file:
            complexity = "complex"
            reason = "File attachment detected"
        else:
            complexity = self._llm_complexity_check(question)
            reason = "LLM multi-step assessment"
        
        print(f"📊 Complexity: {complexity} ({reason})")
        
        if self.logging:
            self.logging.set_complexity(complexity)
            self.logging.log_step("complexity_result", f"Final complexity: {complexity} - {reason}")
        
        # RAG for complex questions if needed
        similar_examples = state.get("similar_examples", [])
        if complexity == "complex" and self.config.skip_rag_for_simple and not similar_examples:
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
                
                if self.logging:
                    self.logging.set_similar_examples_count(len(similar_examples))
                    
            except Exception as e:
                print(f"⚠️ RAG retrieval error: {e}")
        
        return {
            "complexity": complexity,
            "similar_examples": similar_examples,
            "steps": state["steps"] + [f"Complexity assessed: {complexity} ({reason})"]
        }

    def _llm_complexity_check(self, question: str) -> str:
        """
        Does answering this question require external tools or just training knowledge?
        """
        
        prompt = f"""Question: {question}

To answer this question, do I need to search the web, access files, or use external tools?
Or can I answer it directly from my training knowledge alone?

SIMPLE = I can answer from my training knowledge alone
COMPLEX = I need to search the web, access files, or use tools to find current/specific information

Answer just: SIMPLE or COMPLEX"""
        
        try:
            response = llm_invoke_with_retry(self.orchestration_model, [HumanMessage(content=prompt)])
            result = response.content.strip().upper()
            
            if "SIMPLE" in result:
                return "simple"
            else:
                return "complex"
                
        except Exception as e:
            print(f"⚠️ LLM complexity check failed: {str(e)}, defaulting to complex")
            return "complex"

    def _route_by_complexity(self, state: GAIAState) -> str:
        """Routing function for conditional edges"""
        return state.get("complexity", "complex")

    def _one_shot_answering_node(self, state: GAIAState):
        """Direct LLM answering for simple questions"""
        task_id = state.get("task_id", "")
        
        print("⚡ Using one-shot direct answering")
        
        track("Direct LLM answering for simple question", self.config)
        
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
            },
            r'(?i)final\s*answer\s*:\s*(.+?)(?:\n|$)',
            r'(?i).*final\s*answer\s*:\s*(.+)',
        ]
        
        for pattern in final_answer_patterns:
            match = safe_regex_search(pattern, answer, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    answer = extracted
                    if hasattr(self, 'logging_setup') and self.logging_setup:
                        self.logging_setup.log_step(
                            action='gaia_format_extract',
                            details=f'Extracted: \'{answer}\'',
                            node_name='format_answer'
                        )
                    break
        
        # SAFE: Remove prefixes
        prefixes_to_remove = [
            "final answer:", "the answer is:", "answer:", "result:",
            "solution:", "the result is:", "therefore", "so", "thus"
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        # Clean quotes and punctuation
        answer = answer.strip('.,!?:;"\'')
        
        # Question-specific formatting
        question = getattr(self, '_current_question', '').lower()
        
        # SAFE: Standard GAIA formatting
        if not any(special in question for special in ["comma", "list", "countries"]):
            # Remove articles (a, an, the) from the beginning
            import re
            answer = re.sub(r'^(a|an|the)\s+', '', answer, flags=re.IGNORECASE)
        
        # Number formatting - remove commas from numbers
        if answer.replace(',', '').replace('.', '').replace('-', '').isdigit():
            answer = answer.replace(',', '')
        
        # Final cleanup
        answer = answer.strip()
        
        if hasattr(self, 'logging_setup') and self.logging_setup:
            self.logging_setup.log_step(
                action='gaia_format_final',
                details=f'Final formatted: \'{answer}\'',
                node_name='format_answer'
            )
        
        return answer if answer else "No answer"

    def process_question(self, question: str, task_id: str = None) -> Dict:
        """
        Main entry point with context bridge integration and enhanced vision browser support.
        
        Args:
            question: The question to process
            task_id: Optional task identifier
            
        Returns:
            Dictionary with complete processing results including metrics and vision browser usage
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
                "execution_metrics": execution_metrics,
                "vision_browser_used": result.get("vision_browser_used", False),
                "execution_strategy": result.get("execution_strategy", "unknown")
            }
            
            success = result.get("execution_successful", True)
            final_answer = result.get("final_answer", str(result)) if isinstance(result, dict) else str(result)
            
            # Enhanced logging with vision browser info
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
                "vision_browser_used": result.get("vision_browser_used", False),
                "execution_strategy": result.get("execution_strategy", "unknown"),
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
                "vision_browser_used": False,
                "execution_strategy": "error",
                "file_info": {"file_name": "", "file_path": "", "has_file": False}
            }

def safe_regex_search(pattern: str, text: str, flags=0):
    """Safe regex search with error handling"""
    if not isinstance(text, str):
        return None
        
    try:
        compiled_pattern = re.compile(pattern, flags)
        return compiled_pattern.search(text)
    except (re.error, MemoryError, Exception) as e:
        print(f"⚠️  Regex search error: {e}")
        return None

def safe_regex_sub(pattern: str, replacement: str, text: str, flags=0) -> str:
    """Safely apply regex substitution with error handling"""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    try:
        compiled_pattern = re.compile(pattern, flags)
        return compiled_pattern.sub(replacement, text)
    except (re.error, MemoryError, Exception) as e:
        print(f"⚠️  Regex error: {e}, returning original text")
        return text

def safe_regex_findall(pattern: str, text: str, flags=0) -> List[str]:
    """Safe regex findall with error handling"""
    if not isinstance(text, str):
        return []
        
    try:
        compiled_pattern = re.compile(pattern, flags)
        return compiled_pattern.findall(text)
    except (re.error, MemoryError, Exception) as e:
        print(f"⚠️  Regex findall error: {e}")
        return []


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 GAIA Agent - Enhanced with Vision Browser CodeAgent")
    print("=" * 70)
    print("✅ Four specialized agents: data_analyst, web_researcher, content_processor, vision_browser_agent")
    print("✅ Vision Browser Agent: CodeAgent with selenium/helium for visual web navigation")
    print("✅ Intelligent routing based on file type and task requirements")
    print("✅ Enhanced coordinator with vision browser support")
    print("✅ Container-ready browser automation with fallback strategies")
    print("")
    print("Use agent_testing.py for testing")# agent_logic.py
# GAIA Agent System using SmolagAgents Manager Pattern with Vision Browser CodeAgent

import os
import uuid
import re
import backoff
import requests
import tempfile
from typing import Any, TypedDict, Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from pathlib import Path
from contextvars import ContextVar
from PIL import Image
from io import BytesIO

# Core dependencies
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

import openai

# SmolagAgents imports
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    FinalAnswerTool,
    AgentLogger,
    LogLevel,
    PythonInterpreterTool,
    SpeechToTextTool
)

# Import retriever system
from dev_retriever import load_gaia_retriever

# Import logging
from agent_logging import AgentLoggingSetup

# Tools import statements
try:
    from tools.vision_browser_tool import VisionWebBrowserTool
    VISION_BROWSER_AVAILABLE = True
    print("✅ VisionWebBrowserTool imported")
except ImportError as e:
    print(f"⚠️ VisionWebBrowserTool not available: {e}")
    print("💡 Install dependencies: pip install helium selenium")
    VisionWebBrowserTool = None
    VISION_BROWSER_AVAILABLE = False

try:
    from tools.content_retriever_tool import (ContentRetrieverTool)
    CUSTOM_TOOLS_AVAILABLE = True
    print("✅ ContentRetrieverTool imported")
except ImportError as e:
    print(f"⚠️  ContentRetrieverTool not available: {e}")
    CUSTOM_TOOLS_AVAILABLE = False

# Add this block for LangChain tools
try:
    from tools.langchain_tools import (
        get_langchain_tools,
        get_tool_status,
        search_web_serper,
        search_wikipedia,
        search_arxiv
    )
    LANGCHAIN_TOOLS_AVAILABLE = True
    print("✅ LangChain research tools imported")
    
    # Test tool availability immediately
    tool_status = get_tool_status()
    print(f"📊 LangChain tools status: {tool_status}")
    
except ImportError as e:
    print(f"⚠️  LangChain tools not available: {e}")
    LANGCHAIN_TOOLS_AVAILABLE = False
except Exception as e:
    print(f"⚠️  LangChain tools error: {e}")
    LANGCHAIN_TOOLS_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GAIAConfig:
    """Configuration for GAIA agent"""
    model_provider: str = "openrouter"
    model_name: str = "google/gemini-2.5-flash"
    temperature: float = 0.3
    agent_evaluation_api: str = "https://agents-course-unit4-scoring.hf.space/"
    enable_url_downloading: bool = True
    download_timeout: int = 30
    
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
# Context bridge for execution tracking
# ============================================================================
class ContextBridge:
    """Simple execution tracking integrated into GAIAAgent"""
    
    current_operation: ContextVar[Optional[str]] = ContextVar('current_operation', default=None)
    step_counter: ContextVar[int] = ContextVar('step_counter', default=0)
    execution_start: ContextVar[Optional[float]] = ContextVar('execution_start', default=None)
    active_task_id: ContextVar[Optional[str]] = ContextVar('active_task_id', default=None)
    last_error: ContextVar[Optional[str]] = ContextVar('last_error', default=None)
    
    @classmethod
    def start_task_execution(cls, task_id: str):
        """Start tracking task execution"""
        cls.active_task_id.set(task_id)
        cls.execution_start.set(time.time())
        cls.step_counter.set(0)
        cls.last_error.set(None)
        print(f"🚀 Started execution tracking: {task_id}")
    
    @classmethod
    def track_operation(cls, operation: str):
        """Track current operation"""
        cls.current_operation.set(operation)
        current_step = cls.step_counter.get(0)
        cls.step_counter.set(current_step + 1)
        print(f"📝 Step {current_step + 1}: {operation}")
    
    @classmethod
    def track_error(cls, error: str):
        """Track error occurrence"""
        cls.last_error.set(error)
        cls.track_operation(f"ERROR: {error}")
        print(f"❌ Error tracked: {error}")
    
    @classmethod
    def get_execution_metrics(cls) -> Dict:
        """Get current execution metrics"""
        start_time = cls.execution_start.get()
        return {
            "execution_time": time.time() - start_time if start_time else 0,
            "steps_executed": cls.step_counter.get(0),
            "current_operation": cls.current_operation.get(),
            "last_error": cls.last_error.get()
        }
    
    @classmethod
    def clear_tracking(cls):
        """Clear execution tracking"""
        task_id = cls.active_task_id.get()
        metrics = cls.get_execution_metrics()
        
        cls.active_task_id.set(None)
        cls.execution_start.set(None)
        cls.step_counter.set(0)
        cls.current_operation.set(None)
        cls.last_error.set(None)
        
        print(f"🏁 Execution complete: {task_id}, {metrics['steps_executed']} steps, {metrics['execution_time']:.2f}s")

def track(operation: str, config):
    """Simple tracking helper"""
    if config and config.enable_context_bridge:
        try:
            ContextBridge.track_operation(operation)
        except Exception as e:
            if hasattr(config, 'debug_mode') and config.debug_mode:
                print(f"⚠️  Tracking error: {e}")
            pass  

# ============================================================================
# LLM RETRY LOGIC
# ============================================================================

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, max_tries=3)
def llm_invoke_with_retry(llm, messages):
    """Retry logic for LangChain LLM calls"""
    return llm.invoke(messages)

# ============================================================================
# FILE HANDLING UTILITY FUNCTIONS
# ============================================================================
    
def extract_file_info_from_task_id(task_id: str) -> Dict[str, str]:
    """
    Extract file information from task_id using GAIA dataset.
    """
    if not task_id:
        return {"file_name": "", "file_path": "", "has_file": False}
    
    try:
        # Try to get file info from GAIA dataset
        try:
            from gaia_dataset_utils import GAIADatasetManager
            dataset_manager = GAIADatasetManager("./tests/gaia_data")
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
        except ImportError:
            print("⚠️  GAIADatasetManager not available, using fallback")
        except Exception as e:
            print(f"⚠️  GAIA dataset error: {e}, using fallback")
        
        # Fallback: No file for arbitrary task_ids
        return {"file_name": "", "file_path": "", "has_file": False}
            
    except Exception as e:
        print(f"⚠️  Could not extract file info for task {task_id}: {e}")
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
    """GAIA Agent with vision capabilities implemented in a SmolagAgents 
    coordinator pattern embedded in LangGraph workflow"""
    
    def __init__(self, config: Optional[GAIAConfig] = None) -> None:
        if config is None:
            config = GAIAConfig()
        
        self.config: GAIAConfig = config
        self.capabilities: Dict[str, Any] = {}  # Store capability assessment
        
        print("🔄 Initializing core components...")
        self.retriever: Any = self._initialize_retriever()
        self.orchestration_model: Any = self._initialize_orchestration_model()  # LangChain for workflow
        self.specialist_model: LiteLLMModel = self._initialize_specialist_model()        # LiteLLM for SmolagAgents
        
        # Setup enhanced logging
        if config.enable_csv_logging:
            self.logging: Optional[AgentLoggingSetup] = AgentLoggingSetup(
                debug_mode=config.debug_mode,
                step_log_file=config.step_log_file,
                question_log_file=config.question_log_file
            )
        else:
            self.logging: Optional[AgentLoggingSetup] = None
        
        # Validating vision capabilities
        print("🔍 Validating SmolagAgent components...")
        
        try:
            # Test specialist creation
            print("  → Testing specialist agents creation...")
            test_specialists = self._create_specialist_agents()
            print(f"  ✅ Created {len(test_specialists)} specialists successfully")
            print(f"     Specialists: {list(test_specialists.keys())}")
            
            # Validate capabilities including vision browser agent
            print("  → Validating capabilities...")
            self.capabilities = self._validate_agent_capabilities(test_specialists)
            
            # Test coordinator creation
            print("  → Testing coordinator creation...")
            test_coordinator = self._create_coordinator()
            print("  ✅ Coordinator created successfully")
            print(f"     Managed agents: {len(test_coordinator.managed_agents) if hasattr(test_coordinator, 'managed_agents') else 'Unknown'}")
            
            # Test coordinator task execution
            print("  → Testing coordinator execution...")
            simple_test_result = test_coordinator.run("What is 2+2? Answer with just the number.")
            print(f"  ✅ Coordinator execution test successful")
            print(f"     Result preview: {str(simple_test_result)[:50]}...")
            
            # Store validated components for later use
            self._validated_specialists = test_specialists
            self._validated_coordinator = test_coordinator
            
        except Exception as e:
            print(f"\n❌ SMOLAGENT VALIDATION FAILED!")
            print(f"   Error: {str(e)}")
            print(f"   Type: {type(e).__name__}")
            print(f"   Module: {getattr(e, '__module__', 'Unknown')}")
            
            # Additional debugging info
            import traceback
            print(f"\n🔍 Full traceback:")
            traceback.print_exc()
            
            print(f"\n💡 This error occurred during SmolagAgent initialization.")
            print(f"   Common causes:")
            print(f"   - SmolagAgent API changes (deprecated parameters)")
            print(f"   - Missing dependencies or version mismatches")
            print(f"   - Configuration parameter conflicts")
            print(f"   - Model initialization issues")
            
            # Re-raise with clear context
            raise RuntimeError(f"SmolagAgent initialization failed during {e.__class__.__name__}: {str(e)}") from e
        
        print("✅ All SmolagAgent components validated successfully!")
        
        # Report final capability summary
        vision_status = "✅ Full Vision" if self.capabilities.get("vision_browser_ready") else "⚠️ Vision Limited"
        print(f"🎯 GAIA Agent Ready - Vision Capability: {vision_status}")
        
        # Initialize coordinator as None (will be created fresh for each task)
        self.coordinator: Optional[CodeAgent] = None
        
        # Build LangGraph workflow
        print("🔄 Building LangGraph workflow...")
        self.workflow: Any = self._build_workflow()
        
        print("🚀 GAIA Agent initialization complete!")
        print("   → SmolagAgent validation: ✅ PASSED")
        print("   → Vision capabilities: " + vision_status)
        print("   → LangGraph workflow: ✅ BUILT") 
        print("   → Ready for question processing")
            
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
        """Initialize orchestration model (LangChain)"""
        try:
            # Initialize orchestration model (LangChain)
            if self.config.model_provider == "openrouter":
                orchestration_model = ChatOpenAI(
                    model=self.config.model_name,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=self.config.temperature
                )
            elif self.config.model_provider == "anthropic":
                orchestration_model = ChatAnthropic(
                    model=self.config.model_name,
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    base_url="https://api.anthropic.com",
                    temperature=self.config.temperature
                )
            elif self.config.model_provider == "groq":
                orchestration_model = ChatGroq(
                    model=self.config.model_name,
                    google_api_key=os.getenv("GROQ_API_KEY"),
                    base_url="https://api.groq.com",
                    temperature=self.config.temperature
                )
            elif self.config.model_provider == "ollama":
                from langchain_ollama import ChatOllama
                orchestration_model = ChatOllama(
                    model=self.config.model_name,
                    base_url=self.config.api_base or "http://localhost:11434",
                    temperature=self.config.temperature
                )
            else:
                # Fallback to Openrouter
                orchestration_model = ChatOpenAI(
                    model="qwen/qwen3-32b",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=self.config.temperature
                )
            
            print(f"✅ Orchestration model: {getattr(orchestration_model, 'model_name', getattr(orchestration_model, 'model', 'Unknown'))}")
            return orchestration_model
            
        except Exception as e:
            print(f"❌ Error initializing orchestration model: {e}")
            raise

    def _initialize_specialist_model(self) -> LiteLLMModel:
        """Initialize LiteLLM model for SmolagAgents"""
        try:
            if self.config.model_provider == "openrouter":
                model_id = f"openrouter/{self.config.model_name}"
                api_key = os.getenv("OPENROUTER_API_KEY")
            elif self.config.model_provider == "anthropic":
                model_id = f"anthropic/{self.config.model_name}"
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.config.model_provider == "ollama":
                model_id = f"ollama_chat/{self.config.model_name}"
                return LiteLLMModel(
                    model_id=model_id,
                    api_base=self.config.api_base or "http://localhost:11434",
                    num_ctx=self.config.num_ctx,
                    temperature=self.config.temperature)
            elif self.config.model_provider == "groq":
                model_id = f"groq/{self.config.model_name}"
                api_key = os.getenv("GROQ_API_KEY")
            else:
                # Fallback to OpenRouter
                model_id = "openrouter/qwen/qwen-2.5-coder-32b-instruct:free"
                api_key = os.getenv("OPENROUTER_API_KEY")
            
            specialist_model = LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                temperature=self.config.temperature
            )
            
            print(f"✅ Specialist model: {model_id}")
            return specialist_model
            
        except Exception as e:
            print(f"❌ Error initializing specialist model: {e}")
            raise
        
    # ============================================================================
    # FILE HANDLING AND VISION CAPABILITIES USING SMOLAGENTS
    # ============================================================================

    def _check_model_vision_capabilities(self) -> Dict[str, bool]:
        """Check if the current model supports vision capabilities"""
        model_info = {
            "has_vision": False,
            "supports_images": False,
            "model_type": "unknown",
            "confidence": "low"
        }
        
        try:
            # Get model identifier from specialist model
            model_id = getattr(self.specialist_model, 'model_id', '').lower()
            
            # Known vision-capable model patterns
            vision_model_patterns = {
                # OpenAI models
                'gpt-4-vision': True,
                'gpt-4o': True,
                'gpt-4-turbo': True,
                
                # Anthropic models
                'claude-sonnet-4': True,  
                'claude-sonnet-3.7': True,
                'claude-sonnet-3.5': True,
                
                # Google models
                'gemini-pro-vision': True,
                'gemini-1.5': True,
                'gemini-2': True,
                'gemini-2.5': True,
                
                # Open source models
                'llava': True,
                'qwen-vl': True,
                'instructblip': True,
                'blip': True,
                
                # Known text-only models
                'gpt-3.5': False,
                'text-davinci': False,
                'qwen-qwq': False,  # Current model in config
                'llama': False,
                'mistral': False,
                'mixtral': False
            }
            
            # Check against known patterns
            for pattern, has_vision in vision_model_patterns.items():
                if pattern in model_id:
                    model_info.update({
                        "has_vision": has_vision,
                        "supports_images": has_vision,
                        "model_type": model_id,
                        "confidence": "high"
                    })
                    break
            else:
                # Unknown model - make conservative assumption
                model_info.update({
                    "has_vision": False,
                    "supports_images": False,
                    "model_type": model_id,
                    "confidence": "low"
                })
                
            # Additional checks for provider-specific patterns
            if 'openrouter/' in model_id:
                # OpenRouter - check the actual model name after the slash
                actual_model = model_id.split('/')[-1]
                for pattern, has_vision in vision_model_patterns.items():
                    if pattern in actual_model:
                        model_info["has_vision"] = has_vision
                        model_info["supports_images"] = has_vision
                        model_info["confidence"] = "high"
                        break
            
        except Exception as e:
            print(f"⚠️ Could not determine model vision capabilities: {e}")
            model_info.update({
                "has_vision": False,
                "supports_images": False,
                "model_type": "error_detecting",
                "confidence": "error"
            })
        
        return model_info

    def _validate_agent_capabilities(self, specialist_agents):
        """Validate capabilities of all specialist agents"""
        
        # Check model capabilities
        model_capabilities = self._check_model_vision_capabilities()
        
        # Check agent availability
        has_vision_browser = 'vision_browser_agent' in specialist_agents
        has_content_processor = 'content_processor' in specialist_agents
        has_data_analyst = 'data_analyst' in specialist_agents
        has_web_researcher = 'web_researcher' in specialist_agents
        
        # Check tool availability
        vision_browser_tool_available = VISION_BROWSER_AVAILABLE
        content_tools_available = CUSTOM_TOOLS_AVAILABLE
        
        capabilities = {
            # Agent availability
            "data_analyst_ready": has_data_analyst,
            "web_researcher_ready": has_web_researcher,
            "content_processor_ready": has_content_processor,
            "vision_browser_ready": has_vision_browser and vision_browser_tool_available,
            
            # Model capabilities
            "model_supports_vision": model_capabilities["has_vision"],
            "model_type": model_capabilities["model_type"],
            
            # Combined capabilities
            "full_vision_capability": (has_vision_browser and 
                                     vision_browser_tool_available and 
                                     model_capabilities["has_vision"]),
            "text_processing_ready": has_content_processor and content_tools_available,
            "data_processing_ready": has_data_analyst,
            "web_search_ready": has_web_researcher,
            
            # Tool status
            "vision_browser_tool_available": vision_browser_tool_available,
            "content_tools_available": content_tools_available,
            
            # Agent count
            "total_agents": len(specialist_agents)
        }
        
        # Status report
        print(f"📊 Agent Capabilities Assessment:")
        print(f"   Data Analyst: {'✅' if capabilities['data_analyst_ready'] else '❌'}")
        print(f"   Web Researcher: {'✅' if capabilities['web_researcher_ready'] else '❌'}")
        print(f"   Content Processor: {'✅' if capabilities['content_processor_ready'] else '❌'}")
        print(f"   Vision Browser: {'✅' if capabilities['vision_browser_ready'] else '❌'}")
        print(f"   Model Vision: {'✅' if capabilities['model_supports_vision'] else '❌'}")
        print(f"   Full Vision: {'✅' if capabilities['full_vision_capability'] else '⚠️'}")
        
        return capabilities

    def _analyze_image_metadata(self, file_path: str) -> Dict[str, Any]:
        """Analyze image file for enhanced vision processing"""
        if not file_path or not os.path.exists(file_path):
            return {"error": "Image file not found"}
        
        try:
            from PIL import Image
            import mimetypes
            
            with Image.open(file_path) as img:
                image_info = {
                    "dimensions": img.size,
                    "format": img.format,
                    "mode": img.mode,
                    "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info,
                    "file_size": os.path.getsize(file_path),
                    "mime_type": mimetypes.guess_type(file_path)[0],
                    "estimated_complexity": "high" if max(img.size) > 2000 else "medium"
                }
                
                # Detect potential content types
                width, height = img.size
                aspect_ratio = width / height
                
                if aspect_ratio > 2.0:
                    image_info["likely_content"] = "panoramic_or_document"
                elif 0.8 <= aspect_ratio <= 1.2:
                    image_info["likely_content"] = "square_diagram_or_chart"
                elif width > 1000 and height > 800:
                    image_info["likely_content"] = "detailed_image_or_screenshot"
                else:
                    image_info["likely_content"] = "standard_image"
                
                return image_info
                
        except Exception as e:
            return {"error": f"Image analysis failed: {str(e)}"}

    def _analyze_file_metadata(self, file_name: str, file_path: str) -> Dict[str, Any]:
        """Simple file analysis without agent_files dependency"""
        if not file_name:
            return {"no_file": True}
        
        try:
            # Choose best available file path
            if file_name and file_name.strip() and os.path.exists(file_name):
                best_path = file_name
            elif file_path and file_path.strip() and os.path.exists(file_path):
                best_path = file_path
            else:
                best_path = file_name or file_path
            
            # Basic file info
            path_obj = Path(file_name) if file_name else Path(file_path)
            extension = path_obj.suffix.lower() if path_obj.suffix else ""
            
            # File existence and size
            file_exists = os.path.exists(best_path) if best_path else False
            file_size = os.path.getsize(best_path) if file_exists else 0
            
            # MIME type
            import mimetypes
            mime_type = mimetypes.guess_type(file_name or file_path)[0] or 'application/octet-stream'
            
            # Enhanced categorization with vision browser routing
            if extension in ['.xlsx', '.csv', '.xls', '.tsv']:
                category, recommended_specialist = "data", "data_analyst"
            elif extension in ['.pdf', '.docx', '.doc', '.txt', '.rtf']:
                category, recommended_specialist = "document", "content_processor"
            elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                # NEW: Route images to vision browser for visual analysis
                category, recommended_specialist = "image", "vision_browser_agent"
            elif extension in ['.mp3', '.mp4', '.wav', '.m4a', '.mov', '.avi']:
                category, recommended_specialist = "media", "content_processor"
            elif extension in ['.py', '.js', '.sql', '.json', '.xml', '.yaml', '.yml']:
                category, recommended_specialist = "code", "content_processor"
            elif extension in ['.zip', '.tar', '.gz', '.rar']:
                category, recommended_specialist = "archive", "content_processor"
            else:
                category, recommended_specialist = "unknown", "content_processor"
            
            return {
                "file_name": file_name,
                "file_path": file_path,
                "best_path": best_path,
                "extension": extension,
                "mime_type": mime_type,
                "category": category,
                "recommended_specialist": recommended_specialist,
                "file_exists": file_exists,
                "file_size": file_size,
                "estimated_complexity": "high" if file_size > 10_000_000 else "medium"
            }
            
        except Exception as e:
            # Error fallback
            return {
                "file_name": file_name,
                "file_path": file_path,
                "best_path": file_path or file_name,
                "category": "unknown",
                "recommended_specialist": "content_processor",
                "error": str(e),
                "file_exists": False,
                "estimated_complexity": "medium"
            }

    def _load_image_for_agent(self, file_path: str) -> Optional[Image.Image]:
        """Load image file for SmolagAgent processing"""
        try:
            if not file_path or not os.path.exists(file_path):
                print(f"❌ Image file not found: {file_path}")
                return None
                
            image = Image.open(file_path).convert("RGB")
            print(f"✅ Image loaded: {file_path}")
            return image
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
        
    def _download_file_once(self, task_id: str, file_name: str) -> str:
        """Download file from GAIA API endpoint"""
        try:
            file_url = f"{self.config.agent_evaluation_api}/files/{task_id}"
            response = requests.get(file_url)
            response.raise_for_status()
            
            # Save with proper extension from file_name
            local_path = f"/tmp/{file_name}"  # Preserves .png extension
            with open(local_path, 'wb') as f:
                f.write(response.content)
                
            return local_path
        except Exception as e:
            print(f"❌ File download failed: {e}")
            return None

    # ============================================================================
    # SMOLAGENTS COORDINATOR AND SPECIALISTS - ENHANCED WITH VISION BROWSER
    # ============================================================================        

    def _create_specialist_agents(self) -> Dict[str, Any]:
        """Create specialist agents using smolagents pattern with separate vision browser agent"""
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        
        specialist_agents = {}
        
        # 1. Data Analyst - CodeAgent for direct Python file access
        try:
            data_analyst = CodeAgent(
                name="data_analyst", 
                description="Excel/CSV analysis and numerical calculations using pandas",
                tools=[],
                additional_authorized_imports=[
                    "pandas", "numpy", "openpyxl", "xlrd", "csv",
                    "scipy", "matplotlib", "seaborn", "statistics", 
                    "math", "json"
                ],
                model=self.specialist_model,
                max_steps=self.config.max_agent_steps,
                add_base_tools=True,
                logger=logger
            )
            specialist_agents["data_analyst"] = data_analyst
            print(f"✅ Data Analyst: {len(data_analyst.additional_authorized_imports)} imports")
        except Exception as e:
            print(f"❌ Failed to create data_analyst: {e}")
        
        # 2. Web Researcher - ToolCallingAgent for text-based search
        try:
            web_tools = []
            try:
                from tools import get_web_researcher_tools
                web_tools = get_web_researcher_tools()
            except ImportError:
                print("⚠️ tools module unavailable - using fallback web tools")
                try:
                    from smolagents import VisitWebpageTool, WikipediaSearchTool
                    web_tools.extend([VisitWebpageTool(), WikipediaSearchTool()])
                except Exception as e:
                    print(f"⚠️ Fallback web tools failed: {e}")

            web_researcher = ToolCallingAgent(
                name="web_researcher",
                description="Search web, find information sources, verify accessibility (text-based)",
                tools=web_tools,
                model=self.specialist_model,
                max_steps=self.config.max_agent_steps,
                add_base_tools=len(web_tools) == 0,
                logger=logger
            )
            specialist_agents["web_researcher"] = web_researcher
            print(f"✅ Web Researcher: {len(web_tools)} tools")
        except Exception as e:
            print(f"❌ Failed to create web_researcher: {e}")
        
        # 3. Content Processor - ToolCallingAgent for document/media processing (NO vision browser)
        try:
            content_tools = []
            try:
                from tools import get_content_processor_tools
                content_tools = get_content_processor_tools()
                # Filter out vision browser tool if present
                content_tools = [tool for tool in content_tools 
                               if not hasattr(tool, '__class__') or 
                               'vision' not in tool.__class__.__name__.lower()]
            except ImportError:
                print("⚠️ tools module unavailable - using base content tools")

            content_processor = ToolCallingAgent(
                name="content_processor", 
                description="Extract and process content from documents, audio, video, and text sources",
                tools=content_tools,
                model=self.specialist_model,
                max_steps=self.config.max_agent_steps,
                add_base_tools=True,
                logger=logger
            )
            specialist_agents["content_processor"] = content_processor
            print(f"✅ Content Processor: {len(content_tools)} tools (vision-free)")
        except Exception as e:
            print(f"❌ Failed to create content_processor: {e}")

        # 4. NEW: Vision Browser Agent - CodeAgent for visual web navigation
        try:
            if VISION_BROWSER_AVAILABLE:
                vision_browser_agent = CodeAgent(
                    name="vision_browser_agent",
                    description="""Visual web navigation and screenshot analysis specialist.
                    
Handles:
- Visual web browsing with screenshots
- Image analysis and OCR
- Interactive web navigation
- Screenshot-based verification
- Visual content extraction

Uses browser automation tools with direct code execution for maximum control.""",
                    tools=[],  # Will use browser tools via imports
                    additional_authorized_imports=[
                        # Browser automation
                        "selenium", "helium", 
                        
                        # Image processing
                        "PIL", "io", "base64", "requests",
                        
                        # Browser setup and configuration
                        "webdriver_manager", "chromedriver_autoinstaller",
                        
                        # File and system operations
                        "os", "sys", "time", "tempfile", "pathlib",
                        
                        # Error handling and retries
                        "contextlib", "functools", "traceback"
                    ],
                    model=self.specialist_model,
                    max_steps=self.config.max_agent_steps,
                    add_base_tools=True,  # Include base tools for completeness
                    logger=logger
                )
                specialist_agents["vision_browser_agent"] = vision_browser_agent
                print(f"✅ Vision Browser Agent: {len(vision_browser_agent.additional_authorized_imports)} imports")
            else:
                print("⚠️ Vision Browser Agent: Not available (missing dependencies)")
        except Exception as e:
            print(f"❌ Failed to create vision_browser_agent: {e}")

        print(f"🎯 Created {len(specialist_agents)} specialist agents total")
        return specialist_agents
        
    def _create_coordinator(self) -> CodeAgent:
        """Create coordinator with enhanced routing for vision browser agent"""
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        
        # Define specialists
        specialist_agents = self._create_specialist_agents()
        
        # Enhanced coordinator with intelligent agent routing
        coordinator = CodeAgent(
            name="gaia_coordinator",
            description="""GAIA task coordinator managing four specialist agents with intelligent routing.

WORKFLOW:
1. ANALYZE TASK: Question analysis and file preprocessing 
2. ROUTE INTELLIGENTLY: Choose appropriate specialist(s) based on task requirements
3. DELEGATE: Assign task to specialist(s) with proper context
4. SYNTHESIZE: Combine results and format GAIA answer

SPECIALIST ROUTING STRATEGY:

DATA ANALYST (CodeAgent):
- Numerical calculations, statistics, math problems
- CSV/Excel file analysis with pandas
- Data processing and aggregation
- Financial calculations, percentages

WEB RESEARCHER (ToolCallingAgent):
- Text-based web searches
- Information discovery and fact-checking
- API-based content retrieval
- Non-visual web content access

CONTENT PROCESSOR (ToolCallingAgent):
- Document processing (PDF, DOCX, TXT)
- Audio/video transcription and analysis
- Text extraction from files
- Archive extraction and processing

VISION BROWSER AGENT (CodeAgent):
- Visual web navigation with screenshots
- Image analysis and OCR tasks
- Interactive web elements requiring visual confirmation
- Screenshot-based verification
- Complex visual content extraction
- Browser automation requiring precise control

ROUTING DECISION MATRIX:
- Image files (.png, .jpg, etc.) → Vision Browser Agent
- Data files (.csv, .xlsx) → Data Analyst
- Documents (.pdf, .docx) → Content Processor
- Web search needs → Web Researcher
- Visual web tasks → Vision Browser Agent
- Mathematical calculations → Data Analyst

The coordinator handles file preparation and intelligent agent selection.""",
            tools=[],  # No tools needed - direct file access via imports
            managed_agents=list(specialist_agents.values()),
            additional_authorized_imports=[
                # File preprocessing (CORE RESPONSIBILITY)
                "zipfile", "tarfile", "gzip", "bz2",  # Archive handling
                
                # File operations and inspection
                "open", "codecs", "chardet", "os", "sys", "io", "pathlib",
                "mimetypes", "tempfile", "shutil",    # File management
                
                # Data preprocessing for specialist preparation
                "pandas", "numpy", "csv", "json", "xml", "base64",
                
                # Binary and media file inspection
                "PIL", "wave", "struct", "binascii",  # Basic media inspection

                # Smart file handling functions
                "smart_file_handler", "is_url",
                
                # Web and networking
                "requests", "urllib", "time", "typing"
            ],
            model=self.specialist_model,
            planning_interval=5,  # More frequent planning for intelligent routing
            max_steps=self.config.max_agent_steps,  # Sufficient steps for coordination + delegation
            logger=logger
        )
        
        return coordinator

    def _build_coordinator_task(self, state: GAIAState) -> str:
        """Build coordination task with enhanced vision browser routing"""
        question = state["question"]
        similar_examples = state.get("similar_examples", [])
        complexity = state.get("complexity", "unknown")
        has_file = state.get("has_file", False)
        file_name = state.get("file_name", "")
        file_path = state.get("file_path", "")
        file_metadata = state.get("file_metadata", {})
        
        # Enhanced file context with vision browser routing
        file_context = ""
        routing_guidance = ""
        
        if has_file:
            # Determine best file reference and provide processing guidance
            if file_name and file_name.strip() and os.path.exists(file_name):
                file_context = f"\nFILE AVAILABLE: {file_name}"
                if file_metadata:
                    file_type = file_metadata.get("category", "unknown")
                    recommended_specialist = file_metadata.get("recommended_specialist", "content_processor")
                    file_context += f"\nFILE TYPE: {file_type}"
                    file_context += f"\nRECOMMENDED PROCESSING: {recommended_specialist}"
                    
                    # Enhanced routing guidance including vision browser
                    if file_type == "data":
                        routing_guidance = "\nROUTING: Use data_analyst for CSV/Excel analysis with pandas"
                    elif file_type == "document":
                        routing_guidance = "\nROUTING: Use content_processor with ContentRetrieverTool"
                    elif file_type == "image":
                        routing_guidance = "\nROUTING: Use vision_browser_agent for visual analysis and OCR"
                    elif file_type == "media":
                        routing_guidance = "\nROUTING: Use content_processor for audio/video transcription"
                    else:
                        routing_guidance = "\nROUTING: Analyze file type and choose appropriate specialist"
                
            elif file_path and file_path.strip() and os.path.exists(file_path):
                file_context = f"\nFILE AVAILABLE: {file_path}"
                if file_name:
                    file_context += f" (original name: {file_name})"
                routing_guidance = "\nROUTING: Inspect file extension and route to appropriate specialist"
                
            else:
                file_context = "\nFILE: Referenced but not accessible - proceed without file"
                routing_guidance = "\nROUTING: Text-based approach, likely web_researcher or direct reasoning"
        
        # Question analysis for routing hints
        question_lower = question.lower()
        additional_routing_hints = ""
        
        if any(term in question_lower for term in ["calculate", "percent", "average", "sum", "count", "data"]):
            additional_routing_hints += "\nHINT: Mathematical/data analysis → Consider data_analyst"
        
        if any(term in question_lower for term in ["search", "find", "look up", "current", "latest"]):
            additional_routing_hints += "\nHINT: Information discovery → Consider web_researcher"
            
        if any(term in question_lower for term in ["image", "picture", "screenshot", "visual", "see", "show"]):
            additional_routing_hints += "\nHINT: Visual analysis → Consider vision_browser_agent"
            
        if any(term in question_lower for term in ["document", "pdf", "text", "extract"]):
            additional_routing_hints += "\nHINT: Document processing → Consider content_processor"
        
        # Build examples context
        examples_context = ""
        if similar_examples:
            examples_context = "\nSIMILAR SUCCESSFUL EXAMPLES:\n"
            for i, example in enumerate(similar_examples[:2], 1):
                examples_context += f"{i}. {example.get('question', '')[:80]}... → {example.get('answer', '')}\n"
        
        # Core coordination task with enhanced routing
        task = f"""GAIA Coordinator Task - Intelligent Agent Routing

QUESTION: {question}
COMPLEXITY: {complexity}{file_context}{routing_guidance}{additional_routing_hints}{examples_context}

AVAILABLE SPECIALISTS:

1. DATA_ANALYST (CodeAgent):
   - Capabilities: pandas, numpy, mathematical calculations
   - Best for: CSV/Excel analysis, numerical computations, statistics
   - File handling: Direct pandas file reading
   
2. WEB_RESEARCHER (ToolCallingAgent):
   - Capabilities: Web search, information discovery
   - Best for: Finding current information, fact-checking
   - File handling: Not applicable
   
3. CONTENT_PROCESSOR (ToolCallingAgent):
   - Capabilities: Document processing, audio/video transcription
   - Best for: PDF/DOCX extraction, media file processing
   - File handling: ContentRetrieverTool with file_path
   
4. VISION_BROWSER_AGENT (CodeAgent):
   - Capabilities: Visual web navigation, image analysis, OCR
   - Best for: Image files, visual web content, screenshot analysis
   - File handling: Direct image loading with PIL

COORDINATOR WORKFLOW:

1. ANALYZE REQUIREMENTS:
   - Question type (calculation, search, visual, document)
   - File type and processing needs
   - Best specialist match

2. INTELLIGENT ROUTING:
   - Single specialist for focused tasks
   - Multiple specialists for complex multi-step tasks
   - Proper delegation with context

3. TASK DELEGATION:
   - Provide clear instructions to chosen specialist(s)
   - Include file_path when applicable
   - Set expectations for output format

4. RESULT SYNTHESIS:
   - Combine specialist outputs if multiple agents used
   - Apply GAIA formatting requirements
   - Ensure answer precision

DELEGATION EXAMPLES:
- For data_analyst: "Analyze the CSV file using pandas and calculate..."
- For web_researcher: "Search for current information about..."
- For content_processor: "Extract content from the file at file_path..."
- For vision_browser_agent: "Analyze the image and identify..."

CRITICAL REQUIREMENTS:
- End with: FINAL ANSWER: [specific answer]
- Format: numbers (no commas), strings (no articles), lists (comma-separated)
- Be precise and factual
- Choose the most appropriate specialist for the task

Execute intelligent routing and delegation now."""

        return task.strip()
                   
    # ============================================================================
    # LANGGRAPH WORKFLOW - UNCHANGED
    # ============================================================================

    def _build_workflow(self):
        """Build LangGraph workflow for question answering"""
        builder = StateGraph(GAIAState)
        
        if self.config.enable_smart_routing:
            # Enhanced smart routing workflow with optimized coordinator
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("complexity_check", self._complexity_check_node)
            builder.add_node("one_shot_answering", self._one_shot_answering_node)
            builder.add_node("coordinator", self._coordinator_node)
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
            
            builder.add_edge("coordinator", "format_answer")
            
            # Simple workflow: one-shot → format
            builder.add_edge("one_shot_answering", "format_answer")
            
            # Both paths converge at formatting
            builder.add_edge("format_answer", END)
            
        else:
            # Linear workflow with optimized coordinator
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("coordinator", self._coordinator_node)
            builder.add_node("format_answer", self._format_answer_node)
            
            # Direct flow - read → coordinate → format
            builder.add_edge(START, "read_question")
            builder.add_edge("read_question", "coordinator")
            builder.add_edge("coordinator", "format_answer")
            builder.add_edge("format_answer", END)
        
        return builder.compile()

    def _read_question_node(self, state: GAIAState) -> GAIAState:
        """Question reading with file info extraction and download"""
        task_id = state.get("task_id")
        question = state["question"]
        
        # Start context bridge tracking
        if self.config.enable_context_bridge:
            ContextBridge.start_task_execution(task_id)
            track("Processing question and extracting file info", self.config)
        
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
        """🎯 SIMPLIFIED: File check + LLM assessment only"""
        question = state["question"]
        has_file = state.get("has_file", False)
        
        if self.logging:
            self.logging.log_step("complexity_analysis", f"Analyzing complexity: {question[:50]}...")
        
        print(f"🧠 Analyzing complexity: {question[:50]}...")
        
        # Only two rules:
        if has_file:
            complexity = "complex"
            reason = "File attachment detected"
        else:
            complexity = self._llm_complexity_check(question)
            reason = "LLM multi-step assessment"
        
        print(f"📊 Complexity: {complexity} ({reason})")
        
        if self.logging:
            self.logging.set_complexity(complexity)
            self.logging.log_step("complexity_result", f"Final complexity: {complexity} - {reason}")
        
        # RAG for complex questions if needed
        similar_examples = state.get("similar_examples", [])
        if complexity == "complex" and self.config.skip_rag_for_simple and not similar_examples:
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
                
                if self.logging:
                    self.logging.set_similar_examples_count(len(similar_examples))
                    
            except Exception as e:
                print(f"⚠️ RAG retrieval error: {e}")
        
        return {
            "complexity": complexity,
            "similar_examples": similar_examples,
            "steps": state["steps"] + [f"Complexity assessed: {complexity} ({reason})"]
        }

    def _llm_complexity_check(self, question: str) -> str:
        """
        Does answering this question require external tools or just training knowledge?
        """
        
        prompt = f"""Question: {question}

To answer this question, do I need to search the web, access files, or use external tools?
Or can I answer it directly from my training knowledge alone?

SIMPLE = I can answer from my training knowledge alone
COMPLEX = I need to search the web, access files, or use tools to find current/specific information

Answer just: SIMPLE or COMPLEX"""
        
        try:
            response = llm_invoke_with_retry(self.orchestration_model, [HumanMessage(content=prompt)])
            result = response.content.strip().upper()
            
            if "SIMPLE" in result:
                return "simple"
            else:
                return "complex"
                
        except Exception as e:
            print(f"⚠️ LLM complexity check failed: {str(e)}, defaulting to complex")
            return "complex"

    def _route_by_complexity(self, state: GAIAState) -> str:
        """Routing function for conditional edges"""
        return state.get("complexity", "complex")

    def _one_shot_answering_node(self, state: GAIAState):
        """Direct LLM answering for simple questions"""
        task_id = state.get("task_id", "")
        
        print("⚡ Using one-shot direct answering")
        
        track("Direct LLM answering for simple question", self.config)
        
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
===========

def _coordinator_node(self, state: GAIAState) -> GAIAState:
        """Enhanced coordinator with vision browser agent support"""
        task_id = state.get("task_id", "unknown")
        question = state["question"]
        
        print(f"🧠 Enhanced Coordinator: {question[:80]}...")
        
        # Enhanced tracking and logging
        track("Starting enhanced coordinator with vision browser support", self.config)
        
        if self.config.enable_context_bridge:
            ContextBridge.track_operation("Starting enhanced coordinator")
        
        if self.logging:
            self.logging.log_step("coordinator_start", "Starting enhanced coordinator analysis and execution", {
                'task_id': task_id,
                'node_name': 'coordinator',
                'question_preview': question[:100]
            })
            self.logging.set_routing_path("enhanced_hierarchical_coordinator")
        
        try:
            # Create fresh coordinator with managed specialists for this task
            track("Creating enhanced coordinator with vision browser agent", self.config)
            self.coordinator = self._create_coordinator()
            
            if self.logging:
                self.logging.log_step("coordinator_created", "Enhanced coordinator with specialists created", {
                    'specialist_count': len(self.coordinator.managed_agents) if hasattr(self.coordinator, 'managed_agents') else 0,
                    'vision_browser_available': VISION_BROWSER_AVAILABLE
                })
            
            # Enhanced file path analysis and selection
            file_path = state.get("file_path", "")
            file_name = state.get("file_name", "")
            has_file = state.get("has_file", False)
            file_metadata = state.get("file_metadata", {})
            
            # Inline file path selection logic
            best_file_path = ""
            file_selection_method = "none"
            
            if has_file:
                track("Analyzing file information for enhanced coordinator", self.config)
                
                # Prefer filename if it exists and has extension
                if file_name and file_name.strip() and os.path.exists(file_name):
                    best_file_path = file_name
                    file_selection_method = "filename_with_extension"
                # Fallback to cache path if it exists
                elif file_path and file_path.strip() and os.path.exists(file_path):
                    best_file_path = file_path
                    file_selection_method = "cache_path_fallback"
                # Return whatever we have, even if it doesn't exist
                elif file_name and file_name.strip():
                    best_file_path = file_name
                    file_selection_method = "filename_preferred"
                elif file_path and file_path.strip():
                    best_file_path = file_path
                    file_selection_method = "cache_path_only"
                else:
                    file_selection_method = "no_accessible_file"
                
                if self.logging:
                    self.logging.log_step("file_analysis", f"File path selected: {file_selection_method}", {
                        'file_name': file_name,
                        'file_path': file_path[:50] + "..." if len(file_path) > 50 else file_path,
                        'best_file_path': best_file_path[:50] + "..." if len(best_file_path) > 50 else best_file_path,
                        'selection_method': file_selection_method,
                        'file_exists': os.path.exists(best_file_path) if best_file_path else False,
                        'file_metadata': file_metadata.get("category", "unknown") if file_metadata else "none"
                    })
                
                print(f"📁 File selection: {file_selection_method} → {os.path.basename(best_file_path) if best_file_path else 'none'}")
            
            # Build coordination task with enhanced context
            track("Building enhanced coordination task with vision browser routing", self.config)
            coordination_task = self._build_coordinator_task(state)
            
            if self.logging:
                self.logging.log_step("task_built", "Enhanced coordination task constructed", {
                    'task_length': len(coordination_task),
                    'has_file_context': has_file,
                    'similar_examples_count': len(state.get("similar_examples", [])),
                    'vision_browser_available': VISION_BROWSER_AVAILABLE
                })
            
            # Context bridge tracking for execution phase
            if self.config.enable_context_bridge:
                ContextBridge.track_operation("Executing enhanced coordinator with vision browser support")
            
            # Enhanced execution strategy determination
            has_image = any(ext in file_name.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']) if file_name else False
            execution_strategy = "no_file"
            
            if best_file_path and has_image and VISION_BROWSER_AVAILABLE:
                execution_strategy = "vision_browser_image_analysis"
            elif best_file_path and has_image:
                execution_strategy = "image_fallback_processing"
            elif best_file_path:
                execution_strategy = "file_processing"
            
            if self.logging:
                self.logging.log_step("execution_strategy", f"Enhanced execution strategy: {execution_strategy}", {
                    'strategy': execution_strategy,
                    'has_file': bool(best_file_path),
                    'is_image': has_image,
                    'vision_browser_available': VISION_BROWSER_AVAILABLE,
                    'file_type': file_metadata.get("category", "unknown") if file_metadata else "unknown"
                })
            
            # Execute coordination based on enhanced strategy
            coordination_result = None
            
            if execution_strategy == "vision_browser_image_analysis":
                track("Loading image for vision browser analysis", self.config)
                image = self._load_image_for_agent(best_file_path)
                
                if image:
                    print("🖼️ Running enhanced coordinator with vision browser support")
                    track("Executing enhanced coordinator with vision browser image analysis", self.config)
                    
                    if self.logging:
                        self.logging.log_step("vision_browser_execution", "Executing with vision browser support", {
                            'image_loaded': True,
                            'file_path': best_file_path,
                            'execution_strategy': 'vision_browser_image_analysis'
                        })
                    
                    coordination_result = self.coordinator.run(
                        coordination_task,
                        images=[image],
                        additional_args={
                            "file_path": best_file_path, 
                            "file_name": file_name,
                            "use_vision_browser": True,
                            "image_analysis_required": True
                        }
                    )
                else:
                    print("⚠️ Image loading failed, falling back to standard file processing")
                    track("Image loading failed, fallback to enhanced file processing", self.config)
                    
                    if self.logging:
                        self.logging.log_step("vision_fallback", "Image loading failed, proceeding with file processing", {
                            'image_loaded': False,
                            'fallback_strategy': 'enhanced_file_processing'
                        })
                    
                    coordination_result = self.coordinator.run(
                        coordination_task,
                        additional_args={
                            "file_path": best_file_path, 
                            "file_name": file_name,
                            "use_vision_browser": False
                        }
                    )
                    
            elif execution_strategy == "image_fallback_processing":
                print(f"📄 Running enhanced coordinator with image file (no vision browser): {os.path.basename(best_file_path)}")
                track("Executing enhanced coordinator with image fallback processing", self.config)
                
                if self.logging:
                    self.logging.log_step("image_fallback_execution", "Executing with image fallback processing", {
                        'file_basename': os.path.basename(best_file_path),
                        'vision_browser_available': False,
                        'file_size': os.path.getsize(best_file_path) if os.path.exists(best_file_path) else 0
                    })
                
                coordination_result = self.coordinator.run(
                    coordination_task,
                    additional_args={
                        "file_path": best_file_path, 
                        "file_name": file_name,
                        "image_fallback": True
                    }
                )
                
            elif execution_strategy == "file_processing":
                print(f"📄 Running enhanced coordinator with file: {os.path.basename(best_file_path)}")
                track("Executing enhanced coordinator with standard file processing", self.config)
                
                if self.logging:
                    self.logging.log_step("file_execution", "Executing with enhanced file processing", {
                        'file_basename': os.path.basename(best_file_path),
                        'file_size': os.path.getsize(best_file_path) if os.path.exists(best_file_path) else 0
                    })
                
                coordination_result = self.coordinator.run(
                    coordination_task,
                    additional_args={"file_path": best_file_path, "file_name": file_name}
                )
                
            else:  # no_file strategy
                print("📝 Running enhanced coordinator without files")
                track("Executing enhanced coordinator without files", self.config)
                
                if self.logging:
                    self.logging.log_step("no_file_execution", "Executing without files", {
                        'text_only': True,
                        'enhanced_routing': True
                    })
                
                coordination_result = self.coordinator.run(coordination_task)
            
            # Post-execution tracking and logging
            track("Enhanced hierarchical coordinator completed analysis and execution", self.config)
            
            if self.config.enable_context_bridge:
                ContextBridge.track_operation("Enhanced coordinator execution completed successfully")
            
            print(f"✅ Enhanced hierarchical coordinator completed: {execution_strategy}")
            
            if self.logging:
                self.logging.log_step("coordinator_complete", "Enhanced hierarchical coordinator execution completed", {
                    'execution_strategy': execution_strategy,
                    'result_length': len(str(coordination_result)),
                    'success': True,
                    'vision_browser_used': execution_strategy == "vision_browser_image_analysis"
                })
            
            # Return enhanced state with execution details
            return {
                **state,
                "agent_used": "enhanced_hierarchical_coordinator",
                "raw_answer": coordination_result,
                "execution_successful": True,
                "execution_strategy": execution_strategy,
                "file_selection_method": file_selection_method,
                "vision_browser_used": execution_strategy == "vision_browser_image_analysis",
                "steps": state["steps"] + [f"Enhanced hierarchical coordinator completed ({execution_strategy})"]
            }
            
        except Exception as e:
            error_msg = f"Enhanced coordinator failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Enhanced error tracking and logging
            track(f"Enhanced coordinator error: {str(e)}", self.config)
            
            if self.config.enable_context_bridge:
                ContextBridge.track_error(error_msg)
            
            if self.logging:
                self.logging.log_step("coordinator_error", error_msg, {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'task_id': task_id,
                    'execution_strategy': locals().get('execution_strategy', 'unknown'),
                    'file_selection_method': locals().get('file_selection_method', 'unknown'),
                    'vision_browser_available': VISION_BROWSER_AVAILABLE
                })
            
            return {
                **state,
                "current_error": error_msg,
                "agent_used": "enhanced_hierarchical_coordinator",
                "raw_answer": error_msg,
                "execution_successful": False,
                "execution_strategy": locals().get('execution_strategy', 'failed'),
                "file_selection_method": locals().get('file_selection_method', 'failed'),
                "vision_browser_used": False,
                "steps": state["steps"] + [f"Enhanced hierarchical coordinator failed: {error_msg}"]
            }

    def _format_answer_node(self, state: GAIAState) -> GAIAState:
        """
        Format the final answer with robust type checking and GAIA compliance.
        
        Handles multiple result formats from coordinator:
        - Integer/float: Direct numerical answers
        - String: Text answers 
        - Dict: Structured results with metadata
        - List: Multiple answers or steps
        """
        # Logging
        if hasattr(self, 'logging_setup') and self.logging_setup:
            self.logging_setup.log_step(
                action='format_start',
                details='Starting answer formatting',
                node_name='format_answer'
            )
        
        try:
            raw_answer = state.get("raw_answer", "")
            task_id = state.get("task_id", "")
            question = state.get("question", "")
            
            # Store question for formatting context
            self._current_question = question
            
            if hasattr(self, 'logging_setup') and self.logging_setup:
                self.logging_setup.log_step(
                    action='extract_answer',
                    details=f'Extracting final answer from raw response: {str(raw_answer)[:50]}...',
                    node_name='format_answer'
                )
            
            # Call the final answer method to extract info
            formatted_answer = self._extract_final_answer(raw_answer)
            
            if hasattr(self, 'logging_setup') and self.logging_setup:
                self.logging_setup.log_step(
                    action='apply_gaia_format',
                    details=f'Applying GAIA formatting to: {formatted_answer}',
                    node_name='format_answer'
                )
            
            # Apply GAIA formatting
            formatted_answer = self._apply_gaia_formatting(formatted_answer)
            
            if hasattr(self, 'logging_setup') and self.logging_setup:
                self.logging_setup.log_step(
                    action='format_complete',
                    details=f'Final formatted answer: {formatted_answer}',
                    node_name='format_answer'
                )
            
            # Write new info to state
            final_state = {
                **state,
                "final_answer": formatted_answer,
                "raw_answer": str(raw_answer),
                "steps": state.get("steps", []) + ["Final answer formatting applied"],
                "execution_successful": True
            }
            
            # LOG COMPLETION
            if hasattr(self, 'logging_setup') and self.logging_setup:
                self.logging_setup.log_step(
                    action='question_complete',
                    details=f'Final answer: {formatted_answer}',
                    node_name='format_answer'
                )
            
            return final_state
            
        except Exception as e:
            # Enhanced error handling
            error_msg = f"Processing failed: {str(e)}"
            
            # PRESERVE ERROR LOGGING
            if hasattr(self, 'logging_setup') and self.logging_setup:
                self.logging_setup.log_step(
                    action='question_complete',
                    details=error_msg,
                    node_name='format_answer'
                )
            
            # Fallback to raw answer
            fallback_answer = state.get("raw_answer", "No answer")
            
            return {
                **state,
                "final_answer": str(fallback_answer).strip() if fallback_answer else "No answer",
                "raw_answer": str(fallback_answer),
                "steps": state.get("steps", []) + [error_msg],
                "execution_successful": False
            }

    def _extract_final_answer(self, raw_answer: str) -> str:
        """
        Extract final answer from agent response with pattern matching
        """
        import re  
        if raw_answer is None:
            if hasattr(self, 'logging_setup') and self.logging_setup:
                self.logging_setup.log_step(
                    action='extract_empty',
                    details='Raw answer is None',
                    node_name='format_answer'
                )
            return "No answer"
        
        # Convert list to string if needed
        if isinstance(raw_answer, list):
            if hasattr(self, 'logging_setup') and self.logging_setup:
                self.logging_setup.log_step(
                    action='extract_list_input',
                    details=f'Converting list to string: {raw_answer}',
                    node_name='format_answer'
                )
            raw_answer = str(raw_answer[0]) if raw_answer else "No answer"
        
        # Convert int/float to string
        if isinstance(raw_answer, (int, float)):
            if hasattr(self, 'logging_setup') and self.logging_setup:
                self.logging_setup.log_step(
                    action='extract_number_input',
                    details=f'Converting {type(raw_answer).__name__} to string: {raw_answer}',
                    node_name='format_answer'
                )
            raw_answer = str(raw_answer)
        
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
        
        import re
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, raw_answer, re.IGNORECASE | re.DOTALL)
            if matches:
                # Take the last match (most likely to be the final answer)
                extracted = matches[-1].strip()
                if extracted and extracted.lower() not in ["", "no answer", "none"]:
                    if hasattr(self, 'logging_setup') and self.logging_setup:
                        self.logging_setup.log_step(
                            action='extract_success',
                            details=f'Pattern {i+1} matched: {extracted}',
                            node_name='format_answer'
                        )
                    return extracted
        
        # Fallback: Look for the last substantial line
        lines = [line.strip() for line in raw_answer.strip().split('\n') if line.strip()]
        
        # Try to find a line that looks like an answer
        for line in reversed(lines):
            if len(line) > 0 and not line.lower().startswith(('i ', 'the ', 'let ', 'to ', 'in ', 'based ')):
                if hasattr(self, 'logging_setup') and self.logging_setup:
                    self.logging_setup.log_step(
                        action='extract_fallback',
                        details=f'Using last substantial line: {line}',
                        node_name='format_answer'
                    )
                return line
        
        # Final fallback
        fallback = lines[-1] if lines else "No answer"
        
        if hasattr(self, 'logging_setup') and self.logging_setup:
            self.logging_setup.log_step(
                action='extract_final_fallback',
                details=f'Final fallback: {fallback}',
                node_name='format_answer'
            )
        
        return fallback

    def _apply_gaia_formatting(self, answer: str) -> str:
        """
        GAIA formatting with safe regex operations and type safety
        """
        import re
        if not answer:
            return "No answer"
        
        # 🔧 TYPE SAFETY: Handle non-string inputs
        if isinstance(answer, (int, float)):
            answer = str(answer)
        elif isinstance(answer, list):
            answer = str(answer[0]) if answer else "No answer"
        else:
            answer = str(answer)
        
        original_answer = answer
        answer = answer.strip()
        
        if hasattr(self, 'logging_setup') and self.logging_setup:
            self.logging_setup.log_step(
                action='gaia_format_start',
                details=f'Original answer: \'{answer}\'',
                node_name='format_answer'
            )
        
        # Helper function for safe regex
        def safe_regex_search(pattern, text, flags=0):
            try:
                import re
                return re.search(pattern, text, flags)
            except Exception:
                return None
        
        # SAFE: Extract from FINAL ANSWER patterns
        final_answer_patterns = [
            r'(?i)^.*?final\s*answer\s*:\s*(.*)$',
            r'(?i)final\s*answer\s*:\s*(.+?)(?:\n|$)',
            r'(?i).*final\s*answer\s*:\s*(.+)',
        ]
        
        for pattern in final_answer_patterns:
            match = safe_regex_search(pattern, answer, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    answer = extracted
                    if hasattr(self, 'logging_setup') and self.logging_setup:
                        self.logging_setup.log_step(
                            action='gaia_format_extract',
                            details=f'Extracted: \'{answer}\'',
                            node_name='format_answer'
                        )
                    break
        
        # SAFE: Remove prefixes
        prefixes_to_remove = [
            "final answer:", "the answer is:", "answer:", "result:",
            "solution:", "the result is:", "therefore", "so", "thus"
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        # Clean quotes and punctuation
        answer = answer.strip('.,!?:;"\'')
        
        # Question-specific formatting
        question = getattr(self, '_current_question', '').lower()
        
        # SAFE: Standard GAIA formatting
        if not any(special in question for special in ["comma", "list", "countries"]):
            # Remove articles (a, an, the) from the beginning
            import re
            answer = re.sub(r'^(a|an|the)\s+', '', answer, flags=re.IGNORECASE)
        
        # Number formatting - remove commas from numbers
        if answer.replace(',', '').replace('.', '').replace('-', '').isdigit():
            answer = answer.replace(',', '')
        
        # Final cleanup
        answer = answer.strip()
        
        if hasattr(self, 'logging_setup') and self.logging_setup:
            self.logging_setup.log_step(
                action='gaia_format_final',
                details=f'Final formatted: \'{answer}\'',
                node_name='format_answer'
            )
        
        return answer if answer else "No answer"

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

def safe_regex_search(pattern: str, text: str, flags=0):
    """Safe regex search with error handling"""
    if not isinstance(text, str):
        return None
        
    try:
        compiled_pattern = re.compile(pattern, flags)
        return compiled_pattern.search(text)
    except (re.error, MemoryError, Exception) as e:
        print(f"⚠️  Regex search error: {e}")
        return None

def safe_regex_sub(pattern: str, replacement: str, text: str, flags=0) -> str:
    """Safely apply regex substitution with error handling"""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    try:
        compiled_pattern = re.compile(pattern, flags)
        return compiled_pattern.sub(replacement, text)
    except (re.error, MemoryError, Exception) as e:
        print(f"⚠️  Regex error: {e}, returning original text")
        return text

def safe_regex_findall(pattern: str, text: str, flags=0) -> List[str]:
    """Safe regex findall with error handling"""
    if not isinstance(text, str):
        return []
        
    try:
        compiled_pattern = re.compile(pattern, flags)
        return compiled_pattern.findall(text)
    except (re.error, MemoryError, Exception) as e:
        print(f"⚠️  Regex findall error: {e}")
        return []


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