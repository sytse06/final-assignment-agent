# agent_logic.py
# GAIA Agent System using SmolagAgents Manager Pattern

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
    model_provider: str = "groq"
    model_name: str = "qwen-qwq-32b"
    temperature: float = 0.3
    agent_evaluation_api: str = "https://agents-course-unit4-scoring.hf.space/"
    
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

def _load_image_for_agent(self, file_path: str) -> Image.Image:
    """Load image file for SmolagAgent processing"""
    try:
        if os.path.exists(file_path):
            image = Image.open(file_path).convert("RGB")
            print(f"✅ Image loaded: {file_path}")
            return image
        else:
            print(f"❌ Image file not found: {file_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

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
            # Test specialist creation immediately - this will reveal parameter issues
            print("  → Testing specialist agents creation...")
            test_specialists = self._create_specialist_agents()
            print(f"  ✅ Created {len(test_specialists)} specialists successfully")
            print(f"     Specialists: {list(test_specialists.keys())}")
            
            # Validate content processor capabilities including model vision support
            print("  → Validating content processor capabilities...")
            if 'content_processor' in test_specialists:
                content_tools = getattr(test_specialists['content_processor'], 'tools', [])
                self.capabilities = self._validate_content_processor_tools(content_tools)
            else:
                print("⚠️ No content processor found in specialists")
                self.capabilities = {"effective_vision_capability": False}
            
            # Test coordinator creation
            print("  → Testing coordinator creation...")
            test_coordinator = self._create_coordinator()
            print("  ✅ Coordinator created successfully")
            print(f"     Managed agents: {len(test_coordinator.managed_agents) if hasattr(test_coordinator, 'managed_agents') else 'Unknown'}")
            
            # Test coordinator task execution
            print("  → Testing coordinator execution...")
            if self.capabilities.get("model_supports_vision", False):
                simple_test_result = test_coordinator.run("What is 2+2? Answer with just the number.")
            else:
                print("    (Using text-only test - model has no vision capabilities)")
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
        vision_status = "✅ Full Vision" if self.capabilities.get("effective_vision_capability") else "⚠️ OCR Fallback Only"
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
            if self.config.model_provider == "groq":
                orchestration_model = ChatGroq(
                    model=self.config.model_name,
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=self.config.temperature
                )
            elif self.config.model_provider == "openrouter":
                orchestration_model = ChatOpenAI(
                    model=self.config.model_name,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=self.config.temperature
                )
            elif self.config.model_provider == "google":
                orchestration_model = ChatGoogleGenerativeAI(
                    model=self.config.model_name,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
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
        """🔥 NEW: Initialize LiteLLM model for SmolagAgents"""
        try:
            if self.config.model_provider == "groq":
                model_id = f"groq/{self.config.model_name}"
                api_key = os.getenv("GROQ_API_KEY")
            elif self.config.model_provider == "openrouter":
                model_id = f"openrouter/{self.config.model_name}"
                api_key = os.getenv("OPENROUTER_API_KEY")
            elif self.config.model_provider == "ollama":
                model_id = f"ollama_chat/{self.config.model_name}"
                return LiteLLMModel(
                    model_id=model_id,
                    api_base=self.config.api_base or "http://localhost:11434",
                    num_ctx=self.config.num_ctx,
                    temperature=self.config.temperature)
            elif self.config.model_provider == "google":
                model_id = f"gemini/{self.config.model_name}"
                api_key = os.getenv("GOOGLE_API_KEY")
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
                'claude-3': True,
                'claude-3.5': True,
                
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

    def _validate_content_processor_tools(self, content_tools):
        """Enhanced validation for vision and content tools + model capabilities"""
        tool_names = [getattr(tool, 'name', tool.__class__.__name__) for tool in content_tools]
        
        # Tool capability validation
        has_vision_browser = any('Vision' in name or 'Browser' in name for name in tool_names)
        has_content_retriever = any('Content' in name or 'Retriever' in name for name in tool_names)
        has_speech_tool = any('Speech' in name or 'Text' in name for name in tool_names)
        
        # Model capability validation
        model_capabilities = self._check_model_vision_capabilities()
        
        # Comprehensive capabilities assessment
        capabilities = {
            "vision_navigation": has_vision_browser,
            "content_extraction": has_content_retriever, 
            "audio_processing": has_speech_tool,
            "total_tools": len(content_tools),
            
            # Model capabilities
            "model_supports_vision": model_capabilities["has_vision"],
            "model_supports_images": model_capabilities["supports_images"],
            "model_type": model_capabilities["model_type"],
            "model_confidence": model_capabilities["confidence"],
            
            # Combined capability assessment
            "effective_vision_capability": has_vision_browser and model_capabilities["has_vision"],
            "fallback_strategy": "ocr_text_extraction" if not model_capabilities["has_vision"] else "full_vision"
        }
        
        # Status reporting with actionable information
        print(f"📊 Content Processor Capabilities Assessment:")
        print(f"   🔧 Tools: {len(content_tools)} total")
        print(f"   🌐 Vision Browser: {'✅' if has_vision_browser else '❌'}")
        print(f"   📄 Content Retriever: {'✅' if has_content_retriever else '❌'}")
        print(f"   🎵 Speech Processing: {'✅' if has_speech_tool else '❌'}")
        print(f"   👁️  Model Vision Support: {'✅' if model_capabilities['has_vision'] else '❌'} ({model_capabilities['model_type']})")
        print(f"   🎯 Effective Vision Capability: {'✅ Full Vision' if capabilities['effective_vision_capability'] else '⚠️ OCR Fallback Only'}")
        
        # Warnings and recommendations
        if not has_vision_browser:
            print("⚠️  No vision browser tools - install vision_browser_tool for web content acquisition")
        if not has_content_retriever:
            print("⚠️  No content retrieval tools - limited document processing capabilities")
        if not model_capabilities["has_vision"]:
            print("ℹ️  Text-only model detected - vision questions will use OCR text extraction fallback")
            print("💡 For full vision capabilities, consider using: GPT-4V, Claude-3, Gemini-Pro-Vision, or LLaVA")
        if has_vision_browser and not model_capabilities["has_vision"]:
            print("⚠️  Vision tools available but model can't process images - will use OCR extraction")
        
        return capabilities

    def _download_file_once(self, task_id: str, file_name: str) -> Optional[str]:
        """Download attached file with comprehensive error handling"""
        try:
            agent_evaluation_api = self.config.agent_evaluation_api
            file_url = f"{agent_evaluation_api}files/{task_id}"
            
            print(f"📡 Downloading: {file_url}")
            
            response = requests.get(
                file_url,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=30,
                stream=True
            )
            
            if response.status_code == 200:
                # Create temp file with correct extension
                file_extension = os.path.splitext(file_name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp_file.write(chunk)
                    local_path = tmp_file.name
                
                file_size = os.path.getsize(local_path)
                print(f"✅ Downloaded: {local_path} ({file_size} bytes)")
                return local_path
                
            else:
                print(f"❌ Download failed: HTTP {response.status_code}")
                return None
                
        except requests.RequestException as e:
            print(f"❌ Network error downloading file: {e}")
            return None
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None

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
        """Enhanced file analysis with vision-aware metadata"""
        if not file_name:
            return {"no_file": True}
        
        try:
            # Basic file analysis
            path_obj = Path(file_name)
            extension = path_obj.suffix.lower()
            
            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_name)
            
            # Categorize file type with enhanced image handling
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
                recommended_specialist = "content_processor"
                specialist_guidance = {
                    "tool_command": "Use ContentRetrieverTool with file_path from additional_args",
                    "imports_needed": [],
                    "processing_strategy": "Extract content → analyze text → answer question"
                }
            elif extension in ['.mp3', '.mp4', '.wav', '.m4a', '.mov']:
                category = "media"
                processing_approach = "transcription"
                recommended_specialist = "content_processor"
                specialist_guidance = {
                    "tool_command": "Use SpeechToTextTool with file_path from additional_args",
                    "imports_needed": [],
                    "processing_strategy": "Transcribe audio → analyze transcript → extract information"
                }
            elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                category = "image"
                processing_approach = "vision_analysis"
                recommended_specialist = "content_processor"
                
                # 🔥 NEW: Enhanced image analysis
                image_metadata = self._analyze_image_metadata(file_path)
                
                # Adjust processing strategy based on model capabilities
                if self.capabilities.get("model_supports_vision", False):
                    processing_strategy = "Load image → analyze visual content → extract information"
                    tool_command = "Use vision tools with file_path and images from additional_args"
                else:
                    processing_strategy = "Load image → OCR text extraction → analyze text"
                    tool_command = "Use ContentRetrieverTool for OCR text extraction"
                
                specialist_guidance = {
                    "tool_command": tool_command,
                    "imports_needed": ["PIL"],
                    "processing_strategy": processing_strategy,
                    "image_info": image_metadata,
                    "vision_capable": self.capabilities.get("model_supports_vision", False)
                }
            else:
                category = "unknown"
                processing_approach = "content_extraction"
                recommended_specialist = "content_processor"
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
                "recommended_specialist": "content_processor",
                "error": str(e)
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

    # ============================================================================
    # SMOLAGENTS COORDINATOR AND SPECIALISTS
    # ============================================================================        

    def _create_specialist_agents(self) -> Dict[str, Any]:
        """Create specialist agents using smolagents pattern"""
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        
        # 1. Data Analyst - CodeAgent for direct Python file access
        data_analyst = CodeAgent(
            name="data_analyst", 
            description="Excel/CSV analysis and numerical calculations using pandas",
            tools=[],
            additional_authorized_imports=[
                "pandas", "numpy", "openpyxl", "xlrd", "csv",
                "scipy", "matplotlib", "seaborn", 
                "sklearn", "scikit-learn", "statistics", "math"
            ],
            model=self.specialist_model,
            max_steps=self.config.max_agent_steps,
            add_base_tools=True,
            logger=logger
        )
        
        # 2. Web Researcher - ToolCallingAgent for search with robust fallback system
        web_tools = []
        web_tool_source = "none"  # Track what tools we're using
        
        # Try LangChain tools first (superior quality)
        if LANGCHAIN_TOOLS_AVAILABLE:
            try:
                available_research_tools = get_langchain_tools()
                tool_status = get_tool_status()
                
                if tool_status['research_tools_available']:
                    web_tools.extend(available_research_tools)
                    web_tool_source = "langchain"
                    print(f"✅ Added {len(available_research_tools)} LangChain research tools")
                    
                    # Add complementary VisitWebpageTool (still useful with LangChain)
                    try:
                        from smolagents import VisitWebpageTool
                        web_tools.append(VisitWebpageTool())
                        print("✅ Added VisitWebpageTool for webpage content extraction")
                    except Exception as e:
                        print(f"⚠️  VisitWebpageTool failed: {e}")
                        
            except Exception as e:
                print(f"⚠️  LangChain tools error: {e}")
        
        # Fallback to native SmolagAgent tools ONLY if LangChain tools failed
        if not web_tools:
            print("🔄 Using native SmolagAgent tools...")
            web_tool_source = "native"
            
            # Try GoogleSearchTool
            if os.getenv("SERPER_API_KEY") or os.getenv("SERPAPI_API_KEY"):
                try:
                    from smolagents import GoogleSearchTool
                    provider = "serper" if os.getenv("SERPER_API_KEY") else "serpapi"
                    web_tools.append(GoogleSearchTool(provider=provider))
                    print(f"✅ Added GoogleSearchTool ({provider})")
                except Exception as e:
                    print(f"⚠️  GoogleSearchTool failed: {e}")
            
            # Add other native tools
            try:
                from smolagents import VisitWebpageTool, WikipediaSearchTool
                web_tools.extend([VisitWebpageTool(), WikipediaSearchTool()])
                print("✅ Added VisitWebpageTool and WikipediaSearchTool")
            except Exception as e:
                print(f"⚠️  Native tools failed: {e}")
            
            # Final fallback to DuckDuckGo
            if not web_tools:
                try:
                    import duckduckgo_search  # Check dependency
                    from smolagents import DuckDuckGoSearchTool
                    web_tools.append(DuckDuckGoSearchTool())
                    print("✅ Added DuckDuckGoSearchTool")
                except ImportError:
                    print("⚠️  Install: pip install duckduckgo-search")
                except Exception as e:
                    print(f"⚠️  DuckDuckGoSearchTool failed: {e}")
        
        # Create web researcher description
        if web_tools:
            tool_names = [getattr(t, 'name', t.__class__.__name__) for t in web_tools]
            web_researcher_description = f"Web researcher with {len(web_tools)} tools: {', '.join(tool_names)}"
        else:
            web_researcher_description = "Basic web researcher (no tools available)"
            print("❌ No web tools available")

        # Check base tools dependency
        use_base_tools = False
        try:
            import duckduckgo_search
            use_base_tools = True
        except ImportError:
            print("⚠️  Base tools disabled (missing duckduckgo-search)")

        # Create web researcher
        web_researcher = ToolCallingAgent(
            name="web_researcher",
            description=web_researcher_description,
            tools=web_tools,
            model=self.specialist_model,
            max_steps=self.config.max_agent_steps,
            add_base_tools=use_base_tools,
            logger=logger
        )
        
        # Status report
        if web_tool_source == "langchain":
            print(f"✅ Created web_researcher with LangChain tools: Enhanced capabilities")
        elif web_tool_source == "native":
            print(f"✅ Created web_researcher with native tools: Basic capabilities")
        else:
            print(f"⚠️  Created web_researcher with no tools: Limited capabilities")
        
        # 3. Content Processor - Multi-modal content acquisition and processing
        content_tools = []

        # Add VisionWebBrowserTool if available
        try:
            from vision_browser_tool import VisionWebBrowserTool  # Corrected path
            vision_browser = VisionWebBrowserTool()
            if hasattr(vision_browser, 'setup'):
                vision_browser.setup()
            content_tools.append(vision_browser)
            print("✅ Added VisionWebBrowserTool to content processor")
        except ImportError as e:
            print(f"⚠️  VisionWebBrowserTool requires helium selenium: {e}")
        except Exception as e:
            print(f"⚠️  Error adding VisionWebBrowserTool: {e}")

        # Add ContentRetrieverTool if available
        if CUSTOM_TOOLS_AVAILABLE:
            try:
                content_retriever = ContentRetrieverTool()
                content_tools.append(content_retriever)
                print("✅ Added ContentRetrieverTool to content processor")
            except Exception as e:
                print(f"⚠️  Error adding ContentRetrieverTool: {e}")

        # Add SpeechToTextTool if available        
        try:
            speech_tool = SpeechToTextTool()
            content_tools.append(speech_tool)
            print("✅ Added SpeechToTextTool to content processor")
        except ImportError:
            print("⚠️  SpeechToTextTool requires transformers - skipping")
        except Exception as e:
            print(f"⚠️  Error adding SpeechToTextTool: {e}")

        # Create content processor description
        content_tool_names = [getattr(tool, 'name', tool.__class__.__name__) for tool in content_tools]

        content_processor_description = f"""Multi-modal content acquisition and processing specialist.

        WORKFLOW: Navigate pages → Acquire content → Process content

        AVAILABLE TOOLS: {', '.join(content_tool_names)}

        CAPABILITIES:
        - Web navigation and content discovery
        - PDF, document, and media file acquisition  
        - Text extraction and content analysis
        - Audio transcription and processing

        STRATEGY: Use vision_web_browser for content acquisition, then appropriate processing tool"""

        content_processor = ToolCallingAgent(
            name="content_processor", 
            description=content_processor_description,
            tools=content_tools,
            model=self.specialist_model,
            max_steps=self.config.max_agent_steps,
            add_base_tools=True,
            logger=logger
        )

    # Create coordinator description
        # Return dict
        specialist_agents = {
            "data_analyst": data_analyst,
            "web_researcher": web_researcher,
            "content_processor": content_processor
        }
        
        print(f"🎯 Created {len(specialist_agents)} specialist agents:")
        print(f"   data_analyst: CodeAgent with direct Python file access")
        print(f"   web_researcher: ToolCallingAgent with web search tools")
        print(f"   content_processor: ToolCallingAgent with multi-modal content tools")
        
        return specialist_agents

    def _create_coordinator(self) -> CodeAgent:
        """Create coordinator using smolagents hierarchical pattern"""
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        
        # Define specialists
        specialist_agents = self._create_specialist_agents()
        
        # Coordinator gets hierarchy from managed_agents
        coordinator = CodeAgent(
            name="gaia_coordinator",
            description="""Coordinator that manages three specialist agents for GAIA tasks.

            MANAGED AGENTS:
            - data_analyst: For Excel/CSV analysis and calculations  
            - web_researcher: For web searches and information gathering
            - content_processor: For multi-modal content acquisition and processing

            WORKFLOW:
            1. Analyze the question and identify required capabilities
            2. Delegate to appropriate specialist agent(s) using their names
            3. Coordinate multiple specialists if needed
            4. Synthesize results into final answer
            """,
            tools=[],  # No direct tools - delegates to managed agents
            managed_agents=list(specialist_agents.values()),
            additional_authorized_imports=[
                "open", "codecs", "chardet",
                "os", "sys", "io", "pathlib",
                "mimetypes", "requests", "json", "zipfile"
            ],
            model=self.specialist_model,
            planning_interval=7,
            max_steps=12,
            logger=logger
        )
        
        return coordinator

    def _build_coordinator_task(self, state: GAIAState) -> str:
        """Build coordination task for hierarchical coordinator with managed specialists"""
        
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
        
        # Task for coordinator with managed specialists
        task = f"""
        You are the GAIA coordinator with three managed specialist agents. Analyze this question and execute using your specialists.

        QUESTION: {question}
        COMPLEXITY: {complexity}
        FILE INFO: {file_name} (path: {file_path}, has_file: {has_file})
        {examples_context}

        YOUR MANAGED SPECIALISTS:
        - data_analyst: CodeAgent for Excel/CSV analysis and calculations (direct file access)
        - web_researcher: ToolCallingAgent for web searches and current information  
        - content_processor: ToolCallingAgent for multi-modal content acquisition and processing

        ANALYSIS AND EXECUTION:

        1. PROBLEM ANALYSIS:
        Analyze the fundamental problem in this question:
        - Is this primarily a mathematical calculation?
        - Does it require current/recent information from the web?
        - Does it involve file processing or content acquisition?
        - What type of reasoning is needed?

        Consider the question: "{question}"

        2. PROCESSING APPROACH:
        Determine the appropriate processing approach:

        ```python
        # Analyze processing approach
        question_text = "{question}"
        has_attached_file = "{has_file}" == "True"

        if has_attached_file:
            # FILE-BASED ANALYSIS (served via URL in deployment)
            from pathlib import Path
            
            file_path = Path("{file_name}")
            extension = file_path.suffix.lower()
            
            if extension in ['.xlsx', '.csv', '.xls', '.tsv']:
                processing_approach = "data_analysis"
                best_specialist = "data_analyst"
                print(f"Data file detected → use data_analyst specialist")
            elif extension in ['.pdf', '.docx', '.doc', '.txt', '.mp3', '.mp4', '.wav']:
                processing_approach = "content_processing"
                best_specialist = "content_processor"
                print(f"Document/media file detected → use content_processor specialist")
            else:
                processing_approach = "content_processing"
                best_specialist = "content_processor"
                print(f"Unknown file type → use content_processor specialist")

        elif any(discovery_signal in question_text.lower() for discovery_signal in ["chapter", "section", "find", "locate", "download"]):
            # WEB CONTENT ANALYSIS
            # Consider if you need to visually explore this webpage to find the right content
            # Establish:
            # - Does this seem like a direct link to a document/file/image?
            # - Does the question suggest to navigate or discover content on the page?
            # - Is looking at the page the fastest path to problem solving?
            
            if any(indicator in question_text.lower() for indicator in [
                "chapter", "section", "part", "find", "locate", "mentioned", "discussed"
            ]) or not question_text.strip().endswith(('.pdf', '.doc', '.docx', '.txt')):
                processing_approach = "multimodal_web_content"
                best_specialist = "content_processor"
                strategy_hint = "Use vision browser to explore and acquire content, then process"
                print(f"Web exploration needed → use content_processor with vision browser")
            else:
                processing_approach = "direct_web_content"
                best_specialist = "content_processor"
                strategy_hint = "Try direct content processing first"
                print(f"Direct web content → use content_processor")
            
        else:
            # GENERAL INFORMATION QUERY
            processing_approach = "web_search"
            best_specialist = "web_researcher"
            print(f"General information query → use web_researcher specialist")

        print(f"Processing approach: {processing_approach} → {best_specialist}")
        ```

        3. SPECIALIST SELECTION AND EXECUTION:
        Based on your analysis, select the most appropriate specialist:

        ```python
        # Final specialist selection based on question requirements and processing approach
        question_lower = "{question}".lower()

        # Override specialist selection based on question type
        if "calculate" in question_lower or "data" in question_lower or "number" in question_lower:
            if processing_approach == "data_analysis":
                selected_specialist = "data_analyst"
                reasoning = "Data file + calculation requirements → data_analyst specialist"
            else:
                selected_specialist = "data_analyst"  
                reasoning = "Calculation requirements → data_analyst specialist"
        elif "current" in question_lower or "latest" in question_lower or "recent" in question_lower:
            selected_specialist = "web_researcher"
            reasoning = "Current information needed → web_researcher specialist"
        elif processing_approach in ["multimodal_web_content", "content_processing", "direct_web_content"]:
            selected_specialist = "content_processor"
            reasoning = f"Content processing needed → content_processor specialist"
        else:
            # Default to best specialist from processing analysis
            selected_specialist = best_specialist
            reasoning = f"Processing analysis → {best_specialist} specialist"

        print(f"SELECTED SPECIALIST: {{selected_specialist}}")
        print(f"REASONING: {{reasoning}}")
        ```

        4. EXECUTE WITH SELECTED SPECIALIST:
        Now execute the task using your selected specialist with appropriate strategy:

        SPECIALIST EXECUTION GUIDELINES:

        If using **data_analyst**:
        - You can directly access files using pandas: `pd.read_excel("{file_path}")` or `pd.read_csv("{file_path}")`
        - Use numerical computation libraries: numpy, scipy, statistics, math
        - Perform calculations and data analysis directly in Python

        If using **web_researcher**:
        - Search for current information using GoogleSearchTool
        - Use VisitWebpageTool to extract content from specific URLs
        - Use WikipediaSearchTool for factual information

        If using **content_processor**:
        - For attached files: Use ContentRetrieverTool for document extraction (files via additional_args)
        - For web content acquisition: Strategic guidance based on processing approach:
        * Multimodal web content: "Use vision browser to explore page and acquire content, then process with content retriever"
        * Direct web content: "Use content retriever directly if URL points to document"
        - Use SpeechToTextTool for audio transcription (files via additional_args)

        STRATEGIC GUIDANCE FOR CONTENT PROCESSOR:
        ```python
        if processing_approach == "multimodal_web_content":
            strategy_guidance = f"""
            CONTENT ACQUISITION STRATEGY:
            - Available tools: {', '.join(content_tool_names)}
            - Use vision_browser_tool for navigation and content discovery
            - Use vision_browser_tool to navigate and discover content
            - Look for download links, embedded content, or navigation requirements
            - Acquire content files or direct URLs
            - Use content_retriever_tool to process acquired content
            - Search for: {question}
            """
            print(f"Providing vision browser strategy to content_processor")
        else:
            strategy_guidance = "Use appropriate content processing tools based on content type"
        ```

        EXECUTE NOW: Use your selected specialist to answer the question: "{question}"

        CRITICAL OUTPUT REQUIREMENTS:
        - End your final response with 'FINAL ANSWER: [specific answer]'
        - Follow GAIA format: numbers (no commas), strings (no articles), lists (comma separated)
        - Provide actual answers, never use placeholder text like "[your answer]"
        - Be specific and factual in your response
        """
        
        return task
                   
    # ============================================================================
    # LANGGRAPH WORKFLOW 
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
        """Enhanced complexity check with file awareness"""
        question = state["question"]
        task_id = state.get("task_id", "")
        has_file = state.get("has_file", False)
        
        track("Analyzing question complexity", self.config)
        
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
        
        track(f"Complexity: {complexity} - {reason}", self.config)
        
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

    def _coordinator_node(self, state: GAIAState) -> GAIAState:
        """🔥 CORRECTED: Hierarchical coordinator node - Analysis and execution in one step"""
        print(f"🧠 Coordinator: {state['question'][:80]}...")
        
        track("Starting coordinator", self.config)
        
        if self.config.enable_context_bridge:
            ContextBridge.track_operation("Starting coordinator")
        
        if self.logging:
            self.logging.log_step("coordinator_start", "Starting coordinator analysis and execution")
        
        try:
            # Create fresh coordinator with managed specialists for this task
            self.coordinator = self._create_coordinator()
            
            # Build integrated coordination task (analysis + execution)
            coordination_task = self._build_coordinator_task(state)
            
            if self.config.enable_context_bridge:
                ContextBridge.track_operation("Executing coordinator with managed specialists")
            
            #  Give coordinator file access
            file_path = state.get("file_path", "")
            file_name = state.get("file_name", "")
            has_image = any(ext in file_name.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']) if file_name else False
            
            if file_path and has_image:
                # Load image for vision processing
                image = self._load_image_for_agent(file_path)
                
                if image:
                    print("🖼️ Running coordinator with vision support")
                    # Pass image to coordinator using SmolagAgents vision pattern
                    coordination_result = self.coordinator.run(
                        coordination_task,
                        images=[image],  # 🔥 KEY: Pass images list
                        additional_args={"file_path": file_path}
                    )
                else:
                    print("⚠️ Image loading failed, proceeding without vision")
                    coordination_result = self.coordinator.run(
                        coordination_task,
                        additional_args={"file_path": file_path}
                    )
            elif file_path:
                # Non-image file processing
                coordination_result = self.coordinator.run(
                    coordination_task,
                    additional_args={"file_path": file_path}
                )
            else:
                # No file needed
                coordination_result = self.coordinator.run(coordination_task)
            
            track("Hierarchical coordinator completed analysis and execution", self.config)
            
            print(f"✅ Hierarchical coordinator completed analysis and execution")
            
            if self.logging:
                self.logging.log_step("coordinator_complete", "Hierarchical coordinator analysis and execution completed")
            
            # Return with execution results (not just analysis)
            return {
                **state,
                "agent_used": "hierarchical_coordinator",
                "raw_answer": coordination_result,
                "execution_successful": True,
                "steps": state["steps"] + ["Hierarchical coordinator analysis and execution completed"]
            }
            
        except Exception as e:
            error_msg = f"Coordinator failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            if self.config.enable_context_bridge:
                ContextBridge.track_error(error_msg)
            
            if self.logging:
                self.logging.log_step("coordinator_error", error_msg)
            
            return {
                **state,
                "current_error": error_msg,
                "agent_used": "hierarchical_coordinator",
                "raw_answer": error_msg,
                "execution_successful": False,
                "steps": state["steps"] + [f"Hierarchical coordinator failed: {error_msg}"]
            }

    def _format_answer_node(self, state: GAIAState):
        """Enhanced answer formatting with question context"""
        raw_answer = state.get("raw_answer", "")
        task_id = state.get("task_id", "")
        question = state.get("question", "")
        
        # Store question for formatting context
        self._current_question = question
        
        track("Formatting final answer", self.config)
        
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
                track(f"Final answer: {formatted_answer}", self.config)
            
            return {
                "final_answer": formatted_answer,
                "execution_metrics": execution_metrics,
                "steps": state["steps"] + ["Final answer formatting applied"]
            }
            
        except Exception as e:
            error_msg = f"Answer formatting error: {str(e)}"
            
            track(f"Format error: {error_msg}", self.config)
            execution_metrics = ContextBridge.get_execution_metrics() if self.config.enable_context_bridge else {}
            
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
        """GAIA formatting with safe regex operations"""
        if not answer:
            return "No answer"
        
        original_answer = answer
        answer = answer.strip()
        
        if self.logging:
            self.logging.log_step("gaia_format_start", f"Original answer: '{answer}'")
        
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
                    if self.logging:
                        self.logging.log_step("gaia_format_extract", f"Extracted: '{answer}'")
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
            # SAFE: Remove articles
            answer = safe_regex_sub(r'\b(the|a|an)\s+', '', answer, re.IGNORECASE)
            
            # SAFE: Remove units (carefully)
            if not any(keep_unit in question for keep_unit in ["$", "%", "units", "meters"]):
                answer = safe_regex_sub(r'[^\w\s,.-]', '', answer)
        
        # SAFE: Extract numbers for specific question types
        if "how many" in question:
            numbers = safe_regex_findall(r'\b\d+\b', answer)
            if numbers:
                answer = numbers[0]
        
        # Clean whitespace
        answer = ' '.join(answer.split())
        
        if self.logging:
            self.logging.log_step("gaia_format_final", f"Final: '{answer}'")
        
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