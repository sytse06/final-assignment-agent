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

# Import file handling utilities
from agent_files import (
    smart_file_path_selection,
    analyze_file_metadata
)

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

    def _validate_content_processor_tools(self, content_tools):
        """Streamlined validation - check global flags instead of tool inspection"""
        
        # Skip tool inspection, use global availability flags
        has_vision_browser = VISION_BROWSER_AVAILABLE
        has_content_retriever = CUSTOM_TOOLS_AVAILABLE  
        has_speech_tool = True  # SpeechToText usually works
        
        # Model capabilities
        model_capabilities = self._check_model_vision_capabilities()
        
        # Simple capabilities assessment
        capabilities = {
            "vision_navigation": has_vision_browser,
            "content_extraction": has_content_retriever, 
            "audio_processing": has_speech_tool,
            "total_tools": len(content_tools),
            "model_supports_vision": model_capabilities["has_vision"],
            "effective_vision_capability": has_vision_browser and model_capabilities["has_vision"],
        }
        
        # Clean status report
        print(f"📊 Content Processor: {len(content_tools)} tools")
        print(f"   Vision: {'✅' if capabilities['effective_vision_capability'] else '⚠️ OCR Only'}")
        print(f"   Model: {model_capabilities['model_type']}")
        
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
        """Enhanced file analysis using agent_files utilities"""
        if not file_name:
            return {"no_file": True}
        
        try:
            return analyze_file_metadata(file_name, file_path, self.capabilities)
        except Exception as e:
            return {
                "file_name": file_name,
                "file_path": file_path,
                "category": "unknown",
                "processing_approach": "content_extraction",
                "recommended_specialist": "content_processor",
                "error": str(e),
                "file_exists": os.path.exists(file_path) if file_path else False
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
            
            # 2. Web Researcher - Focused on search and discovery
            try:
                from tools import get_web_researcher_tools
                web_tools = get_web_researcher_tools()
            except ImportError:
                print("⚠️ tools module unavailable - using fallback")
                web_tools = []
                try:
                    from smolagents import VisitWebpageTool, WikipediaSearchTool
                    web_tools.extend([VisitWebpageTool(), WikipediaSearchTool()])
                except Exception as e:
                    print(f"⚠️ Fallback tools failed: {e}")

            web_researcher = ToolCallingAgent(
                name="web_researcher",
                description="Search web, find information sources, verify accessibility",
                tools=web_tools,
                model=self.specialist_model,
                max_steps=self.config.max_agent_steps,
                add_base_tools=len(web_tools) == 0,
                logger=logger
            )
            
            # 3. Content Processor - Deep content analysis
            try:
                from tools import get_content_processor_tools
                content_tools = get_content_processor_tools()
            except ImportError:
                print("⚠️ tools module unavailable - no specialized tools")
                content_tools = []

            content_processor = ToolCallingAgent(
                name="content_processor", 
                description="Extract and process content from documents, videos, audio, and complex sources",
                tools=content_tools,
                model=self.specialist_model,
                max_steps=self.config.max_agent_steps,
                add_base_tools=True,
                logger=logger
            )

            # Return specialist agents
            specialist_agents = {
                "data_analyst": data_analyst,
                "web_researcher": web_researcher,
                "content_processor": content_processor
            }
            
            print(f"🎯 Created {len(specialist_agents)} specialist agents:")
            print(f"   📊 data_analyst: {len(data_analyst.additional_authorized_imports)} imports")
            print(f"   🔍 web_researcher: {len(web_tools)} tools")
            print(f"   📱 content_processor: {len(content_tools)} tools")
            
            return specialist_agents
        
    def _create_coordinator(self) -> CodeAgent:
            """Create coordinator using smolagents hierarchical pattern with file-first strategy"""
            logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
            
            # Define specialists
            specialist_agents = self._create_specialist_agents()
            
            # Enhanced coordinator with direct file access via CodeAgent imports
            coordinator = CodeAgent(
                name="gaia_coordinator",
                description="""GAIA task coordinator managing three specialist agents.

        WORKFLOW:

        1. ANALYZE TASK: Question analysis and file preprocessing (extract archives, inspect contents)
        2. CHOOSE SPECIALIST: Based on question type and available files/data
        3. DELEGATE: Assign complete task to appropriate specialist(s)
        4. SYNTHESIZE: Combine results and format GAIA answer

        SPECIALISTS AVAILABLE:
        - data_analyst: Numerical analysis, calculations, data processing
        - web_researcher: Search and discovery, online information gathering  
        - content_processor: Document/media processing, content extraction

        The coordinator handles file preparation so specialists can focus on their core capabilities.""",
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
                planning_interval=5,  # More frequent planning for file-first workflow
                max_steps=15,  # Extra steps for file access + coordination
                logger=logger
            )
            
            return coordinator

    def _build_coordinator_task(self, state: GAIAState) -> str:
        """🔥 ENHANCED: Build coordination task with smart file information"""
        question = state["question"]
        similar_examples = state.get("similar_examples", [])
        complexity = state.get("complexity", "unknown")
        has_file = state.get("has_file", False)
        file_name = state.get("file_name", "")
        file_path = state.get("file_path", "")
        file_metadata = state.get("file_metadata", {})
        
        # Build enhanced file context with processing guidance
        file_context = ""
        if has_file:
            # Determine best file reference and provide processing guidance
            if file_name and file_name.strip() and os.path.exists(file_name):
                # Filename exists and has extension
                file_context = f"\nFILE AVAILABLE: {file_name}"
                if file_metadata:
                    file_type = file_metadata.get("category", "unknown")
                    recommended_specialist = file_metadata.get("recommended_specialist", "content_processor")
                    file_context += f"\nFILE TYPE: {file_type}"
                    file_context += f"\nRECOMMENDED PROCESSING: {recommended_specialist}"
                    
                    # Add specific processing guidance
                    if file_type == "data":
                        file_context += "\nGUIDANCE: Use data_analyst for CSV/Excel analysis with pandas"
                    elif file_type == "document":
                        file_context += "\nGUIDANCE: Use content_processor with file_path from additional_args"
                    elif file_type == "image":
                        file_context += "\nGUIDANCE: Use content_processor with vision capabilities"
                    elif file_type == "media":
                        file_context += "\nGUIDANCE: Use content_processor for audio/video transcription"
                
            elif file_path and file_path.strip() and os.path.exists(file_path):
                # Cache path exists but may lack extension
                file_context = f"\nFILE AVAILABLE: {file_path}"
                if file_name:
                    file_context += f" (original name: {file_name})"
                file_context += "\nGUIDANCE: Pass file_path to specialist - ContentRetrieverTool will handle extension detection"
                
            else:
                file_context = "\nFILE: Referenced but not accessible - proceed without file"
        
        # Build examples context
        examples_context = ""
        if similar_examples:
            examples_context = "\nSIMILAR SUCCESSFUL EXAMPLES:\n"
            for i, example in enumerate(similar_examples[:2], 1):
                examples_context += f"{i}. {example.get('question', '')[:80]}... → {example.get('answer', '')}\n"
        
        # Core coordination task with enhanced file handling instructions
        task = f"""GAIA Coordinator Task

    QUESTION: {question}
    COMPLEXITY: {complexity}{file_context}{examples_context}

    YOUR SPECIALISTS:
    - data_analyst: CodeAgent with pandas/numpy for data analysis and calculations
    - web_researcher: ToolCallingAgent for web search and information gathering  
    - content_processor: ToolCallingAgent with ContentRetrieverTool for document/media processing

    COORDINATOR RESPONSIBILITIES:
    1. ANALYZE FILES: If file provided, inspect and preprocess as needed
    2. ROUTE INTELLIGENTLY: Choose appropriate specialist based on task and file type
    3. DELEGATE CLEARLY: Provide specialist with file_path and clear instructions
    4. SYNTHESIZE RESULTS: Combine specialist outputs into final answer

    FILE HANDLING STRATEGY:
    - Simple data files (CSV, JSON): Consider processing directly with pandas/json
    - Complex documents (PDF, DOCX): Delegate to content_processor with file_path
    - Archives (ZIP, TAR): Extract contents first, then process individual files
    - Images/Media: Delegate to content_processor with appropriate tools

    SPECIALIST DELEGATION:
    When delegating to content_processor, use: "Process the file at file_path from additional_args"
    When delegating to data_analyst, provide clear data processing instructions
    When delegating to web_researcher, provide specific search queries

    CRITICAL REQUIREMENTS:
    - End with: FINAL ANSWER: [specific answer]
    - Format: numbers (no commas), strings (no articles), lists (comma-separated)
    - Be precise and factual

    Execute now."""

        return task.strip()
                   
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

    def _coordinator_node(self, state: GAIAState) -> GAIAState:
        """🔥 ENHANCED: Hierarchical coordinator with comprehensive logging and tracking"""
        task_id = state.get("task_id", "unknown")
        question = state["question"]
        
        print(f"🧠 Coordinator: {question[:80]}...")
        
        # Enhanced tracking and logging
        track("Starting coordinator", self.config)
        
        if self.config.enable_context_bridge:
            ContextBridge.track_operation("Starting coordinator")
        
        if self.logging:
            self.logging.log_step("coordinator_start", "Starting coordinator analysis and execution", {
                'task_id': task_id,
                'node_name': 'coordinator',
                'question_preview': question[:100]
            })
            self.logging.set_routing_path("hierarchical_coordinator")
        
        try:
            # Create fresh coordinator with managed specialists for this task
            track("Creating coordinator with managed specialists", self.config)
            self.coordinator = self._create_coordinator()
            
            if self.logging:
                self.logging.log_step("coordinator_created", "Coordinator with specialists created", {
                    'specialist_count': len(self.coordinator.managed_agents) if hasattr(self.coordinator, 'managed_agents') else 0
                })
            
            # File path analysis and selection
            file_path = state.get("file_path", "")
            file_name = state.get("file_name", "")
            has_file = state.get("has_file", False)
            file_metadata = state.get("file_metadata", {})
            
            # 🔧 USE agent_files utility for smart file path selection
            best_file_path = ""
            file_selection_method = "none"
            
            if has_file:
                track("Analyzing file information for coordinator", self.config)
                best_file_path = smart_file_path_selection(file_name, file_path) if has_file else ""
                
                if best_file_path == file_name:
                    file_selection_method = "filename_with_extension"
                elif best_file_path == file_path:
                    file_selection_method = "cache_path_fallback"
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
            track("Building coordination task with file context", self.config)
            coordination_task = self._build_coordinator_task(state)
            
            if self.logging:
                self.logging.log_step("task_built", "Coordination task constructed", {
                    'task_length': len(coordination_task),
                    'has_file_context': has_file,
                    'similar_examples_count': len(state.get("similar_examples", []))
                })
            
            # Context bridge tracking for execution phase
            if self.config.enable_context_bridge:
                ContextBridge.track_operation("Executing coordinator with managed specialists")
            
            # Determine execution strategy based on file type
            has_image = any(ext in file_name.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']) if file_name else False
            execution_strategy = "no_file"
            
            if best_file_path and has_image:
                execution_strategy = "image_vision_processing"
            elif best_file_path:
                execution_strategy = "file_processing"
            
            if self.logging:
                self.logging.log_step("execution_strategy", f"Execution strategy: {execution_strategy}", {
                    'strategy': execution_strategy,
                    'has_file': bool(best_file_path),
                    'is_image': has_image,
                    'file_type': file_metadata.get("category", "unknown") if file_metadata else "unknown"
                })
            
            # Execute coordination based on strategy
            coordination_result = None
            
            if execution_strategy == "image_vision_processing":
                track("Loading image for vision processing", self.config)
                image = self._load_image_for_agent(best_file_path)
                
                if image:
                    print("🖼️ Running coordinator with vision support")
                    track("Executing coordinator with vision support", self.config)
                    
                    if self.logging:
                        self.logging.log_step("vision_execution", "Executing with vision support", {
                            'image_loaded': True,
                            'file_path': best_file_path
                        })
                    
                    coordination_result = self.coordinator.run(
                        coordination_task,
                        images=[image],
                        additional_args={"file_path": best_file_path, "file_name": file_name}
                    )
                else:
                    print("⚠️ Image loading failed, proceeding without vision")
                    track("Image loading failed, fallback to file processing", self.config)
                    
                    if self.logging:
                        self.logging.log_step("vision_fallback", "Image loading failed, proceeding without vision", {
                            'image_loaded': False,
                            'fallback_strategy': 'file_processing'
                        })
                    
                    coordination_result = self.coordinator.run(
                        coordination_task,
                        additional_args={"file_path": best_file_path, "file_name": file_name}
                    )
                    
            elif execution_strategy == "file_processing":
                print(f"📄 Running coordinator with file: {os.path.basename(best_file_path)}")
                track("Executing coordinator with file processing", self.config)
                
                if self.logging:
                    self.logging.log_step("file_execution", "Executing with file processing", {
                        'file_basename': os.path.basename(best_file_path),
                        'file_size': os.path.getsize(best_file_path) if os.path.exists(best_file_path) else 0
                    })
                
                coordination_result = self.coordinator.run(
                    coordination_task,
                    additional_args={"file_path": best_file_path, "file_name": file_name}
                )
                
            else:  # no_file strategy
                print("📝 Running coordinator without files")
                track("Executing coordinator without files", self.config)
                
                if self.logging:
                    self.logging.log_step("no_file_execution", "Executing without files", {
                        'text_only': True
                    })
                
                coordination_result = self.coordinator.run(coordination_task)
            
            # Post-execution tracking and logging
            track("Hierarchical coordinator completed analysis and execution", self.config)
            
            if self.config.enable_context_bridge:
                ContextBridge.track_operation("Coordinator execution completed successfully")
            
            print(f"✅ Hierarchical coordinator completed: {execution_strategy}")
            
            if self.logging:
                self.logging.log_step("coordinator_complete", "Hierarchical coordinator execution completed", {
                    'execution_strategy': execution_strategy,
                    'result_length': len(str(coordination_result)),
                    'success': True
                })
            
            # Return enhanced state with execution details
            return {
                **state,
                "agent_used": "hierarchical_coordinator",
                "raw_answer": coordination_result,
                "execution_successful": True,
                "execution_strategy": execution_strategy,
                "file_selection_method": file_selection_method,
                "steps": state["steps"] + [f"Hierarchical coordinator completed ({execution_strategy})"]
            }
            
        except Exception as e:
            error_msg = f"Coordinator failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Enhanced error tracking and logging
            track(f"Coordinator error: {str(e)}", self.config)
            
            if self.config.enable_context_bridge:
                ContextBridge.track_error(error_msg)
            
            if self.logging:
                self.logging.log_step("coordinator_error", error_msg, {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'task_id': task_id,
                    'execution_strategy': locals().get('execution_strategy', 'unknown'),
                    'file_selection_method': locals().get('file_selection_method', 'unknown')
                })
            
            return {
                **state,
                "current_error": error_msg,
                "agent_used": "hierarchical_coordinator",
                "raw_answer": error_msg,
                "execution_successful": False,
                "execution_strategy": locals().get('execution_strategy', 'failed'),
                "file_selection_method": locals().get('file_selection_method', 'failed'),
                "steps": state["steps"] + [f"Hierarchical coordinator failed: {error_msg}"]
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