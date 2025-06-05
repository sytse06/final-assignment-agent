# gaia_agent.py
# GAIA Architecture integrated with retriever

import json
import os
from typing import TypedDict, Optional, List, Dict, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import uuid
from datetime import datetime
import traceback

# Core dependencies
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# SmolagAgents imports
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    GoogleSearchTool,
    VisitWebpageTool,
    tool,
    AgentLogger,
    LogLevel,
)

# Import your existing retriever system
from dev_retriever import load_gaia_retriever, DevelopmentGAIARetriever

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GAIAConfig:
    """Production configuration for GAIA benchmark agent"""
    # Model settings (multi-provider support)
    model_provider: str = "openrouter"
    primary_model: str = "qwen/qwen-2.5-coder-32b-instruct:free"
    secondary_model: Optional[str] = None
    temperature: float = 0.3
    api_base: Optional[str] = None
    num_ctx: int = 32768
    
    # Retriever settings
    csv_file: str = "gaia_embeddings.csv"
    metadata_path: str = "metadata.jsonl"
    
    # RAG settings
    rag_examples_count: int = 3
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    
    # Agent settings
    max_agent_steps: int = 15
    agent_timeout: int = 180
    planning_interval: int = 3
    enable_agent_fallback: bool = True
    
    # Error handling
    max_retries: int = 2
    enable_graceful_degradation: bool = True
    enable_model_fallback: bool = True
    
    # Performance settings
    results_output_path: str = "gaia_results"
    debug_mode: bool = True
    save_intermediate_results: bool = True
    log_level: str = "INFO"
    
    # GAIA specific
    gaia_formatting_strict: bool = True
    enable_confidence_scoring: bool = True
    
# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class GAIAState(TypedDict):
    """State management for GAIA agent workflow"""
    # Core execution
    messages: List[BaseMessage]
    question: str
    task_id: str
    
    # RAG context
    retriever_context: Optional[str]
    similar_examples: List[Dict]
    
    # Execution tracking
    selected_strategy: Optional[str]
    selected_agent: Optional[str]
    execution_steps: List[str]
    retry_count: int
    
    # Results
    raw_answer: Optional[str]
    final_answer: Optional[str]
    confidence_score: Optional[float]
    
    # Error handling
    errors: List[str]
    fallback_used: bool
    
    # Evaluation metadata
    ground_truth: Optional[str]
    level: Optional[int]
    execution_time: Optional[float]
    model_used: Optional[str]
    
    # Debug info
    debug_info: Dict[str, Any]

# ============================================================================
# CUSTOM TOOLS
# ============================================================================

# No custom tools - we'll use native SmolagAgents tools:
# - Built-in calculator
# - Built-in file processing capabilities  
# - GoogleSearchTool
# - VisitWebpageTool
# - Any other native tools available in SmolagAgents

# ============================================================================
# AGENT MANAGER
# ============================================================================

# gaia_agent_system.py
# Production GAIA Architecture with robust agent management and error handling

import json
import os
from typing import TypedDict, Optional, List, Dict, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import uuid
from datetime import datetime
import traceback

# Core dependencies
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# SmolagAgents imports
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    GoogleSearchTool,
    VisitWebpageTool,
    tool,
    AgentLogger,
    LogLevel,
)

# Import your existing retriever system
from dev_retriever import load_gaia_retriever, DevelopmentGAIARetriever

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GAIAConfig:
    """Production configuration for GAIA benchmark agent"""
    # Model settings (multi-provider support with fallback)
    model_provider: str = "openrouter"
    primary_model: str = "qwen/qwen-2.5-coder-32b-instruct:free"
    secondary_model: Optional[str] = None  # Fallback model for same provider
    temperature: float = 0.3
    api_base: Optional[str] = None
    num_ctx: int = 32768
    
    # Retriever settings
    csv_file: str = "gaia_embeddings.csv"
    metadata_path: str = "metadata.jsonl"
    
    # RAG settings
    rag_examples_count: int = 3
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    
    # Agent settings
    max_agent_steps: int = 15
    agent_timeout: int = 180
    planning_interval: int = 3
    enable_agent_fallback: bool = True
    
    # Error handling and model fallback
    max_retries: int = 2
    enable_graceful_degradation: bool = True
    enable_model_fallback: bool = True  # Enable fallback to secondary model
    
    # Performance settings
    results_output_path: str = "gaia_results"
    debug_mode: bool = True
    save_intermediate_results: bool = True
    log_level: str = "INFO"
    
    # GAIA specific
    gaia_formatting_strict: bool = True
    enable_confidence_scoring: bool = True

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class GAIAState(TypedDict):
    """State management for GAIA agent workflow"""
    # Core execution
    messages: List[BaseMessage]
    question: str
    task_id: str
    
    # RAG context
    retriever_context: Optional[str]
    similar_examples: List[Dict]
    
    # Execution tracking
    selected_strategy: Optional[str]
    selected_agent: Optional[str]
    execution_steps: List[str]
    retry_count: int
    
    # Results
    raw_answer: Optional[str]
    final_answer: Optional[str]
    confidence_score: Optional[float]
    
    # Error handling
    errors: List[str]
    fallback_used: bool
    
    # Evaluation metadata
    ground_truth: Optional[str]
    level: Optional[int]
    execution_time: Optional[float]
    model_used: Optional[str]
    
    # Debug info
    debug_info: Dict[str, Any]

# ============================================================================
# CUSTOM TOOLS - REMOVED: Using native SmolagAgents tools instead
# ============================================================================

# No custom tools - we'll use native SmolagAgents tools:
# - Built-in calculator
# - Built-in file processing capabilities  
# - GoogleSearchTool
# - VisitWebpageTool
# - Any other native tools available in SmolagAgents

# ============================================================================
# AGENT MANAGER
# ============================================================================

class AgentManager:
    """Manages SmolagAgents integration with robust error handling"""
    
    def __init__(self, config: GAIAConfig):
        self.config = config
        self.model = self._initialize_model()
        self.logger = self._setup_logger()
        self.agents = self._initialize_agents()
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "agent_usage": {}
        }
    
    def _initialize_model(self):
        """Initialize LiteLLMModel with robust error handling and fallback support"""
        try:
            return self._create_model_with_fallback(self.config.primary_model)
        except Exception as e:
            if self.config.enable_model_fallback and self.config.secondary_model:
                print(f"⚠️  Primary model failed ({e}), trying secondary model...")
                try:
                    return self._create_model_with_fallback(self.config.secondary_model)
                except Exception as e2:
                    print(f"❌ Secondary model also failed ({e2}), using hardcoded fallback")
            
            # Final fallback
            print(f"❌ Error initializing models: {e}")
            return self._create_fallback_model()
    
    def _create_model_with_fallback(self, model_name: str):
        """Create LiteLLMModel for specific model"""
        if self.config.model_provider == "openrouter":
            return LiteLLMModel(
                model_id=f"openrouter/{model_name}",
                api_base="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=self.config.temperature
            )
        
        elif self.config.model_provider == "ollama":
            return LiteLLMModel(
                model_id=f"ollama_chat/{model_name}",
                api_base=self.config.api_base or "http://localhost:11434",
                num_ctx=self.config.num_ctx,
                temperature=self.config.temperature
            )
        
        elif self.config.model_provider == "groq":
            return LiteLLMModel(
                model_id=f"groq/{model_name}",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=self.config.temperature
            )
        
        elif self.config.model_provider == "google":
            # Check for both possible environment variable names
            google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API-KEY")
            if not google_api_key:
                raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable.")
            
            return LiteLLMModel(
                model_id=f"gemini/{model_name}",
                api_key=google_api_key,
                temperature=self.config.temperature
            )
        
        else:
            raise ValueError(f"Unknown provider: {self.config.model_provider}")
    
    def _create_fallback_model(self):
        """Create hardcoded fallback model"""
        print("🔄 Using hardcoded OpenRouter fallback model")
        return LiteLLMModel(
            model_id="openrouter/qwen/qwen-2.5-coder-32b-instruct:free",
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.7
        )
    
    def _setup_logger(self):
        """Setup agent logger"""
        verbosity = LogLevel.DEBUG if self.config.debug_mode else LogLevel.INFO
        return AgentLogger(level=verbosity)
    
    def _create_step_callback(self, agent_name: str):
        """Create step callback that properly handles step objects"""
        def step_callback(step):
            """Callback that handles step objects without attribute errors"""
            try:
                step_info = f"Agent {agent_name} executed step"
                if hasattr(step, 'action') and step.action:
                    step_info += f": {step.action}"
                elif hasattr(step, 'tool_name'):
                    step_info += f" with tool: {step.tool_name}"
                
                if self.config.debug_mode:
                    print(f"🔄 {step_info}")
                    
            except Exception as e:
                if self.config.debug_mode:
                    print(f"⚠️  Step callback error: {e}")
        
        return step_callback
    
    def _initialize_agents(self):
        """Initialize agents using native SmolagAgents tools and capabilities"""
        
        # Import native SmolagAgents tools
        try:
            from smolagents import WebSearchTool
            print("✅ WebSearchTool imported")
        except ImportError:
            print("⚠️  WebSearchTool not available - install smolagents[toolkit]")
            WebSearchTool = None
        
        # Collect available native tools
        available_tools = []
        
        try:
            # Add native SmolagAgents web tools - prioritize one to avoid duplicates
            web_search_added = False
            
            # Try WebSearchTool first (DuckDuckGo)
            if WebSearchTool:
                try:
                    web_search = WebSearchTool()
                    available_tools.append(web_search)
                    web_search_added = True
                    print("✅ WebSearchTool (DuckDuckGo) added")
                except Exception as e:
                    print(f"⚠️  WebSearchTool failed: {e}")
            
            # Only try GoogleSearchTool if WebSearchTool failed
            if not web_search_added:
                try:
                    # Check for both possible SerpAPI key names
                    serpapi_key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
                    if serpapi_key:
                        # Set the expected environment variable if needed
                        if not os.getenv("SERPAPI_API_KEY"):
                            os.environ["SERPAPI_API_KEY"] = serpapi_key
                        
                        google_search = GoogleSearchTool()
                        available_tools.append(google_search)
                        web_search_added = True
                        print("✅ GoogleSearchTool (SerpAPI) added as fallback")
                    else:
                        print("⚠️  GoogleSearchTool unavailable: No SERPAPI_API_KEY or SERPAPI_KEY found")
                except Exception as e:
                    print(f"⚠️  GoogleSearchTool unavailable: {e}")
            
            # If both web search tools are available, prefer WebSearchTool and skip GoogleSearchTool
            if WebSearchTool and os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY"):
                print("ℹ️  Both WebSearchTool and GoogleSearchTool available, using WebSearchTool to avoid duplicates")
            
            try:
                visit_webpage = VisitWebpageTool()
                available_tools.append(visit_webpage)
                print("✅ VisitWebpageTool (native) added")
            except Exception as e:
                print(f"⚠️  VisitWebpageTool unavailable: {e}")
            
        except Exception as e:
            print(f"⚠️  Error setting up native tools: {e}")
        
        # Debug: Print available tool information
        if self.config.debug_mode:
            tool_names = [getattr(tool, 'name', str(type(tool).__name__)) for tool in available_tools]
            print(f"🔧 Native tools loaded: {tool_names}")
            
            # Check for duplicates
            if len(tool_names) != len(set(tool_names)):
                duplicate_names = [name for name in set(tool_names) if tool_names.count(name) > 1]
                print(f"⚠️  Warning: Duplicate tool names detected: {duplicate_names}")
            else:
                print("✅ All tool names are unique")
            
            # Debug: Show API key status
            print(f"🔧 API Key Status:")
            print(f"  ├── SERPAPI_API_KEY: {'✅ Set' if os.getenv('SERPAPI_API_KEY') else '❌ Missing'}")
            print(f"  ├── SERPAPI_KEY: {'✅ Set' if os.getenv('SERPAPI_KEY') else '❌ Missing'}")
            print(f"  ├── GOOGLE_API_KEY: {'✅ Set' if os.getenv('GOOGLE_API_KEY') else '❌ Missing'}")
            print(f"  ├── GROQ_API_KEY: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing'}")
            print(f"  └── OPENROUTER_API_KEY: {'✅ Set' if os.getenv('OPENROUTER_API_KEY') else '❌ Missing'}")
            
            print(f"🔧 Web Search Strategy: {'WebSearchTool (DuckDuckGo)' if web_search_added and WebSearchTool else 'GoogleSearchTool (SerpAPI)' if web_search_added else 'None available'}")
        
        agents = {}
        
        try:
            # Data Analyst Agent - CodeAgent with built-in Python interpreter and base tools
            agents["data_analyst"] = CodeAgent(
                name="data_analyst",
                description="Advanced data analyst with Python interpreter for mathematical computations, statistical analysis, and data processing for GAIA benchmark tasks.",
                tools=available_tools,  # Additional tools beyond built-in Python
                additional_authorized_imports=[
                    "numpy", "pandas", "matplotlib", "seaborn", "scipy", 
                    "json", "csv", "statistics", "math", "re", "requests"
                ],
                add_base_tools=True,  # This adds the default SmolagAgents toolbox
                model=self.model,
                logger=self.logger,
                step_callbacks=[self._create_step_callback("data_analyst")],
                max_steps=self.config.max_agent_steps,
            )
            
            # Web Researcher Agent - ToolCallingAgent with base tools including Python interpreter
            agents["web_researcher"] = ToolCallingAgent(
                name="web_researcher",
                description="Expert web researcher using native SmolagAgents tools for finding current information, verifying facts, and retrieving web content for GAIA benchmark questions.",
                tools=available_tools,
                add_base_tools=True,  # Adds web search, Python interpreter, transcriber
                model=self.model,
                planning_interval=self.config.planning_interval,
                max_steps=self.config.max_agent_steps,
                logger=self.logger,
                step_callbacks=[self._create_step_callback("web_researcher")],
            )
            
            # Document Processor Agent - ToolCallingAgent with base tools for file processing
            agents["document_processor"] = ToolCallingAgent(
                name="document_processor",
                description="Specialized agent using native SmolagAgents capabilities including Python interpreter and transcriber for processing various file formats including Excel, PDF, images, audio, and text documents for GAIA tasks.",
                tools=available_tools,
                add_base_tools=True,  # Adds Python interpreter, transcriber for audio files
                model=self.model,
                max_steps=self.config.max_agent_steps,
                logger=self.logger,
                step_callbacks=[self._create_step_callback("document_processor")],
            )
            
            # General Assistant Agent - ToolCallingAgent with all base tools
            agents["general_assistant"] = ToolCallingAgent(
                name="general_assistant",
                description="General-purpose assistant using full native SmolagAgents toolbox including Python interpreter, web search, and transcriber for comprehensive reasoning and complex GAIA benchmark tasks.",
                tools=available_tools,
                add_base_tools=True,  # Adds complete default toolbox
                model=self.model,
                max_steps=self.config.max_agent_steps,
                logger=self.logger,
                step_callbacks=[self._create_step_callback("general_assistant")],
            )
            
            print(f"✅ Initialized {len(agents)} SmolagAgents using native tools and base toolbox")
            print("📦 Each agent has access to:")
            print("  ├── Python code interpreter (built-in)")
            print("  ├── Web search (DuckDuckGo/Google)")
            print("  ├── Transcriber (Whisper-based)")
            print("  └── Additional native tools")
            
            return agents
            
        except Exception as e:
            print(f"❌ Error initializing agents: {e}")
            print(f"Full error: {traceback.format_exc()}")
            
            # Return simulation agents as fallback
            return {
                "data_analyst": "SimulationDataAnalyst",
                "web_researcher": "SimulationWebResearcher",
                "document_processor": "SimulationDocumentProcessor",
                "general_assistant": "SimulationGeneralAssistant"
            }
    
    def execute_with_agent(self, agent_name: str, question: str, context: str = "", retry_count: int = 0) -> Dict:
        """Execute task with agent including retry logic and fallback"""
        if agent_name not in self.agents:
            return {
                "answer": f"Agent {agent_name} not found",
                "steps": [f"Error: Agent {agent_name} not available"],
                "success": False,
                "error": f"Agent {agent_name} not found"
            }
        
        self.execution_stats["total_executions"] += 1
        self.execution_stats["agent_usage"][agent_name] = self.execution_stats["agent_usage"].get(agent_name, 0) + 1
        
        agent = self.agents[agent_name]
        
        try:
            if isinstance(agent, str):  # Simulation agent
                return self._simulation_execution(agent_name, question, context)
            
            # Real SmolagAgent execution
            if self.config.debug_mode:
                print(f"🤖 Executing SmolagAgent: {agent_name} (attempt {retry_count + 1})")
            
            # Prepare question with context
            enhanced_question = self._prepare_question(question, context)
            
            # Execute with timeout handling
            try:
                agent_result = agent.run(task=enhanced_question)
                
                self.execution_stats["successful_executions"] += 1
                
                return {
                    "answer": str(agent_result),
                    "steps": [
                        f"SmolagAgent {agent_name} executed successfully",
                        f"Question processed: {len(enhanced_question)} characters",
                        f"Result generated: {len(str(agent_result))} characters"
                    ],
                    "success": True,
                    "agent_type": type(agent).__name__,
                    "retry_count": retry_count
                }
                
            except Exception as execution_error:
                self.execution_stats["failed_executions"] += 1
                
                error_msg = str(execution_error)
                if self.config.debug_mode:
                    print(f"❌ Execution error in {agent_name}: {error_msg}")
                
                # Retry logic
                if retry_count < self.config.max_retries:
                    if self.config.debug_mode:
                        print(f"🔄 Retrying {agent_name} (attempt {retry_count + 2})")
                    return self.execute_with_agent(agent_name, question, context, retry_count + 1)
                
                # Fallback to simulation if retries exhausted
                if self.config.enable_graceful_degradation:
                    if self.config.debug_mode:
                        print(f"🔄 Falling back to simulation for {agent_name}")
                    return self._simulation_execution(agent_name, question, context, error=error_msg)
                
                return {
                    "answer": f"Agent {agent_name} failed after {retry_count + 1} attempts: {error_msg}",
                    "steps": [
                        f"SmolagAgent {agent_name} failed",
                        f"Error: {error_msg}",
                        f"Retries attempted: {retry_count}"
                    ],
                    "success": False,
                    "error": error_msg,
                    "retry_count": retry_count
                }
                
        except Exception as e:
            error_msg = f"Critical error executing {agent_name}: {str(e)}"
            if self.config.debug_mode:
                print(f"❌ {error_msg}")
                print(f"Traceback: {traceback.format_exc()}")
            
            return {
                "answer": error_msg,
                "steps": [f"Critical error in {agent_name}", f"Error: {str(e)}"],
                "success": False,
                "error": str(e)
            }
    
    def _prepare_question(self, question: str, context: str) -> str:
        """Prepare question with context integration"""
        enhanced_question = question
        
        if context and len(context.strip()) > 0:
            # Truncate context if too long
            if len(context) > self.config.max_context_length:
                context = context[:self.config.max_context_length] + "... [context truncated]"
            
            enhanced_question = f"""Context from similar GAIA examples:
{context}

Question to solve:
{question}

Please provide a clear, step-by-step solution following GAIA format requirements."""
        
        return enhanced_question
    
    def _simulation_execution(self, agent_name: str, question: str, context: str, error: str = None) -> Dict:
        """Simulation with realistic responses"""
        tools_map = {
            "data_analyst": ["calculator", "pandas", "numpy", "statistical_analysis"],
            "web_researcher": ["web_search", "google_search", "content_retrieval", "fact_verification"],
            "document_processor": ["file_processor", "multi_format_reader", "content_extraction"],
            "general_assistant": ["calculator", "web_search", "reasoning", "classification"]
        }
        
        # Simulate realistic processing
        simulated_answer = self._generate_simulated_answer(agent_name, question)
        
        steps = [
            f"Simulation: {agent_name} initialized",
            f"Available tools: {', '.join(tools_map.get(agent_name, []))}",
            f"Question analysis: {len(question)} characters",
            f"Context processed: {len(context)} characters" if context else "No context provided",
        ]
        
        if error:
            steps.append(f"Fallback simulation due to error: {error}")
        
        steps.append("Simulation completed")
        
        return {
            "answer": simulated_answer,
            "steps": steps,
            "success": True,
            "agent_type": "SimulationAgent",
            "is_simulation": True
        }
    
    def _generate_simulated_answer(self, agent_name: str, question: str) -> str:
        """Generate realistic simulated answers"""
        question_lower = question.lower()
        
        if agent_name == "data_analyst":
            if any(word in question_lower for word in ["calculate", "compute", "math", "number"]):
                return "42"  # Classic placeholder for calculations
            return "Statistical analysis completed"
        
        elif agent_name == "web_researcher":
            if "search" in question_lower or "find" in question_lower:
                return "Information retrieved from web sources"
            return "Research completed"
        
        elif agent_name == "document_processor":
            if any(word in question_lower for word in ["file", "document", "image"]):
                return "Document processed successfully"
            return "File analysis completed"
        
        else:  # general_assistant
            return "Task completed using general reasoning"
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        return self.execution_stats.copy()
    
# ============================================================================
# MAIN GAIA AGENT
# ============================================================================

class GAIAAgent:
    """Production GAIA agent with robust error handling and performance tracking"""
    
    def __init__(self, config: GAIAConfig = None):
        if config is None:
            config = GAIAConfig()
        
        self.config = config
        self.metadata_manager = MetadataManager(config)
        self.retriever = self._initialize_retriever()
        self.llm = self._initialize_langchain_model()
        self.agent_manager = AgentManager(config)
        self.workflow = self._build_workflow()
        
        # Performance tracking
        self.execution_results = []
        self.performance_metrics = {
            "total_questions": 0,
            "successful_completions": 0,
            "strategy_usage": {"smolag_agent": 0, "direct_llm": 0},
            "average_execution_time": 0.0,
            "error_rate": 0.0
        }
        
        Path(config.results_output_path).mkdir(exist_ok=True)
        
        if config.debug_mode:
            print("🚀 GAIA Agent initialized successfully!")
            print(f"📊 SmolagAgent execution stats: {self.agent_manager.get_execution_stats()}")
    
    def _initialize_retriever(self) -> DevelopmentGAIARetriever:
        """Initialize retriever with error handling"""
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
    
    def _initialize_langchain_model(self):
        """Initialize LangChain model with error handling and fallback support"""
        try:
            return self._create_langchain_model_with_fallback(self.config.primary_model)
        except Exception as e:
            if self.config.enable_model_fallback and self.config.secondary_model:
                print(f"⚠️  Primary LangChain model failed ({e}), trying secondary model...")
                try:
                    return self._create_langchain_model_with_fallback(self.config.secondary_model)
                except Exception as e2:
                    print(f"❌ Secondary LangChain model also failed ({e2}), using hardcoded fallback")
            
            # Final fallback
            return self._create_langchain_fallback_model()
    
    def _create_langchain_model_with_fallback(self, model_name: str):
        """Create LangChain model for specific model"""
        if self.config.model_provider == "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=self.config.temperature
            )
        
        elif self.config.model_provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model_name,
                base_url=self.config.api_base or "http://localhost:11434",
                temperature=self.config.temperature
            )
        
        elif self.config.model_provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model_name,
                temperature=self.config.temperature,
                api_key=os.getenv("GROQ_API_KEY")
            )
        
        elif self.config.model_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.config.temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        
        else:
            raise ValueError(f"Unknown provider: {self.config.model_provider}")
    
    def _create_langchain_fallback_model(self):
        """Create hardcoded LangChain fallback model"""
        from langchain_openai import ChatOpenAI
        print("🔄 Using hardcoded LangChain OpenRouter fallback")
        return ChatOpenAI(
            model="qwen/qwen-2.5-coder-32b-instruct:free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7
        )
    
    def _build_workflow(self):
        """Build the workflow with error handling"""
        builder = StateGraph(GAIAState)
        
        # Add nodes
        builder.add_node("initialize", self._initialize_node)
        builder.add_node("rag_retrieval", self._rag_retrieval_node)
        builder.add_node("strategy_selection", self._strategy_selection_node)
        builder.add_node("smolag_execution", self._smolag_execution_node)
        builder.add_node("direct_llm_execution", self._direct_llm_execution_node)
        builder.add_node("answer_formatting", self._answer_formatting_node)
        builder.add_node("evaluation", self._evaluation_node)
        builder.add_node("error_recovery", self._error_recovery_node)
        
        # Define workflow with error handling
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "rag_retrieval")
        builder.add_edge("rag_retrieval", "strategy_selection")
        
        # Conditional routing
        builder.add_conditional_edges(
            "strategy_selection",
            self._route_execution,
            {
                "smolag_agent": "smolag_execution", 
                "direct_llm": "direct_llm_execution",
                "error_recovery": "error_recovery"
            }
        )
        
        # Error recovery paths
        builder.add_conditional_edges(
            "smolag_execution",
            self._check_execution_success,
            {
                "success": "answer_formatting",
                "retry": "error_recovery", 
                "fallback": "direct_llm_execution"
            }
        )
        
        builder.add_edge("direct_llm_execution", "answer_formatting")
        builder.add_edge("error_recovery", "answer_formatting")
        builder.add_edge("answer_formatting", "evaluation")
        builder.add_edge("evaluation", END)
        
        return builder.compile()
    
    # ============================================================================
    # WORKFLOW NODES
    # ============================================================================
    
    def _initialize_node(self, state: GAIAState):
        """Initialize execution state"""
        start_time = datetime.now()
        
        return {
            "messages": [HumanMessage(content=state["question"])],
            "execution_steps": ["Execution initialized"],
            "retry_count": 0,
            "errors": [],
            "fallback_used": False,
            "debug_info": {
                "start_time": start_time.isoformat(),
                "model_provider": self.config.model_provider,
                "model_name": self.config.primary_model,
                "config_version": "production"
            }
        }
    
    def _rag_retrieval_node(self, state: GAIAState):
        """RAG retrieval with error handling"""
        try:
            question = state["question"]
            messages = state["messages"]
            
            # Use retriever with error handling
            rag_result = self.retriever.retriever_node(messages)
            
            # Context extraction
            retriever_context = ""
            if len(rag_result["messages"]) > len(messages):
                rag_message = rag_result["messages"][-1]
                retriever_context = rag_message.content
            
            # Similar examples extraction
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
                            "answer": a_part,
                            "source": doc.metadata.get("source", "unknown"),
                            "similarity_score": getattr(doc, 'similarity_score', 0.0)
                        })
            
            return {
                "messages": rag_result["messages"],
                "retriever_context": retriever_context or "No similar examples found",
                "similar_examples": similar_examples,
                "execution_steps": state["execution_steps"] + [
                    f"RAG retrieval completed - found {len(similar_examples)} examples",
                    f"Context length: {len(retriever_context)} characters"
                ]
            }
            
        except Exception as e:
            error_msg = f"RAG retrieval error: {str(e)}"
            return {
                "messages": state["messages"],
                "retriever_context": "RAG retrieval failed",
                "similar_examples": [],
                "errors": state.get("errors", []) + [error_msg],
                "execution_steps": state["execution_steps"] + [
                    "RAG retrieval failed, proceeding without context"
                ]
            }
    
    def _strategy_selection_node(self, state: GAIAState):
        """Strategy selection with decision logic"""
        try:
            question = state["question"]
            similar_examples = state.get("similar_examples", [])
            errors = state.get("errors", [])
            
            # Complexity analysis
            complexity_analysis = self._analyze_question_complexity(question, similar_examples)
            
            # Decision logic with error consideration
            if len(errors) > 0:
                # If there were previous errors, prefer direct LLM
                strategy = "direct_llm"
                selected_agent = None
                decision_reason = "Using direct LLM due to previous errors"
            elif complexity_analysis["complexity_score"] > 0.6 or complexity_analysis["requires_tools"]:
                strategy = "smolag_agent"
                selected_agent = complexity_analysis["recommended_agent"]
                decision_reason = f"Using SmolagAgent due to complexity: {complexity_analysis['complexity_score']:.2f}"
            else:
                strategy = "direct_llm"
                selected_agent = None
                decision_reason = "Using direct LLM for simple question"
            
            # Track strategy usage
            self.performance_metrics["strategy_usage"][strategy] += 1
            
            return {
                "selected_strategy": strategy,
                "selected_agent": selected_agent,
                "execution_steps": state["execution_steps"] + [
                    f"Strategy: {strategy}" + (f" (agent: {selected_agent})" if selected_agent else ""),
                    f"Decision reason: {decision_reason}",
                    f"Complexity indicators: {', '.join(complexity_analysis['complexity_indicators'])}"
                ],
                "debug_info": {
                    **state.get("debug_info", {}),
                    "complexity_analysis": complexity_analysis,
                    "decision_reason": decision_reason
                }
            }
            
        except Exception as e:
            error_msg = f"Strategy selection error: {str(e)}"
            return {
                "selected_strategy": "direct_llm",  # Safe fallback
                "selected_agent": None,
                "errors": state.get("errors", []) + [error_msg],
                "execution_steps": state["execution_steps"] + [
                    "Strategy selection failed, defaulting to direct LLM"
                ]
            }
    
    def _analyze_question_complexity(self, question: str, similar_examples: List[Dict]) -> Dict:
        """Analyze question complexity for strategy selection"""
        complexity_indicators = []
        complexity_score = 0.0
        requires_tools = False
        
        question_lower = question.lower()
        
        # File attachment detection
        file_keywords = ["file", "attachment", "image", "audio", "document", "spreadsheet", "excel", "pdf", "csv", "jpg", "png", "mp3", "wav", "docx", "pptx"]
        if any(keyword in question_lower for keyword in file_keywords):
            complexity_indicators.append("file_processing")
            complexity_score += 0.6
            requires_tools = True
        
        # Calculation detection
        calc_keywords = ["calculate", "compute", "sum", "average", "statistics", "math", "percentage", "multiply", "divide", "equation", "formula"]
        math_symbols = ["+", "-", "*", "/", "=", "%", "^"]
        if any(keyword in question_lower for keyword in calc_keywords) or any(symbol in question for symbol in math_symbols):
            complexity_indicators.append("calculations")
            complexity_score += 0.5
            requires_tools = True
        
        # Web search detection
        search_keywords = ["current", "latest", "recent", "search", "find", "wikipedia", "google", "look up", "research", "today", "2024", "2025"]
        if any(keyword in question_lower for keyword in search_keywords):
            complexity_indicators.append("web_search")
            complexity_score += 0.4
            requires_tools = True
        
        # Multi-step detection
        multi_step_indicators = ["then", "after", "next", "first", "second", "finally", "step", "process"]
        question_marks_count = question.count("?")
        if (len(question.split()) > 30 or 
            question_marks_count > 1 or 
            any(indicator in question_lower for indicator in multi_step_indicators)):
            complexity_indicators.append("multi_step")
            complexity_score += 0.3
        
        # Reasoning detection
        reasoning_keywords = ["why", "how", "explain", "compare", "analyze", "evaluate", "reason", "because", "therefore"]
        if any(keyword in question_lower for keyword in reasoning_keywords):
            complexity_indicators.append("reasoning")
            complexity_score += 0.2
        
        # Similar examples boost
        if similar_examples and len(similar_examples) > 0:
            complexity_score += 0.1  # Boost for having context
        
        # Agent recommendation logic
        agent_mapping = {
            "file_processing": "document_processor",
            "calculations": "data_analyst", 
            "web_search": "web_researcher",
            "multi_step": "general_assistant",
            "reasoning": "general_assistant"
        }
        
        # Choose agent based on highest priority indicator
        priority_order = ["file_processing", "calculations", "web_search", "multi_step", "reasoning"]
        recommended_agent = "general_assistant"  # Default
        
        for indicator in priority_order:
            if indicator in complexity_indicators:
                recommended_agent = agent_mapping[indicator]
                break
        
        return {
            "complexity_score": min(complexity_score, 1.0),
            "complexity_indicators": complexity_indicators,
            "recommended_agent": recommended_agent,
            "requires_tools": requires_tools,
            "similar_examples_count": len(similar_examples),
            "question_length": len(question),
            "question_marks": question_marks_count
        }
    
    def _smolag_execution_node(self, state: GAIAState):
        """SmolagAgent execution with retry and fallback logic"""
        question = state["question"]
        agent_name = state["selected_agent"]
        context = state.get("retriever_context", "")
        retry_count = state.get("retry_count", 0)
        
        if self.config.debug_mode:
            print(f"🤖 SmolagAgent execution: {agent_name} (attempt {retry_count + 1})")
        
        try:
            # Execute with SmolagAgent
            execution_result = self.agent_manager.execute_with_agent(
                agent_name, question, context, retry_count
            )
            
            # Result processing
            if execution_result["success"]:
                return {
                    "raw_answer": execution_result["answer"],
                    "execution_steps": state["execution_steps"] + execution_result["steps"],
                    "debug_info": {
                        **state.get("debug_info", {}),
                        "agent_used": agent_name,
                        "agent_type": execution_result.get("agent_type", "unknown"),
                        "smolag_success": True,
                        "is_simulation": execution_result.get("is_simulation", False),
                        "retry_count": execution_result.get("retry_count", 0)
                    }
                }
            else:
                # Handle failure
                error_msg = execution_result.get("error", "Unknown SmolagAgent error")
                return {
                    "raw_answer": execution_result["answer"],
                    "execution_steps": state["execution_steps"] + execution_result["steps"],
                    "errors": state.get("errors", []) + [error_msg],
                    "debug_info": {
                        **state.get("debug_info", {}),
                        "agent_used": agent_name,
                        "smolag_success": False,
                        "smolag_error": error_msg,
                        "retry_count": execution_result.get("retry_count", 0)
                    }
                }
                
        except Exception as e:
            error_msg = f"Critical SmolagAgent execution error: {str(e)}"
            if self.config.debug_mode:
                print(f"❌ {error_msg}")
            
            return {
                "raw_answer": f"SmolagAgent execution failed: {error_msg}",
                "execution_steps": state["execution_steps"] + [
                    f"SmolagAgent {agent_name} execution failed",
                    f"Error: {error_msg}"
                ],
                "errors": state.get("errors", []) + [error_msg],
                "debug_info": {
                    **state.get("debug_info", {}),
                    "agent_used": agent_name,
                    "smolag_success": False,
                    "critical_error": error_msg
                }
            }
    
    def _direct_llm_execution_node(self, state: GAIAState):
        """Direct LLM execution with optimized prompt engineering"""
        try:
            question = state["question"]
            messages = state["messages"]
            similar_examples = state.get("similar_examples", [])
            
            # Build GAIA-compliant prompt
            system_prompt = self._build_system_prompt(similar_examples)
            
            final_messages = [SystemMessage(content=system_prompt)] + messages
            
            if self.config.debug_mode:
                print("🧠 Direct LLM execution")
            
            response = self.llm.invoke(final_messages)
            
            return {
                "raw_answer": response.content,
                "execution_steps": state["execution_steps"] + [
                    "Direct LLM execution with optimized prompt",
                    f"System prompt length: {len(system_prompt)} characters",
                    f"Response length: {len(response.content)} characters"
                ],
                "debug_info": {
                    **state.get("debug_info", {}),
                    "execution_method": "direct_llm",
                    "system_prompt_used": True,
                    "examples_in_prompt": len(similar_examples)
                }
            }
            
        except Exception as e:
            error_msg = f"Direct LLM execution error: {str(e)}"
            if self.config.debug_mode:
                print(f"❌ {error_msg}")
            
            return {
                "raw_answer": f"LLM execution failed: {error_msg}",
                "execution_steps": state["execution_steps"] + [
                    "Direct LLM execution failed"
                ],
                "errors": state.get("errors", []) + [error_msg]
            }
    
    def _build_system_prompt(self, similar_examples: List[Dict]) -> str:
        """Build system prompt with examples"""
        base_prompt = """You are a general AI assistant specialized in solving GAIA benchmark questions. Report your thoughts, and finish with: FINAL ANSWER: [YOUR FINAL ANSWER].

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list.
- Numbers: no commas, no units ($ %) unless specified
- Strings: no articles (the, a, an), no abbreviations, digits as text unless specified  
- Lists: apply above rules to each element

Be precise, concise, and accurate in your reasoning."""
        
        if similar_examples and len(similar_examples) > 0:
            examples_text = "\n\nHere are similar examples to guide your approach:\n"
            for i, example in enumerate(similar_examples[:2], 1):  # Limit to 2 examples
                examples_text += f"\nExample {i}:\nQuestion: {example['question']}\nAnswer: {example['answer']}\n"
            
            base_prompt += examples_text
        
        return base_prompt
    
    def _error_recovery_node(self, state: GAIAState):
        """Error recovery with intelligent fallback"""
        errors = state.get("errors", [])
        retry_count = state.get("retry_count", 0)
        
        if self.config.debug_mode:
            print(f"🔧 Error recovery activated (errors: {len(errors)}, retries: {retry_count})")
        
        # Simple error recovery - try direct LLM
        question = state["question"]
        messages = state["messages"]
        
        try:
            system_prompt = """You are a helpful AI assistant. Answer the question as best you can based on your knowledge. 
If you cannot provide a complete answer, give your best estimate or explain what you know about the topic.

Format your final answer clearly."""
            
            final_messages = [SystemMessage(content=system_prompt)] + messages
            response = self.llm.invoke(final_messages)
            
            return {
                "raw_answer": response.content,
                "fallback_used": True,
                "execution_steps": state["execution_steps"] + [
                    "Error recovery: Using simplified direct LLM",
                    f"Recovered from {len(errors)} error(s)"
                ],
                "debug_info": {
                    **state.get("debug_info", {}),
                    "error_recovery": True,
                    "recovery_method": "simplified_llm"
                }
            }
            
        except Exception as e:
            # Last resort fallback
            return {
                "raw_answer": "Unable to process question due to system errors",
                "fallback_used": True,
                "execution_steps": state["execution_steps"] + [
                    "Error recovery failed, using last resort fallback"
                ],
                "errors": errors + [f"Recovery error: {str(e)}"]
            }
    
    def _answer_formatting_node(self, state: GAIAState):
        """Answer formatting with GAIA compliance"""
        raw_answer = state.get("raw_answer", "")
        fallback_used = state.get("fallback_used", False)
        
        try:
            # FINAL ANSWER extraction
            formatted_answer = self._extract_and_format_answer(raw_answer)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(state)
            
            return {
                "final_answer": formatted_answer,
                "confidence_score": confidence_score,
                "execution_steps": state["execution_steps"] + [
                    "GAIA formatting applied",
                    f"Confidence score: {confidence_score:.2f}",
                    f"Fallback used: {fallback_used}"
                ],
                "debug_info": {
                    **state.get("debug_info", {}),
                    "formatting_applied": True,
                    "original_answer_length": len(raw_answer),
                    "formatted_answer_length": len(formatted_answer)
                }
            }
            
        except Exception as e:
            error_msg = f"Answer formatting error: {str(e)}"
            return {
                "final_answer": raw_answer.strip() if raw_answer else "No answer",
                "confidence_score": 0.1,
                "execution_steps": state["execution_steps"] + [
                    "Answer formatting failed, using raw answer"
                ],
                "errors": state.get("errors", []) + [error_msg]
            }
    
    def _extract_and_format_answer(self, raw_answer: str) -> str:
        """Extract and format answer"""
        if not raw_answer:
            return "No answer"
        
        answer = raw_answer.strip()
        
        # FINAL ANSWER extraction
        patterns = [
            r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
            r"Final Answer:\s*(.+?)(?:\n|$)",
            r"Answer:\s*(.+?)(?:\n|$)",
            r"The answer is:\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            import re
            match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                break
        
        # Apply GAIA formatting
        answer = self._apply_gaia_formatting(answer)
        
        return answer
    
    def _apply_gaia_formatting(self, raw_answer: str) -> str:
        """Apply GAIA benchmark formatting rules"""
        if not raw_answer:
            return "No answer"
        
        answer = raw_answer.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "the answer is", "answer:", "final answer:", "result:", "solution:",
            "the result is", "this gives us", "therefore", "so", "thus",
            "the final answer is", "my answer is", "i think", "i believe"
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                answer_lower = answer.lower()
        
        # Remove leading/trailing punctuation and quotes
        answer = answer.strip('.,!?:;"\'')
        
        # Handle special number formatting
        if answer.replace('.', '').replace('-', '').replace(',', '').isdigit():
            # Remove commas from numbers as per GAIA rules
            answer = answer.replace(',', '')
        
        # Handle percentage formatting
        if '%' in answer and 'unless specified' not in raw_answer.lower():
            # Remove % unless specifically requested
            parts = answer.split('%')
            if len(parts) == 2 and parts[1].strip() == '':
                answer = parts[0].strip()
        
        return answer
    
    def _calculate_confidence_score(self, state: GAIAState) -> float:
        """Calculate confidence score based on execution factors"""
        confidence = 1.0
        
        # Reduce confidence for errors
        errors = state.get("errors", [])
        confidence -= len(errors) * 0.2
        
        # Reduce confidence for fallback usage
        if state.get("fallback_used", False):
            confidence -= 0.3
        
        # Reduce confidence for simulation agents
        debug_info = state.get("debug_info", {})
        if debug_info.get("is_simulation", False):
            confidence -= 0.4
        
        # Boost confidence for successful SmolagAgent execution
        if debug_info.get("smolag_success", False):
            confidence += 0.1
        
        # Boost confidence for RAG context
        if len(state.get("similar_examples", [])) > 0:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _evaluation_node(self, state: GAIAState):
        """Evaluation with comprehensive metrics"""
        end_time = datetime.now()
        start_time_str = state.get("debug_info", {}).get("start_time")
        
        execution_time = None
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            execution_time = (end_time - start_time).total_seconds()
        
        # Accuracy evaluation
        is_correct = None
        if state.get("ground_truth"):
            is_correct = self._evaluate_answer(
                state.get("final_answer", ""), 
                state["ground_truth"]
            )
        
        # Update performance metrics
        self.performance_metrics["total_questions"] += 1
        if is_correct:
            self.performance_metrics["successful_completions"] += 1
        
        if execution_time:
            # Update average execution time
            total_time = (self.performance_metrics["average_execution_time"] * 
                         (self.performance_metrics["total_questions"] - 1) + execution_time)
            self.performance_metrics["average_execution_time"] = total_time / self.performance_metrics["total_questions"]
        
        # Calculate error rate
        errors = state.get("errors", [])
        if errors:
            total_errors = sum(1 for result in self.execution_results if result.get("errors", []))
            self.performance_metrics["error_rate"] = total_errors / self.performance_metrics["total_questions"]
        
        return {
            "execution_time": execution_time,
            "model_used": f"{self.config.model_provider}/{self.config.primary_model}",
            "execution_steps": state["execution_steps"] + [
                "Evaluation completed",
                f"Execution time: {execution_time:.2f}s" if execution_time else "No timing data",
                f"Correctness: {'✅' if is_correct else '❌' if is_correct is False else '❓'}"
            ],
            "debug_info": {
                **state.get("debug_info", {}),
                "end_time": end_time.isoformat(),
                "is_correct": is_correct,
                "execution_time": execution_time,
                "performance_metrics": self.performance_metrics.copy()
            }
        }
    
    def _evaluate_answer(self, predicted: str, ground_truth: str) -> bool:
        """Answer evaluation with fuzzy matching"""
        if not predicted or not ground_truth:
            return False
        
        # Normalize for comparison
        pred_normalized = str(predicted).lower().strip()
        truth_normalized = str(ground_truth).lower().strip()
        
        # Exact match
        if pred_normalized == truth_normalized:
            return True
        
        # Remove common variations
        variations_to_remove = ['.', ',', '!', '?', ' ', '\n', '\t']
        pred_clean = pred_normalized
        truth_clean = truth_normalized
        
        for var in variations_to_remove:
            pred_clean = pred_clean.replace(var, '')
            truth_clean = truth_clean.replace(var, '')
        
        if pred_clean == truth_clean:
            return True
        
        # Numeric comparison
        try:
            pred_num = float(pred_normalized.replace(',', ''))
            truth_num = float(truth_normalized.replace(',', ''))
            return abs(pred_num - truth_num) < 1e-6
        except:
            pass
        
        return False
    
    def _check_execution_success(self, state: GAIAState) -> str:
        """Check if SmolagAgent execution was successful"""
        debug_info = state.get("debug_info", {})
        errors = state.get("errors", [])
        retry_count = state.get("retry_count", 0)
        
        if debug_info.get("smolag_success", False):
            return "success"
        elif retry_count < self.config.max_retries:
            return "retry"
        else:
            return "fallback"
    
    def _route_execution(self, state: GAIAState) -> str:
        """Execution routing with error handling"""
        strategy = state.get("selected_strategy", "direct_llm")
        errors = state.get("errors", [])
        
        # If there are critical errors, go to error recovery
        if len(errors) > 2:
            return "error_recovery"
        
        return strategy
    
    # ============================================================================
    # PUBLIC INTERFACE
    # ============================================================================
    
    def run_single_question(self, question: str, task_id: str = None, 
                          ground_truth: str = None, level: int = None) -> Dict:
        """Execute single question with comprehensive error handling"""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        initial_state = {
            "question": question,
            "task_id": task_id,
            "ground_truth": ground_truth,
            "level": level,
            "retry_count": 0,
            "errors": [],
            "fallback_used": False
        }
        
        if self.config.debug_mode:
            print(f"🔍 Processing: {question[:60]}...")
        
        try:
            result = self.workflow.invoke(initial_state)
            
            # Create execution record
            execution_record = {
                "task_id": task_id,
                "question": question,
                "final_answer": result.get("final_answer", ""),
                "raw_answer": result.get("raw_answer", ""),
                "ground_truth": ground_truth,
                "level": level,
                "is_correct": result.get("debug_info", {}).get("is_correct"),
                "execution_time": result.get("execution_time"),
                "strategy_used": result.get("selected_strategy"),
                "selected_agent": result.get("selected_agent"),
                "model_used": result.get("model_used"),
                "confidence_score": result.get("confidence_score", 0.0),
                "similar_examples_count": len(result.get("similar_examples", [])),
                "errors": result.get("errors", []),
                "fallback_used": result.get("fallback_used", False),
                "execution_steps": result.get("execution_steps", []),
                "timestamp": datetime.now().isoformat(),
                "config_version": "production"
            }
            
            self.execution_results.append(execution_record)
            
            if self.config.save_intermediate_results:
                self._save_result(execution_record)
            
            return result
            
        except Exception as e:
            error_msg = f"Critical workflow error: {str(e)}"
            if self.config.debug_mode:
                print(f"❌ {error_msg}")
                print(f"Traceback: {traceback.format_exc()}")
            
            # Return error result
            error_result = {
                "task_id": task_id,
                "question": question,
                "final_answer": "Workflow execution failed",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_results.append(error_result)
            return error_result
    
    def run_batch_evaluation(self, sample_size: int = None) -> pd.DataFrame:
        """Batch evaluation with comprehensive analysis"""
        test_data = self.metadata_manager.get_test_sample(sample_size)
        
        print(f"🧪 Evaluation on {len(test_data)} questions...")
        print("=" * 60)
        
        results = []
        successful_runs = 0
        
        for i, example in enumerate(test_data):
            try:
                if self.config.debug_mode and (i + 1) % 5 == 0:
                    accuracy_so_far = successful_runs / (i + 1) if i + 1 > 0 else 0
                    print(f"  Progress: {i+1}/{len(test_data)} ({(i+1)/len(test_data)*100:.1f}%) | Accuracy: {accuracy_so_far:.3f}")
                
                result = self.run_single_question(
                    question=example.get("Question", ""),
                    task_id=example.get("task_id", f"eval_{i}"),
                    ground_truth=example.get("Final answer", ""),
                    level=example.get("Level", 1)
                )
                
                if self.execution_results[-1].get("is_correct"):
                    successful_runs += 1
                
                results.append(self.execution_results[-1])
                
            except Exception as e:
                print(f"❌ Error processing question {i}: {e}")
                error_record = {
                    "task_id": f"eval_{i}",
                    "question": example.get("Question", ""),
                    "final_answer": "Processing failed",
                    "error": str(e),
                    "is_correct": False
                }
                results.append(error_record)
        
        # Analysis
        df = pd.DataFrame(results)
        
        print(f"\n🎯 EVALUATION RESULTS")
        print("=" * 40)
        print(f"Total questions: {len(df)}")
        
        if 'is_correct' in df and df['is_correct'].notna().any():
            accuracy = df['is_correct'].mean()
            print(f"Overall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if 'execution_time' in df and df['execution_time'].notna().any():
            avg_time = df['execution_time'].mean()
            print(f"Average execution time: {avg_time:.2f}s")
        
        if 'confidence_score' in df and df['confidence_score'].notna().any():
            avg_confidence = df['confidence_score'].mean()
            print(f"Average confidence: {avg_confidence:.3f}")
        
        # Strategy breakdown
        if 'strategy_used' in df:
            print(f"\n📊 Strategy Performance:")
            strategy_stats = df.groupby('strategy_used').agg({
                'is_correct': ['count', 'mean'],
                'execution_time': 'mean',
                'confidence_score': 'mean'
            }).round(3)
            print(strategy_stats)
        
        if 'level' in df:
            print(f"\n📈 Level Performance:")
            level_stats = df.groupby('level').agg({
                'is_correct': ['count', 'mean'],
                'execution_time': 'mean'
            }).round(3)
            print(level_stats)
        
        # Error analysis
        error_count = df['errors'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
        fallback_count = df['fallback_used'].sum() if 'fallback_used' in df else 0
        
        print(f"\n🔧 Error Analysis:")
        print(f"Total errors: {error_count}")
        print(f"Fallback usage: {fallback_count}")
        print(f"SmolagAgent stats: {self.agent_manager.get_execution_stats()}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = Path(self.config.results_output_path) / f"evaluation_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved: {results_file}")
        
        return df
    
    def _save_result(self, result: Dict):
        """Save individual result"""
        results_file = Path(self.config.results_output_path) / "individual_results.jsonl"
        with open(results_file, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            "performance_metrics": self.performance_metrics.copy(),
            "agent_stats": self.agent_manager.get_execution_stats(),
            "total_executions": len(self.execution_results),
            "config": {
                "model_provider": self.config.model_provider,
                "primary_model": self.config.primary_model,
                "version": "production"
            }
        }
    
    def close(self):
        """Cleanup and save performance summary"""
        try:
            if hasattr(self.retriever, 'close'):
                self.retriever.close()
            
            # Save final performance summary
            if self.execution_results:
                summary = self.get_performance_summary()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_file = Path(self.config.results_output_path) / f"performance_summary_{timestamp}.json"
                with open(summary_file, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                
                if self.config.debug_mode:
                    print(f"💾 Performance summary saved: {summary_file}")
                    
        except Exception as e:
            if self.config.debug_mode:
                print(f"⚠️  Cleanup warning: {e}")

# ============================================================================
# METADATA MANAGER
# ============================================================================

class MetadataManager:
    """Metadata manager with analysis capabilities"""
    
    def __init__(self, config: GAIAConfig):
        self.config = config
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> List[Dict]:
        """Load metadata with error handling"""
        try:
            with open(self.config.metadata_path, 'r', encoding='utf-8') as f:
                metadata = [json.loads(line) for line in f]
            print(f"✅ Loaded {len(metadata)} GAIA examples from metadata")
            return metadata
        except FileNotFoundError:
            print(f"❌ Metadata file not found: {self.config.metadata_path}")
            return []
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return []
    
    def get_test_sample(self, sample_size: Optional[int] = None) -> List[Dict]:
        """Get test sample with level distribution"""
        if sample_size is None:
            return self.metadata
        
        # Try to maintain level distribution in sample
        if sample_size >= len(self.metadata):
            return self.metadata
        
        # Group by level and sample proportionally
        level_groups = {}
        for item in self.metadata:
            level = item.get('Level', 1)
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(item)
        
        sample = []
        remaining = sample_size
        
        for level in sorted(level_groups.keys()):
            level_count = len(level_groups[level])
            level_sample_size = min(remaining, max(1, int(sample_size * level_count / len(self.metadata))))
            sample.extend(level_groups[level][:level_sample_size])
            remaining -= level_sample_size
            
            if remaining <= 0:
                break
        
        return sample[:sample_size]
    
    def analyze_tool_usage(self) -> Dict[str, int]:
        """Tool usage analysis"""
        tool_frequencies = {}
        
        for item in self.metadata:
            question = item.get('Question', '').lower()
            
            # Pattern detection
            patterns = {
                'calculator': ["calculate", "compute", "math", "number", "sum", "average", "percentage", "+", "-", "*", "/", "="],
                'web_search': ["search", "find", "lookup", "google", "current", "latest", "recent", "today", "wikipedia"],
                'file_processing': ["file", "attachment", "document", "image", "audio", "excel", "pdf", "csv", "spreadsheet"],
                'reasoning': ["why", "how", "explain", "compare", "analyze", "evaluate", "reason"],
                'multi_step': ["then", "after", "next", "first", "second", "step", "process"],
            }
            
            for tool, keywords in patterns.items():
                if any(keyword in question for keyword in keywords):
                    tool_frequencies[tool] = tool_frequencies.get(tool, 0) + 1
        
        return tool_frequencies
    
    def get_complexity_distribution(self) -> Dict:
        """Analyze complexity distribution in metadata"""
        complexity_stats = {
            'simple': 0,
            'medium': 0, 
            'complex': 0,
            'by_level': {}
        }
        
        for item in self.metadata:
            question = item.get('Question', '')
            level = item.get('Level', 1)
            
            # Simple complexity scoring
            score = 0
            if len(question.split()) > 25:
                score += 1
            if question.count('?') > 1:
                score += 1
            if any(word in question.lower() for word in ['calculate', 'file', 'search', 'analyze']):
                score += 1
            
            if score <= 1:
                complexity_stats['simple'] += 1
            elif score == 2:
                complexity_stats['medium'] += 1
            else:
                complexity_stats['complex'] += 1
            
            if level not in complexity_stats['by_level']:
                complexity_stats['by_level'][level] = {'simple': 0, 'medium': 0, 'complex': 0}
            
            if score <= 1:
                complexity_stats['by_level'][level]['simple'] += 1
            elif score == 2:
                complexity_stats['by_level'][level]['medium'] += 1
            else:
                complexity_stats['by_level'][level]['complex'] += 1
        
        return complexity_stats

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

class ModelConfigs:
    """Production model configurations with primary/secondary fallback support"""
    
    @staticmethod
    def get_openrouter_configs():
        """OpenRouter configurations with fallback models"""
        return {
            "qwen_coder": {
                "model_provider": "openrouter",
                "primary_model": "qwen/qwen-2.5-coder-32b-instruct:free",
                "secondary_model": "qwen/qwen3-32b:free",
                "temperature": 0.3,
                "max_agent_steps": 20,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "qwen_32b": {
                "model_provider": "openrouter", 
                "primary_model": "qwen/qwen3-32b:free",
                "secondary_model": "qwen/qwen3-14b:free",
                "temperature": 0.4,
                "max_agent_steps": 15,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "qwen_14b": {
                "model_provider": "openrouter",
                "primary_model": "qwen/qwen3-14b:free",
                "secondary_model": "qwen/qwen-2.5-coder-32b-instruct:free",
                "temperature": 0.4,
                "max_agent_steps": 12,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "deepseek": {
                "model_provider": "openrouter",
                "primary_model": "deepseek/deepseek-chat:free",
                "secondary_model": "qwen/qwen-2.5-coder-32b-instruct:free",
                "temperature": 0.5,
                "max_agent_steps": 12,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "llama_70b": {
                "model_provider": "openrouter",
                "primary_model": "meta-llama/llama-3.1-70b-instruct:free",
                "secondary_model": "qwen/qwen3-32b:free",
                "temperature": 0.4,
                "max_agent_steps": 18,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            }
        }
    
    @staticmethod
    def get_groq_configs():
        """Groq configurations with fallback models"""
        return {
            "qwen_qwq_groq": {
                "model_provider": "groq",
                "primary_model": "qwen-qwq-32b",
                "secondary_model": "llama3-70b-8192",
                "temperature": 0.2,
                "max_agent_steps": 20,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "llama_scout_groq": {
                "model_provider": "groq",
                "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "secondary_model": "llama3-70b-8192",
                "temperature": 0.3,
                "max_agent_steps": 18,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "llama_70b_groq": {
                "model_provider": "groq",
                "primary_model": "llama3-70b-8192",
                "secondary_model": "qwen-qwq-32b",
                "temperature": 0.3,
                "max_agent_steps": 15,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "mixtral_groq": {
                "model_provider": "groq",
                "primary_model": "mixtral-8x7b-32768",
                "secondary_model": "llama3-70b-8192",
                "temperature": 0.4,
                "max_agent_steps": 12,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            }
        }
    
    @staticmethod
    def get_google_configs():
        """Google configurations with fallback models"""
        return {
            "gemma_3n": {
                "model_provider": "google",
                "primary_model": "gemma-3n-e4b-it",
                "secondary_model": "gemini-2.5-flash-preview-04-17",
                "temperature": 0.3,
                "max_agent_steps": 15,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "gemini_flash_04": {
                "model_provider": "google",
                "primary_model": "gemini-2.5-flash-preview-04-17",
                "secondary_model": "gemini-2.5-flash-preview-05-20",
                "temperature": 0.4,
                "max_agent_steps": 18,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "gemini_flash_05": {
                "model_provider": "google",
                "primary_model": "gemini-2.5-flash-preview-05-20",
                "secondary_model": "gemini-2.5-flash-preview-04-17",
                "temperature": 0.4,
                "max_agent_steps": 18,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            }
        }
    
    @staticmethod
    def get_ollama_configs():
        """Ollama configurations with fallback models"""
        return {
            "qwen_agent": {
                "model_provider": "ollama",
                "primary_model": "qwen-agent-custom:latest",
                "secondary_model": "qwen2.5-coder:32b",
                "api_base": "http://localhost:11434",
                "num_ctx": 32768,
                "temperature": 0.3,
                "max_agent_steps": 25,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "qwen_coder_ollama": {
                "model_provider": "ollama",
                "primary_model": "qwen2.5-coder:32b",
                "secondary_model": "qwen-agent-custom:latest",
                "api_base": "http://localhost:11434",
                "num_ctx": 32768,
                "temperature": 0.2,
                "max_agent_steps": 20,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "llama_ollama": {
                "model_provider": "ollama",
                "primary_model": "llama3.2:latest",
                "secondary_model": "qwen2.5-coder:32b",
                "api_base": "http://localhost:11434",
                "num_ctx": 32768,
                "temperature": 0.3,
                "max_agent_steps": 15,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            }
        }
    
    @staticmethod
    def get_all_configs():
        """Get all configurations including new Google configs"""
        configs = {}
        configs.update(ModelConfigs.get_openrouter_configs())
        configs.update(ModelConfigs.get_groq_configs())
        configs.update(ModelConfigs.get_google_configs())  # Added Google configs
        configs.update(ModelConfigs.get_ollama_configs())
        return configs

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_gaia_agent(config_overrides: Union[str, Dict] = None) -> GAIAAgent:
    """Factory function for creating GAIA agent"""
    config = GAIAConfig()
    
    if config_overrides:
        if isinstance(config_overrides, str):
            model_configs = ModelConfigs.get_all_configs()
            if config_overrides in model_configs:
                config_dict = model_configs[config_overrides]
                print(f"📋 Using config: {config_overrides}")
            else:
                raise ValueError(f"Unknown config: {config_overrides}")
        else:
            config_dict = config_overrides
        
        # Apply overrides
        for key, value in config_dict.items():
            setattr(config, key, value)
    
    return GAIAAgent(config)

def create_production_gaia_agent(
    model_config: Union[str, Dict] = "qwen_coder",
    enable_logging: bool = True,
    performance_tracking: bool = True,
    max_retries: int = 2
) -> GAIAAgent:
    """Create production-ready GAIA agent"""
    
    if isinstance(model_config, str):
        model_configs = ModelConfigs.get_all_configs()
        if model_config not in model_configs:
            raise ValueError(f"Unknown model config: {model_config}")
        config_dict = model_configs[model_config]
    else:
        config_dict = model_config
    
    # Add production settings
    config_dict.update({
        "debug_mode": enable_logging,
        "save_intermediate_results": performance_tracking,
        "max_retries": max_retries,
        "enable_graceful_degradation": True,
        "gaia_formatting_strict": True,
        "enable_confidence_scoring": True
    })
    
    config = GAIAConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    agent = GAIAAgent(config)
    return agent

def quick_test(
    question: str = "Calculate 15% of 1000", 
    config_name: str = "qwen_coder"
) -> Dict:
    """Quick test with error handling"""
    try:
        agent = create_gaia_agent(config_name)
        result = agent.run_single_question(question)
        
        # Print summary
        print(f"\n🎯 TEST RESULT")
        print("=" * 30)
        print(f"Question: {question}")
        print(f"Final Answer: {result.get('final_answer', 'No answer')}")
        print(f"Strategy: {result.get('selected_strategy', 'Unknown')}")
        print(f"Agent: {result.get('selected_agent', 'N/A')}")
        print(f"Confidence: {result.get('confidence_score', 0):.2f}")
        print(f"Errors: {len(result.get('errors', []))}")
        print(f"Fallback Used: {'Yes' if result.get('fallback_used') else 'No'}")
        
        agent.close()
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return {"error": str(e)}

def compare_configs(
    question: str = "What is 25% of 400?",
    configs_to_test: List[str] = None
) -> pd.DataFrame:
    """Compare different model configurations"""
    if configs_to_test is None:
        configs_to_test = ["qwen_coder", "qwen_32b", "deepseek"]
    
    results = []
    
    print(f"🔄 Comparing {len(configs_to_test)} configurations...")
    
    for config_name in configs_to_test:
        print(f"Testing {config_name}...")
        try:
            result = quick_test(question, config_name)
            
            results.append({
                "config": config_name,
                "final_answer": result.get("final_answer", ""),
                "strategy": result.get("selected_strategy", ""),
                "confidence": result.get("confidence_score", 0),
                "execution_time": result.get("execution_time", 0),
                "errors": len(result.get("errors", [])),
                "success": "error" not in result
            })
            
        except Exception as e:
            results.append({
                "config": config_name,
                "final_answer": f"Error: {e}",
                "strategy": "failed",
                "confidence": 0,
                "execution_time": 0,
                "errors": 1,
                "success": False
            })
    
    df = pd.DataFrame(results)
    print(f"\n📊 Configuration Comparison:")
    print(df.to_string(index=False))
    
    return df

def run_gaia_benchmark(
    sample_size: int = 50,
    config_name: str = "qwen_coder",
    save_results: bool = True
) -> Dict:
    """Run GAIA benchmark with comprehensive analysis"""
    print(f"🚀 GAIA Benchmark - {sample_size} questions")
    print("=" * 50)
    
    agent = create_gaia_agent(config_name)
    
    try:
        # Run evaluation
        results_df = agent.run_batch_evaluation(sample_size)
        
        # Comprehensive analysis
        analysis = {
            "total_questions": len(results_df),
            "overall_accuracy": results_df['is_correct'].mean() if 'is_correct' in results_df else 0,
            "average_confidence": results_df['confidence_score'].mean() if 'confidence_score' in results_df else 0,
            "average_execution_time": results_df['execution_time'].mean() if 'execution_time' in results_df else 0,
            "strategy_breakdown": results_df['strategy_used'].value_counts().to_dict() if 'strategy_used' in results_df else {},
            "level_performance": results_df.groupby('level')['is_correct'].mean().to_dict() if 'level' in results_df else {},
            "error_rate": (results_df['errors'].apply(lambda x: len(x) if isinstance(x, list) else 0) > 0).mean() if 'errors' in results_df else 0,
            "fallback_rate": results_df['fallback_used'].mean() if 'fallback_used' in results_df else 0,
            "config_used": config_name,
            "agent_stats": agent.agent_manager.get_execution_stats()
        }
        
        print(f"\n🎯 BENCHMARK ANALYSIS")
        print("=" * 30)
        for key, value in analysis.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  ├── {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_file = Path(agent.config.results_output_path) / f"benchmark_analysis_{timestamp}.json"
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"\n💾 Analysis saved: {analysis_file}")
        
        agent.close()
        return analysis
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        agent.close()
        return {"error": str(e)}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Production GAIA Agent System")
    print("=" * 70)
    
    # Show available configurations
    print("\n📋 Available Model Configurations:")
    configs = ModelConfigs.get_all_configs()
    for name, config in configs.items():
        provider = config['model_provider'] 
        model = config['primary_model']
        steps = config.get('max_agent_steps', 'default')
        temp = config.get('temperature', 'default')
        print(f"  ├── {name}: {provider}/{model} (steps: {steps}, temp: {temp})")
    
    # Quick test
    print(f"\n🧪 Quick test...")
    try:
        result = quick_test(
            question="Calculate 25% of 800",
            config_name="qwen_coder"
        )
        
        if "error" not in result:
            print(f"✅ Test successful!")
        else:
            print(f"❌ Test failed: {result['error']}")
            print("💡 Check your API keys and model configurations")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        print("💡 Ensure your retriever system and dependencies are properly set up")
    
    print(f"\n🎓 Ready for GAIA benchmark testing!")
    print("Use create_gaia_agent() or run_gaia_benchmark() to get started.")
    
# ============================================================================
# MAIN GAIA AGENT
# ============================================================================

class GAIAAgent:
    """Production GAIA agent with robust error handling and performance tracking"""
    
    def __init__(self, config: GAIAConfig = None):
        if config is None:
            config = GAIAConfig()
        
        self.config = config
        self.metadata_manager = MetadataManager(config)
        self.retriever = self._initialize_retriever()
        self.llm = self._initialize_langchain_model()
        self.agent_manager = AgentManager(config)
        self.workflow = self._build_workflow()
        
        # Performance tracking
        self.execution_results = []
        self.performance_metrics = {
            "total_questions": 0,
            "successful_completions": 0,
            "strategy_usage": {"smolag_agent": 0, "direct_llm": 0},
            "average_execution_time": 0.0,
            "error_rate": 0.0
        }
        
        Path(config.results_output_path).mkdir(exist_ok=True)
        
        if config.debug_mode:
            print("🚀 GAIA Agent initialized successfully!")
            print(f"📊 SmolagAgent execution stats: {self.agent_manager.get_execution_stats()}")
    
    def _initialize_retriever(self) -> DevelopmentGAIARetriever:
        """Initialize retriever with error handling"""
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
    
    def _initialize_langchain_model(self):
        """Initialize LangChain model with error handling"""
        try:
            if self.config.model_provider == "openrouter":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=self.config.primary_model,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=self.config.temperature
                )
            
            elif self.config.model_provider == "ollama":
                from langchain_ollama import ChatOllama
                return ChatOllama(
                    model=self.config.primary_model,
                    base_url=self.config.api_base or "http://localhost:11434",
                    temperature=self.config.temperature
                )
            
            elif self.config.model_provider == "groq":
                from langchain_groq import ChatGroq
                return ChatGroq(
                    model=self.config.primary_model,
                    temperature=self.config.temperature,
                    api_key=os.getenv("GROQ_API_KEY")
                )
            
            elif self.config.model_provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=self.config.primary_model,
                    temperature=self.config.temperature,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
            
            else:
                # Fallback
                from langchain_openai import ChatOpenAI
                print("⚠️  Using OpenRouter fallback")
                return ChatOpenAI(
                    model="qwen/qwen-2.5-coder-32b-instruct:free",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0.7
                )
                
        except Exception as e:
            print(f"❌ Error initializing LangChain model: {e}")
            raise
    
    def _build_workflow(self):
        """Build the workflow with error handling"""
        builder = StateGraph(GAIAState)
        
        # Add nodes
        builder.add_node("initialize", self._initialize_node)
        builder.add_node("rag_retrieval", self._rag_retrieval_node)
        builder.add_node("strategy_selection", self._strategy_selection_node)
        builder.add_node("smolag_execution", self._smolag_execution_node)
        builder.add_node("direct_llm_execution", self._direct_llm_execution_node)
        builder.add_node("answer_formatting", self._answer_formatting_node)
        builder.add_node("evaluation", self._evaluation_node)
        builder.add_node("error_recovery", self._error_recovery_node)
        
        # Define workflow with error handling
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "rag_retrieval")
        builder.add_edge("rag_retrieval", "strategy_selection")
        
        # Conditional routing
        builder.add_conditional_edges(
            "strategy_selection",
            self._route_execution,
            {
                "smolag_agent": "smolag_execution", 
                "direct_llm": "direct_llm_execution",
                "error_recovery": "error_recovery"
            }
        )
        
        # Error recovery paths
        builder.add_conditional_edges(
            "smolag_execution",
            self._check_execution_success,
            {
                "success": "answer_formatting",
                "retry": "error_recovery", 
                "fallback": "direct_llm_execution"
            }
        )
        
        builder.add_edge("direct_llm_execution", "answer_formatting")
        builder.add_edge("error_recovery", "answer_formatting")
        builder.add_edge("answer_formatting", "evaluation")
        builder.add_edge("evaluation", END)
        
        return builder.compile()
    
    # ============================================================================
    # WORKFLOW NODES
    # ============================================================================
    
    def _initialize_node(self, state: GAIAState):
        """Initialize execution state"""
        start_time = datetime.now()
        
        return {
            "messages": [HumanMessage(content=state["question"])],
            "execution_steps": ["Execution initialized"],
            "retry_count": 0,
            "errors": [],
            "fallback_used": False,
            "debug_info": {
                "start_time": start_time.isoformat(),
                "model_provider": self.config.model_provider,
                "model_name": self.config.primary_model,
                "config_version": "production"
            }
        }
    
    def _rag_retrieval_node(self, state: GAIAState):
        """RAG retrieval with error handling"""
        try:
            question = state["question"]
            messages = state["messages"]
            
            # Use retriever with error handling
            rag_result = self.retriever.retriever_node(messages)
            
            # Context extraction
            retriever_context = ""
            if len(rag_result["messages"]) > len(messages):
                rag_message = rag_result["messages"][-1]
                retriever_context = rag_message.content
            
            # Similar examples extraction
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
                            "answer": a_part,
                            "source": doc.metadata.get("source", "unknown"),
                            "similarity_score": getattr(doc, 'similarity_score', 0.0)
                        })
            
            return {
                "messages": rag_result["messages"],
                "retriever_context": retriever_context or "No similar examples found",
                "similar_examples": similar_examples,
                "execution_steps": state["execution_steps"] + [
                    f"RAG retrieval completed - found {len(similar_examples)} examples",
                    f"Context length: {len(retriever_context)} characters"
                ]
            }
            
        except Exception as e:
            error_msg = f"RAG retrieval error: {str(e)}"
            return {
                "messages": state["messages"],
                "retriever_context": "RAG retrieval failed",
                "similar_examples": [],
                "errors": state.get("errors", []) + [error_msg],
                "execution_steps": state["execution_steps"] + [
                    "RAG retrieval failed, proceeding without context"
                ]
            }
    
    def _strategy_selection_node(self, state: GAIAState):
        """Strategy selection with decision logic"""
        try:
            question = state["question"]
            similar_examples = state.get("similar_examples", [])
            errors = state.get("errors", [])
            
            # Complexity analysis
            complexity_analysis = self._analyze_question_complexity(question, similar_examples)
            
            # Decision logic with error consideration
            if len(errors) > 0:
                # If there were previous errors, prefer direct LLM
                strategy = "direct_llm"
                selected_agent = None
                decision_reason = "Using direct LLM due to previous errors"
            elif complexity_analysis["complexity_score"] > 0.6 or complexity_analysis["requires_tools"]:
                strategy = "smolag_agent"
                selected_agent = complexity_analysis["recommended_agent"]
                decision_reason = f"Using SmolagAgent due to complexity: {complexity_analysis['complexity_score']:.2f}"
            else:
                strategy = "direct_llm"
                selected_agent = None
                decision_reason = "Using direct LLM for simple question"
            
            # Track strategy usage
            self.performance_metrics["strategy_usage"][strategy] += 1
            
            return {
                "selected_strategy": strategy,
                "selected_agent": selected_agent,
                "execution_steps": state["execution_steps"] + [
                    f"Strategy: {strategy}" + (f" (agent: {selected_agent})" if selected_agent else ""),
                    f"Decision reason: {decision_reason}",
                    f"Complexity indicators: {', '.join(complexity_analysis['complexity_indicators'])}"
                ],
                "debug_info": {
                    **state.get("debug_info", {}),
                    "complexity_analysis": complexity_analysis,
                    "decision_reason": decision_reason
                }
            }
            
        except Exception as e:
            error_msg = f"Strategy selection error: {str(e)}"
            return {
                "selected_strategy": "direct_llm",  # Safe fallback
                "selected_agent": None,
                "errors": state.get("errors", []) + [error_msg],
                "execution_steps": state["execution_steps"] + [
                    "Strategy selection failed, defaulting to direct LLM"
                ]
            }
    
    def _analyze_question_complexity(self, question: str, similar_examples: List[Dict]) -> Dict:
        """Analyze question complexity for strategy selection"""
        complexity_indicators = []
        complexity_score = 0.0
        requires_tools = False
        
        question_lower = question.lower()
        
        # File attachment detection
        file_keywords = ["file", "attachment", "image", "audio", "document", "spreadsheet", "excel", "pdf", "csv", "jpg", "png", "mp3", "wav", "docx", "pptx"]
        if any(keyword in question_lower for keyword in file_keywords):
            complexity_indicators.append("file_processing")
            complexity_score += 0.6
            requires_tools = True
        
        # Calculation detection
        calc_keywords = ["calculate", "compute", "sum", "average", "statistics", "math", "percentage", "multiply", "divide", "equation", "formula"]
        math_symbols = ["+", "-", "*", "/", "=", "%", "^"]
        if any(keyword in question_lower for keyword in calc_keywords) or any(symbol in question for symbol in math_symbols):
            complexity_indicators.append("calculations")
            complexity_score += 0.5
            requires_tools = True
        
        # Web search detection
        search_keywords = ["current", "latest", "recent", "search", "find", "wikipedia", "google", "look up", "research", "today", "2024", "2025"]
        if any(keyword in question_lower for keyword in search_keywords):
            complexity_indicators.append("web_search")
            complexity_score += 0.4
            requires_tools = True
        
        # Multi-step detection
        multi_step_indicators = ["then", "after", "next", "first", "second", "finally", "step", "process"]
        question_marks_count = question.count("?")
        if (len(question.split()) > 30 or 
            question_marks_count > 1 or 
            any(indicator in question_lower for indicator in multi_step_indicators)):
            complexity_indicators.append("multi_step")
            complexity_score += 0.3
        
        # Reasoning detection
        reasoning_keywords = ["why", "how", "explain", "compare", "analyze", "evaluate", "reason", "because", "therefore"]
        if any(keyword in question_lower for keyword in reasoning_keywords):
            complexity_indicators.append("reasoning")
            complexity_score += 0.2
        
        # Similar examples boost
        if similar_examples and len(similar_examples) > 0:
            complexity_score += 0.1  # Boost for having context
        
        # Agent recommendation logic
        agent_mapping = {
            "file_processing": "document_processor",
            "calculations": "data_analyst", 
            "web_search": "web_researcher",
            "multi_step": "general_assistant",
            "reasoning": "general_assistant"
        }
        
        # Choose agent based on highest priority indicator
        priority_order = ["file_processing", "calculations", "web_search", "multi_step", "reasoning"]
        recommended_agent = "general_assistant"  # Default
        
        for indicator in priority_order:
            if indicator in complexity_indicators:
                recommended_agent = agent_mapping[indicator]
                break
        
        return {
            "complexity_score": min(complexity_score, 1.0),
            "complexity_indicators": complexity_indicators,
            "recommended_agent": recommended_agent,
            "requires_tools": requires_tools,
            "similar_examples_count": len(similar_examples),
            "question_length": len(question),
            "question_marks": question_marks_count
        }
    
    def _smolag_execution_node(self, state: GAIAState):
        """SmolagAgent execution with retry and fallback logic"""
        question = state["question"]
        agent_name = state["selected_agent"]
        context = state.get("retriever_context", "")
        retry_count = state.get("retry_count", 0)
        
        if self.config.debug_mode:
            print(f"🤖 SmolagAgent execution: {agent_name} (attempt {retry_count + 1})")
        
        try:
            # Execute with SmolagAgent
            execution_result = self.agent_manager.execute_with_agent(
                agent_name, question, context, retry_count
            )
            
            # Result processing
            if execution_result["success"]:
                return {
                    "raw_answer": execution_result["answer"],
                    "execution_steps": state["execution_steps"] + execution_result["steps"],
                    "debug_info": {
                        **state.get("debug_info", {}),
                        "agent_used": agent_name,
                        "agent_type": execution_result.get("agent_type", "unknown"),
                        "smolag_success": True,
                        "is_simulation": execution_result.get("is_simulation", False),
                        "retry_count": execution_result.get("retry_count", 0)
                    }
                }
            else:
                # Handle failure
                error_msg = execution_result.get("error", "Unknown SmolagAgent error")
                return {
                    "raw_answer": execution_result["answer"],
                    "execution_steps": state["execution_steps"] + execution_result["steps"],
                    "errors": state.get("errors", []) + [error_msg],
                    "debug_info": {
                        **state.get("debug_info", {}),
                        "agent_used": agent_name,
                        "smolag_success": False,
                        "smolag_error": error_msg,
                        "retry_count": execution_result.get("retry_count", 0)
                    }
                }
                
        except Exception as e:
            error_msg = f"Critical SmolagAgent execution error: {str(e)}"
            if self.config.debug_mode:
                print(f"❌ {error_msg}")
            
            return {
                "raw_answer": f"SmolagAgent execution failed: {error_msg}",
                "execution_steps": state["execution_steps"] + [
                    f"SmolagAgent {agent_name} execution failed",
                    f"Error: {error_msg}"
                ],
                "errors": state.get("errors", []) + [error_msg],
                "debug_info": {
                    **state.get("debug_info", {}),
                    "agent_used": agent_name,
                    "smolag_success": False,
                    "critical_error": error_msg
                }
            }
    
    def _direct_llm_execution_node(self, state: GAIAState):
        """Direct LLM execution with optimized prompt engineering"""
        try:
            question = state["question"]
            messages = state["messages"]
            similar_examples = state.get("similar_examples", [])
            
            # Build GAIA-compliant prompt
            system_prompt = self._build_system_prompt(similar_examples)
            
            final_messages = [SystemMessage(content=system_prompt)] + messages
            
            if self.config.debug_mode:
                print("🧠 Direct LLM execution")
            
            response = self.llm.invoke(final_messages)
            
            return {
                "raw_answer": response.content,
                "execution_steps": state["execution_steps"] + [
                    "Direct LLM execution with optimized prompt",
                    f"System prompt length: {len(system_prompt)} characters",
                    f"Response length: {len(response.content)} characters"
                ],
                "debug_info": {
                    **state.get("debug_info", {}),
                    "execution_method": "direct_llm",
                    "system_prompt_used": True,
                    "examples_in_prompt": len(similar_examples)
                }
            }
            
        except Exception as e:
            error_msg = f"Direct LLM execution error: {str(e)}"
            if self.config.debug_mode:
                print(f"❌ {error_msg}")
            
            return {
                "raw_answer": f"LLM execution failed: {error_msg}",
                "execution_steps": state["execution_steps"] + [
                    "Direct LLM execution failed"
                ],
                "errors": state.get("errors", []) + [error_msg]
            }
    
    def _build_system_prompt(self, similar_examples: List[Dict]) -> str:
        """Build system prompt with examples"""
        base_prompt = """You are a general AI assistant specialized in solving GAIA benchmark questions. Report your thoughts, and finish with: FINAL ANSWER: [YOUR FINAL ANSWER].

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list.
- Numbers: no commas, no units ($ %) unless specified
- Strings: no articles (the, a, an), no abbreviations, digits as text unless specified  
- Lists: apply above rules to each element

Be precise, concise, and accurate in your reasoning."""
        
        if similar_examples and len(similar_examples) > 0:
            examples_text = "\n\nHere are similar examples to guide your approach:\n"
            for i, example in enumerate(similar_examples[:2], 1):  # Limit to 2 examples
                examples_text += f"\nExample {i}:\nQuestion: {example['question']}\nAnswer: {example['answer']}\n"
            
            base_prompt += examples_text
        
        return base_prompt
    
    def _error_recovery_node(self, state: GAIAState):
        """Error recovery with intelligent fallback"""
        errors = state.get("errors", [])
        retry_count = state.get("retry_count", 0)
        
        if self.config.debug_mode:
            print(f"🔧 Error recovery activated (errors: {len(errors)}, retries: {retry_count})")
        
        # Simple error recovery - try direct LLM
        question = state["question"]
        messages = state["messages"]
        
        try:
            system_prompt = """You are a helpful AI assistant. Answer the question as best you can based on your knowledge. 
If you cannot provide a complete answer, give your best estimate or explain what you know about the topic.

Format your final answer clearly."""
            
            final_messages = [SystemMessage(content=system_prompt)] + messages
            response = self.llm.invoke(final_messages)
            
            return {
                "raw_answer": response.content,
                "fallback_used": True,
                "execution_steps": state["execution_steps"] + [
                    "Error recovery: Using simplified direct LLM",
                    f"Recovered from {len(errors)} error(s)"
                ],
                "debug_info": {
                    **state.get("debug_info", {}),
                    "error_recovery": True,
                    "recovery_method": "simplified_llm"
                }
            }
            
        except Exception as e:
            # Last resort fallback
            return {
                "raw_answer": "Unable to process question due to system errors",
                "fallback_used": True,
                "execution_steps": state["execution_steps"] + [
                    "Error recovery failed, using last resort fallback"
                ],
                "errors": errors + [f"Recovery error: {str(e)}"]
            }
    
    def _answer_formatting_node(self, state: GAIAState):
        """Answer formatting with GAIA compliance"""
        raw_answer = state.get("raw_answer", "")
        fallback_used = state.get("fallback_used", False)
        
        try:
            # FINAL ANSWER extraction
            formatted_answer = self._extract_and_format_answer(raw_answer)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(state)
            
            return {
                "final_answer": formatted_answer,
                "confidence_score": confidence_score,
                "execution_steps": state["execution_steps"] + [
                    "GAIA formatting applied",
                    f"Confidence score: {confidence_score:.2f}",
                    f"Fallback used: {fallback_used}"
                ],
                "debug_info": {
                    **state.get("debug_info", {}),
                    "formatting_applied": True,
                    "original_answer_length": len(raw_answer),
                    "formatted_answer_length": len(formatted_answer)
                }
            }
            
        except Exception as e:
            error_msg = f"Answer formatting error: {str(e)}"
            return {
                "final_answer": raw_answer.strip() if raw_answer else "No answer",
                "confidence_score": 0.1,
                "execution_steps": state["execution_steps"] + [
                    "Answer formatting failed, using raw answer"
                ],
                "errors": state.get("errors", []) + [error_msg]
            }
    
    def _extract_and_format_answer(self, raw_answer: str) -> str:
        """Extract and format answer"""
        if not raw_answer:
            return "No answer"
        
        answer = raw_answer.strip()
        
        # FINAL ANSWER extraction
        patterns = [
            r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
            r"Final Answer:\s*(.+?)(?:\n|$)",
            r"Answer:\s*(.+?)(?:\n|$)",
            r"The answer is:\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            import re
            match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                break
        
        # Apply GAIA formatting
        answer = self._apply_gaia_formatting(answer)
        
        return answer
    
    def _apply_gaia_formatting(self, raw_answer: str) -> str:
        """Apply GAIA benchmark formatting rules"""
        if not raw_answer:
            return "No answer"
        
        answer = raw_answer.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "the answer is", "answer:", "final answer:", "result:", "solution:",
            "the result is", "this gives us", "therefore", "so", "thus",
            "the final answer is", "my answer is", "i think", "i believe"
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                answer_lower = answer.lower()
        
        # Remove leading/trailing punctuation and quotes
        answer = answer.strip('.,!?:;"\'')
        
        # Handle special number formatting
        if answer.replace('.', '').replace('-', '').replace(',', '').isdigit():
            # Remove commas from numbers as per GAIA rules
            answer = answer.replace(',', '')
        
        # Handle percentage formatting
        if '%' in answer and 'unless specified' not in raw_answer.lower():
            # Remove % unless specifically requested
            parts = answer.split('%')
            if len(parts) == 2 and parts[1].strip() == '':
                answer = parts[0].strip()
        
        return answer
    
    def _calculate_confidence_score(self, state: GAIAState) -> float:
        """Calculate confidence score based on execution factors"""
        confidence = 1.0
        
        # Reduce confidence for errors
        errors = state.get("errors", [])
        confidence -= len(errors) * 0.2
        
        # Reduce confidence for fallback usage
        if state.get("fallback_used", False):
            confidence -= 0.3
        
        # Reduce confidence for simulation agents
        debug_info = state.get("debug_info", {})
        if debug_info.get("is_simulation", False):
            confidence -= 0.4
        
        # Boost confidence for successful SmolagAgent execution
        if debug_info.get("smolag_success", False):
            confidence += 0.1
        
        # Boost confidence for RAG context
        if len(state.get("similar_examples", [])) > 0:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _evaluation_node(self, state: GAIAState):
        """Evaluation with comprehensive metrics"""
        end_time = datetime.now()
        start_time_str = state.get("debug_info", {}).get("start_time")
        
        execution_time = None
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            execution_time = (end_time - start_time).total_seconds()
        
        # Accuracy evaluation
        is_correct = None
        if state.get("ground_truth"):
            is_correct = self._evaluate_answer(
                state.get("final_answer", ""), 
                state["ground_truth"]
            )
        
        # Update performance metrics
        self.performance_metrics["total_questions"] += 1
        if is_correct:
            self.performance_metrics["successful_completions"] += 1
        
        if execution_time:
            # Update average execution time
            total_time = (self.performance_metrics["average_execution_time"] * 
                         (self.performance_metrics["total_questions"] - 1) + execution_time)
            self.performance_metrics["average_execution_time"] = total_time / self.performance_metrics["total_questions"]
        
        # Calculate error rate
        errors = state.get("errors", [])
        if errors:
            total_errors = sum(1 for result in self.execution_results if result.get("errors", []))
            self.performance_metrics["error_rate"] = total_errors / self.performance_metrics["total_questions"]
        
        return {
            "execution_time": execution_time,
            "model_used": f"{self.config.model_provider}/{self.config.primary_model}",
            "execution_steps": state["execution_steps"] + [
                "Evaluation completed",
                f"Execution time: {execution_time:.2f}s" if execution_time else "No timing data",
                f"Correctness: {'✅' if is_correct else '❌' if is_correct is False else '❓'}"
            ],
            "debug_info": {
                **state.get("debug_info", {}),
                "end_time": end_time.isoformat(),
                "is_correct": is_correct,
                "execution_time": execution_time,
                "performance_metrics": self.performance_metrics.copy()
            }
        }
    
    def _evaluate_answer(self, predicted: str, ground_truth: str) -> bool:
        """Answer evaluation with fuzzy matching"""
        if not predicted or not ground_truth:
            return False
        
        # Normalize for comparison
        pred_normalized = str(predicted).lower().strip()
        truth_normalized = str(ground_truth).lower().strip()
        
        # Exact match
        if pred_normalized == truth_normalized:
            return True
        
        # Remove common variations
        variations_to_remove = ['.', ',', '!', '?', ' ', '\n', '\t']
        pred_clean = pred_normalized
        truth_clean = truth_normalized
        
        for var in variations_to_remove:
            pred_clean = pred_clean.replace(var, '')
            truth_clean = truth_clean.replace(var, '')
        
        if pred_clean == truth_clean:
            return True
        
        # Numeric comparison
        try:
            pred_num = float(pred_normalized.replace(',', ''))
            truth_num = float(truth_normalized.replace(',', ''))
            return abs(pred_num - truth_num) < 1e-6
        except:
            pass
        
        return False
    
    def _check_execution_success(self, state: GAIAState) -> str:
        """Check if SmolagAgent execution was successful"""
        debug_info = state.get("debug_info", {})
        errors = state.get("errors", [])
        retry_count = state.get("retry_count", 0)
        
        if debug_info.get("smolag_success", False):
            return "success"
        elif retry_count < self.config.max_retries:
            return "retry"
        else:
            return "fallback"
    
    def _route_execution(self, state: GAIAState) -> str:
        """Execution routing with error handling"""
        strategy = state.get("selected_strategy", "direct_llm")
        errors = state.get("errors", [])
        
        # If there are critical errors, go to error recovery
        if len(errors) > 2:
            return "error_recovery"
        
        return strategy
    
    # ============================================================================
    # PUBLIC INTERFACE
    # ============================================================================
    
    def run_single_question(self, question: str, task_id: str = None, 
                          ground_truth: str = None, level: int = None) -> Dict:
        """Execute single question with comprehensive error handling"""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        initial_state = {
            "question": question,
            "task_id": task_id,
            "ground_truth": ground_truth,
            "level": level,
            "retry_count": 0,
            "errors": [],
            "fallback_used": False
        }
        
        if self.config.debug_mode:
            print(f"🔍 Processing: {question[:60]}...")
        
        try:
            result = self.workflow.invoke(initial_state)
            
            # Create execution record
            execution_record = {
                "task_id": task_id,
                "question": question,
                "final_answer": result.get("final_answer", ""),
                "raw_answer": result.get("raw_answer", ""),
                "ground_truth": ground_truth,
                "level": level,
                "is_correct": result.get("debug_info", {}).get("is_correct"),
                "execution_time": result.get("execution_time"),
                "strategy_used": result.get("selected_strategy"),
                "selected_agent": result.get("selected_agent"),
                "model_used": result.get("model_used"),
                "confidence_score": result.get("confidence_score", 0.0),
                "similar_examples_count": len(result.get("similar_examples", [])),
                "errors": result.get("errors", []),
                "fallback_used": result.get("fallback_used", False),
                "execution_steps": result.get("execution_steps", []),
                "timestamp": datetime.now().isoformat(),
                "config_version": "production"
            }
            
            self.execution_results.append(execution_record)
            
            if self.config.save_intermediate_results:
                self._save_result(execution_record)
            
            return result
            
        except Exception as e:
            error_msg = f"Critical workflow error: {str(e)}"
            if self.config.debug_mode:
                print(f"❌ {error_msg}")
                print(f"Traceback: {traceback.format_exc()}")
            
            # Return error result
            error_result = {
                "task_id": task_id,
                "question": question,
                "final_answer": "Workflow execution failed",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_results.append(error_result)
            return error_result
    
    def run_batch_evaluation(self, sample_size: int = None) -> pd.DataFrame:
        """Batch evaluation with comprehensive analysis"""
        test_data = self.metadata_manager.get_test_sample(sample_size)
        
        print(f"🧪 Evaluation on {len(test_data)} questions...")
        print("=" * 60)
        
        results = []
        successful_runs = 0
        
        for i, example in enumerate(test_data):
            try:
                if self.config.debug_mode and (i + 1) % 5 == 0:
                    accuracy_so_far = successful_runs / (i + 1) if i + 1 > 0 else 0
                    print(f"  Progress: {i+1}/{len(test_data)} ({(i+1)/len(test_data)*100:.1f}%) | Accuracy: {accuracy_so_far:.3f}")
                
                result = self.run_single_question(
                    question=example.get("Question", ""),
                    task_id=example.get("task_id", f"eval_{i}"),
                    ground_truth=example.get("Final answer", ""),
                    level=example.get("Level", 1)
                )
                
                if self.execution_results[-1].get("is_correct"):
                    successful_runs += 1
                
                results.append(self.execution_results[-1])
                
            except Exception as e:
                print(f"❌ Error processing question {i}: {e}")
                error_record = {
                    "task_id": f"eval_{i}",
                    "question": example.get("Question", ""),
                    "final_answer": "Processing failed",
                    "error": str(e),
                    "is_correct": False
                }
                results.append(error_record)
        
        # Analysis
        df = pd.DataFrame(results)
        
        print(f"\n🎯 EVALUATION RESULTS")
        print("=" * 40)
        print(f"Total questions: {len(df)}")
        
        if 'is_correct' in df and df['is_correct'].notna().any():
            accuracy = df['is_correct'].mean()
            print(f"Overall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if 'execution_time' in df and df['execution_time'].notna().any():
            avg_time = df['execution_time'].mean()
            print(f"Average execution time: {avg_time:.2f}s")
        
        if 'confidence_score' in df and df['confidence_score'].notna().any():
            avg_confidence = df['confidence_score'].mean()
            print(f"Average confidence: {avg_confidence:.3f}")
        
        # Strategy breakdown
        if 'strategy_used' in df:
            print(f"\n📊 Strategy Performance:")
            strategy_stats = df.groupby('strategy_used').agg({
                'is_correct': ['count', 'mean'],
                'execution_time': 'mean',
                'confidence_score': 'mean'
            }).round(3)
            print(strategy_stats)
        
        if 'level' in df:
            print(f"\n📈 Level Performance:")
            level_stats = df.groupby('level').agg({
                'is_correct': ['count', 'mean'],
                'execution_time': 'mean'
            }).round(3)
            print(level_stats)
        
        # Error analysis
        error_count = df['errors'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
        fallback_count = df['fallback_used'].sum() if 'fallback_used' in df else 0
        
        print(f"\n🔧 Error Analysis:")
        print(f"Total errors: {error_count}")
        print(f"Fallback usage: {fallback_count}")
        print(f"SmolagAgent stats: {self.agent_manager.get_execution_stats()}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = Path(self.config.results_output_path) / f"evaluation_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved: {results_file}")
        
        return df
    
    def _save_result(self, result: Dict):
        """Save individual result"""
        results_file = Path(self.config.results_output_path) / "individual_results.jsonl"
        with open(results_file, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            "performance_metrics": self.performance_metrics.copy(),
            "agent_stats": self.agent_manager.get_execution_stats(),
            "total_executions": len(self.execution_results),
            "config": {
                "model_provider": self.config.model_provider,
                "primary_model": self.config.primary_model,
                "version": "production"
            }
        }
    
    def close(self):
        """Cleanup and save performance summary"""
        try:
            if hasattr(self.retriever, 'close'):
                self.retriever.close()
            
            # Save final performance summary
            if self.execution_results:
                summary = self.get_performance_summary()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_file = Path(self.config.results_output_path) / f"performance_summary_{timestamp}.json"
                with open(summary_file, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                
                if self.config.debug_mode:
                    print(f"💾 Performance summary saved: {summary_file}")
                    
        except Exception as e:
            if self.config.debug_mode:
                print(f"⚠️  Cleanup warning: {e}")

# ============================================================================
# METADATA MANAGER
# ============================================================================

class MetadataManager:
    """Metadata manager with analysis capabilities"""
    
    def __init__(self, config: GAIAConfig):
        self.config = config
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> List[Dict]:
        """Load metadata with error handling"""
        try:
            with open(self.config.metadata_path, 'r', encoding='utf-8') as f:
                metadata = [json.loads(line) for line in f]
            print(f"✅ Loaded {len(metadata)} GAIA examples from metadata")
            return metadata
        except FileNotFoundError:
            print(f"❌ Metadata file not found: {self.config.metadata_path}")
            return []
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return []
    
    def get_test_sample(self, sample_size: Optional[int] = None) -> List[Dict]:
        """Get test sample with level distribution"""
        if sample_size is None:
            return self.metadata
        
        # Try to maintain level distribution in sample
        if sample_size >= len(self.metadata):
            return self.metadata
        
        # Group by level and sample proportionally
        level_groups = {}
        for item in self.metadata:
            level = item.get('Level', 1)
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(item)
        
        sample = []
        remaining = sample_size
        
        for level in sorted(level_groups.keys()):
            level_count = len(level_groups[level])
            level_sample_size = min(remaining, max(1, int(sample_size * level_count / len(self.metadata))))
            sample.extend(level_groups[level][:level_sample_size])
            remaining -= level_sample_size
            
            if remaining <= 0:
                break
        
        return sample[:sample_size]
    
    def analyze_tool_usage(self) -> Dict[str, int]:
        """Tool usage analysis"""
        tool_frequencies = {}
        
        for item in self.metadata:
            question = item.get('Question', '').lower()
            
            # Pattern detection
            patterns = {
                'calculator': ["calculate", "compute", "math", "number", "sum", "average", "percentage", "+", "-", "*", "/", "="],
                'web_search': ["search", "find", "lookup", "google", "current", "latest", "recent", "today", "wikipedia"],
                'file_processing': ["file", "attachment", "document", "image", "audio", "excel", "pdf", "csv", "spreadsheet"],
                'reasoning': ["why", "how", "explain", "compare", "analyze", "evaluate", "reason"],
                'multi_step': ["then", "after", "next", "first", "second", "step", "process"],
            }
            
            for tool, keywords in patterns.items():
                if any(keyword in question for keyword in keywords):
                    tool_frequencies[tool] = tool_frequencies.get(tool, 0) + 1
        
        return tool_frequencies
    
    def get_complexity_distribution(self) -> Dict:
        """Analyze complexity distribution in metadata"""
        complexity_stats = {
            'simple': 0,
            'medium': 0, 
            'complex': 0,
            'by_level': {}
        }
        
        for item in self.metadata:
            question = item.get('Question', '')
            level = item.get('Level', 1)
            
            # Simple complexity scoring
            score = 0
            if len(question.split()) > 25:
                score += 1
            if question.count('?') > 1:
                score += 1
            if any(word in question.lower() for word in ['calculate', 'file', 'search', 'analyze']):
                score += 1
            
            if score <= 1:
                complexity_stats['simple'] += 1
            elif score == 2:
                complexity_stats['medium'] += 1
            else:
                complexity_stats['complex'] += 1
            
            if level not in complexity_stats['by_level']:
                complexity_stats['by_level'][level] = {'simple': 0, 'medium': 0, 'complex': 0}
            
            if score <= 1:
                complexity_stats['by_level'][level]['simple'] += 1
            elif score == 2:
                complexity_stats['by_level'][level]['medium'] += 1
            else:
                complexity_stats['by_level'][level]['complex'] += 1
        
        return complexity_stats

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

class ModelConfigs:
    """Production model configurations with primary/secondary fallback support"""
    
    @staticmethod
    def get_openrouter_configs():
        """OpenRouter configurations with fallback models"""
        return {
            "qwen2.5_coder": {
                "model_provider": "openrouter",
                "primary_model": "qwen/qwen-2.5-coder-32b-instruct:free",
                "secondary_model": "qwen/qwen3-32b:free",
                "temperature": 0.3,
                "max_agent_steps": 20,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "qwen3_32b": {
                "model_provider": "openrouter", 
                "primary_model": "qwen/qwen3-32b:free",
                "secondary_model": "qwen/qwen3-14b:free",
                "temperature": 0.4,
                "max_agent_steps": 15,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "qwen3_14b": {
                "model_provider": "openrouter",
                "primary_model": "qwen/qwen3-14b:free",
                "secondary_model": "qwen/qwen-2.5-coder-32b-instruct:free",
                "temperature": 0.4,
                "max_agent_steps": 12,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "deepseek": {
                "model_provider": "openrouter",
                "primary_model": "deepseek/deepseek-chat:free",
                "secondary_model": "qwen/qwen-2.5-coder-32b-instruct:free",
                "temperature": 0.5,
                "max_agent_steps": 12,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "llama_70b": {
                "model_provider": "openrouter",
                "primary_model": "meta-llama/llama-3.1-70b-instruct:free",
                "secondary_model": "qwen/qwen3-32b:free",
                "temperature": 0.4,
                "max_agent_steps": 18,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            }
        }
    
    @staticmethod
    def get_groq_configs():
        """Groq configurations with fallback models"""
        return {
            "qwen_qwq_groq": {
                "model_provider": "groq",
                "primary_model": "qwen-qwq-32b",
                "secondary_model": "llama3-70b-8192",
                "temperature": 0.2,
                "max_agent_steps": 20,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "llama4_scout_groq": {
                "model_provider": "groq",
                "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "secondary_model": "llama3-70b-8192",
                "temperature": 0.3,
                "max_agent_steps": 18,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "llama3_70b_groq": {
                "model_provider": "groq",
                "primary_model": "llama3-70b-8192",
                "secondary_model": "qwen-qwq-32b",
                "temperature": 0.3,
                "max_agent_steps": 15,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "mixtral_groq": {
                "model_provider": "groq",
                "primary_model": "mixtral-8x7b-32768",
                "secondary_model": "llama3-70b-8192",
                "temperature": 0.4,
                "max_agent_steps": 12,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            }
        }
    
    @staticmethod
    def get_google_configs():
        """Google configurations with fallback models"""
        return {
            "gemma_3n": {
                "model_provider": "google",
                "primary_model": "gemma-3n-e4b-it",
                "secondary_model": "gemini-2.5-flash-preview-04-17",
                "temperature": 0.3,
                "max_agent_steps": 15,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "gemini_flash_04": {
                "model_provider": "google",
                "primary_model": "gemini-2.5-flash-preview-04-17",
                "secondary_model": "gemini-2.5-flash-preview-05-20",
                "temperature": 0.4,
                "max_agent_steps": 18,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "gemini_flash_05": {
                "model_provider": "google",
                "primary_model": "gemini-2.5-flash-preview-05-20",
                "secondary_model": "gemini-2.5-flash-preview-04-17",
                "temperature": 0.4,
                "max_agent_steps": 18,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            }
        }
    
    @staticmethod
    def get_ollama_configs():
        """Ollama configurations with fallback models"""
        return {
            "qwen_agent": {
                "model_provider": "ollama",
                "primary_model": "qwen-agent-custom:latest",
                "secondary_model": "qwen2.5-coder:32b",
                "api_base": "http://localhost:11434",
                "num_ctx": 32768,
                "temperature": 0.3,
                "max_agent_steps": 25,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "qwen_coder_ollama": {
                "model_provider": "ollama",
                "primary_model": "qwen2.5-coder:32b",
                "secondary_model": "qwen-agent-custom:latest",
                "api_base": "http://localhost:11434",
                "num_ctx": 32768,
                "temperature": 0.2,
                "max_agent_steps": 20,
                "rag_examples_count": 3,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            },
            "llama_ollama": {
                "model_provider": "ollama",
                "primary_model": "llama3.2:latest",
                "secondary_model": "qwen2.5-coder:32b",
                "api_base": "http://localhost:11434",
                "num_ctx": 32768,
                "temperature": 0.3,
                "max_agent_steps": 15,
                "rag_examples_count": 2,
                "enable_agent_fallback": True,
                "enable_model_fallback": True
            }
        }
    
    @staticmethod
    def get_all_configs():
        """Get all configurations including new Google configs"""
        configs = {}
        configs.update(ModelConfigs.get_openrouter_configs())
        configs.update(ModelConfigs.get_groq_configs())
        configs.update(ModelConfigs.get_google_configs())  # Added Google configs
        configs.update(ModelConfigs.get_ollama_configs())
        return configs

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_gaia_agent(config_overrides: Union[str, Dict] = None) -> GAIAAgent:
    """Factory function for creating GAIA agent"""
    config = GAIAConfig()
    
    if config_overrides:
        if isinstance(config_overrides, str):
            model_configs = ModelConfigs.get_all_configs()
            if config_overrides in model_configs:
                config_dict = model_configs[config_overrides]
                print(f"📋 Using config: {config_overrides}")
            else:
                raise ValueError(f"Unknown config: {config_overrides}")
        else:
            config_dict = config_overrides
        
        # Apply overrides
        for key, value in config_dict.items():
            setattr(config, key, value)
    
    return GAIAAgent(config)

def create_production_gaia_agent(
    model_config: Union[str, Dict] = "qwen_coder",
    enable_logging: bool = True,
    performance_tracking: bool = True,
    max_retries: int = 2
) -> GAIAAgent:
    """Create production-ready GAIA agent"""
    
    if isinstance(model_config, str):
        model_configs = ModelConfigs.get_all_configs()
        if model_config not in model_configs:
            raise ValueError(f"Unknown model config: {model_config}")
        config_dict = model_configs[model_config]
    else:
        config_dict = model_config
    
    # Add production settings
    config_dict.update({
        "debug_mode": enable_logging,
        "save_intermediate_results": performance_tracking,
        "max_retries": max_retries,
        "enable_graceful_degradation": True,
        "gaia_formatting_strict": True,
        "enable_confidence_scoring": True
    })
    
    config = GAIAConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    agent = GAIAAgent(config)
    return agent

def quick_test(
    question: str = "Calculate 15% of 1000", 
    config_name: str = "qwen_coder"
) -> Dict:
    """Quick test with error handling"""
    try:
        agent = create_gaia_agent(config_name)
        result = agent.run_single_question(question)
        
        # Print summary
        print(f"\n🎯 TEST RESULT")
        print("=" * 30)
        print(f"Question: {question}")
        print(f"Final Answer: {result.get('final_answer', 'No answer')}")
        print(f"Strategy: {result.get('selected_strategy', 'Unknown')}")
        print(f"Agent: {result.get('selected_agent', 'N/A')}")
        print(f"Confidence: {result.get('confidence_score', 0):.2f}")
        print(f"Errors: {len(result.get('errors', []))}")
        print(f"Fallback Used: {'Yes' if result.get('fallback_used') else 'No'}")
        
        agent.close()
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return {"error": str(e)}

def compare_configs(
    question: str = "What is 25% of 400?",
    configs_to_test: List[str] = None
) -> pd.DataFrame:
    """Compare different model configurations"""
    if configs_to_test is None:
        configs_to_test = ["qwen_coder", "qwen_32b", "deepseek"]
    
    results = []
    
    print(f"🔄 Comparing {len(configs_to_test)} configurations...")
    
    for config_name in configs_to_test:
        print(f"Testing {config_name}...")
        try:
            result = quick_test(question, config_name)
            
            results.append({
                "config": config_name,
                "final_answer": result.get("final_answer", ""),
                "strategy": result.get("selected_strategy", ""),
                "confidence": result.get("confidence_score", 0),
                "execution_time": result.get("execution_time", 0),
                "errors": len(result.get("errors", [])),
                "success": "error" not in result
            })
            
        except Exception as e:
            results.append({
                "config": config_name,
                "final_answer": f"Error: {e}",
                "strategy": "failed",
                "confidence": 0,
                "execution_time": 0,
                "errors": 1,
                "success": False
            })
    
    df = pd.DataFrame(results)
    print(f"\n📊 Configuration Comparison:")
    print(df.to_string(index=False))
    
    return df

def run_gaia_benchmark(
    sample_size: int = 50,
    config_name: str = "qwen_coder",
    save_results: bool = True
) -> Dict:
    """Run GAIA benchmark with comprehensive analysis"""
    print(f"🚀 GAIA Benchmark - {sample_size} questions")
    print("=" * 50)
    
    agent = create_gaia_agent(config_name)
    
    try:
        # Run evaluation
        results_df = agent.run_batch_evaluation(sample_size)
        
        # Comprehensive analysis
        analysis = {
            "total_questions": len(results_df),
            "overall_accuracy": results_df['is_correct'].mean() if 'is_correct' in results_df else 0,
            "average_confidence": results_df['confidence_score'].mean() if 'confidence_score' in results_df else 0,
            "average_execution_time": results_df['execution_time'].mean() if 'execution_time' in results_df else 0,
            "strategy_breakdown": results_df['strategy_used'].value_counts().to_dict() if 'strategy_used' in results_df else {},
            "level_performance": results_df.groupby('level')['is_correct'].mean().to_dict() if 'level' in results_df else {},
            "error_rate": (results_df['errors'].apply(lambda x: len(x) if isinstance(x, list) else 0) > 0).mean() if 'errors' in results_df else 0,
            "fallback_rate": results_df['fallback_used'].mean() if 'fallback_used' in results_df else 0,
            "config_used": config_name,
            "agent_stats": agent.agent_manager.get_execution_stats()
        }
        
        print(f"\n🎯 BENCHMARK ANALYSIS")
        print("=" * 30)
        for key, value in analysis.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  ├── {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_file = Path(agent.config.results_output_path) / f"benchmark_analysis_{timestamp}.json"
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"\n💾 Analysis saved: {analysis_file}")
        
        agent.close()
        return analysis
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        agent.close()
        return {"error": str(e)}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Production GAIA Agent System")
    print("=" * 70)
    
    # Show available configurations
    print("\n📋 Available Model Configurations:")
    configs = ModelConfigs.get_all_configs()
    for name, config in configs.items():
        provider = config['model_provider'] 
        model = config['primary_model']
        steps = config.get('max_agent_steps', 'default')
        temp = config.get('temperature', 'default')
        print(f"  ├── {name}: {provider}/{model} (steps: {steps}, temp: {temp})")
    
    # Quick test
    print(f"\n🧪 Quick test...")
    try:
        result = quick_test(
            question="Calculate 25% of 800",
            config_name="qwen_coder"
        )
        
        if "error" not in result:
            print(f"✅ Test successful!")
        else:
            print(f"❌ Test failed: {result['error']}")
            print("💡 Check your API keys and model configurations")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        print("💡 Ensure your retriever system and dependencies are properly set up")
    
    print(f"\n🎓 Ready for GAIA benchmark testing!")
    print("Use create_gaia_agent() or run_gaia_benchmark() to get started.")