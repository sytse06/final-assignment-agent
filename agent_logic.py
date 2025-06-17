# agent_logic.py
# GAIA Agent System using SmolagAgents Manager Pattern

import os
import uuid
import re
import backoff
from typing import TypedDict, Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime, timezone

# Core dependencies
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# SmolagAgents imports
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    GoogleSearchTool,
    VisitWebpageTool,
    AgentLogger,
    LogLevel
)

# Import agent context system
from agent_context import (
    ContextVariableFlow,
    create_context_aware_tools,
    with_context,
    ensure_task_context
)

# Import retriever system
from dev_retriever import load_gaia_retriever

# Import logging
from agent_logging import AgentLoggingSetup

# Import custom and Langchain tools
try:
    from tools import GetAttachmentTool, ContentRetrieverTool
    CUSTOM_TOOLS_AVAILABLE = True
except ImportError:
    print("⚠️  Custom tools not available - using base tools only")
    CUSTOM_TOOLS_AVAILABLE = False
try:
    from tools import ALL_LANGCHAIN_TOOLS
    LANGCHAIN_TOOLS_AVAILABLE = True
    print("✅ LangChain tools imported successfully")
except ImportError as e:
    print(f"⚠️  LangChain tools not available: {e}")
    ALL_LANGCHAIN_TOOLS = []
    LANGCHAIN_TOOLS_AVAILABLE = False

import openai

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
    planning_interval: int = 3
        
    # Routing settings
    enable_smart_routing: bool = True
    skip_rag_for_simple: bool = True
    
    # Context settings
    enable_context_bridge: bool = True
    context_bridge_debug: bool = True
    
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
    task_id: Optional[str]
    question: str
    steps: List[str]
    raw_answer: Optional[str]
    final_answer: Optional[str]
    similar_examples: Optional[List[Dict]]
    complexity: Optional[str]

# ============================================================================
# LLM RETRY LOGIC
# ============================================================================

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, max_tries=3)
def llm_invoke_with_retry(llm, messages):
    """Retry logic for LangChain LLM calls"""
    return llm.invoke(messages)

# ============================================================================
# COMPLEXITY DETECTION HELPERS
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

# ============================================================================
# GAIA FORMATTING VALIDATION
# ============================================================================

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
    """GAIA agent using SmolagAgents manager pattern with context sharing"""
    
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
        print(f"🔍 DEBUG: shared_tools keys: {list(self.shared_tools.keys())}")
        
        # Create context-aware tools (now returns proper Tool objects)
        if config.enable_context_bridge and self.shared_tools:
            self.context_aware_tools = create_context_aware_tools(self.shared_tools)
            print(f"🌉 Context-aware tools created: {len(self.context_aware_tools)} tools")
        else:
            self.context_aware_tools = []
        
        # Create specialist agents
        self.specialists = self._create_specialist_agents()
        
        # Create manager agent
        self.manager = self._create_manager_agent()
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        print("🚀 GAIA Manager Agent initialized!")
    
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
            
    def _create_specialist_agents(self):
        """Create specialist agents with context-aware tools"""
        
        logger = None
        if self.logging and hasattr(self.logging, 'logger'):
            logger = self.logging.logger

        try:
            temporal_ctx = self._get_current_temporal_context()
            if not isinstance(temporal_ctx, dict):
                raise ValueError("Temporal context is not a dictionary")
                
            web_researcher_prompt = f"""You are a web research specialist.

        CURRENT TEMPORAL CONTEXT:
        - Today's date: {temporal_ctx['current_date']}
        - Current year: {temporal_ctx['current_year']}
        - Current month: {temporal_ctx['current_month']}

        CRITICAL: When searching for "current", "recent", "latest", or "now" information, always use {temporal_ctx['current_year']} in your search queries. Do NOT use 2023.

        Your role is to find accurate, up-to-date information using web search tools."""

        except Exception as e:
            print(f"⚠️  Temporal context failed, using fallback: {e}")

        specialists = {}
        env_tools = []

        # Add LangChain research tools
        if LANGCHAIN_TOOLS_AVAILABLE:
            env_tools.extend(ALL_LANGCHAIN_TOOLS)
            print(f"✅ Added {len(ALL_LANGCHAIN_TOOLS)} LangChain tools")
        
        # Data Analyst - computation focus, no file tools
        data_tools = env_tools.copy()  # Base tools only
        
        specialists["data_analyst"] = CodeAgent(
            name="data_analyst",
            description="Data analyst with Python skills for calculations, statistics, and data processing",
            tools=data_tools,
            additional_authorized_imports=[
                "numpy", "pandas", "matplotlib", "seaborn", "scipy", 
                "json", "csv", "statistics", "math", "re"
            ],
            use_structured_outputs_internally=True,
            model=self.model,
            max_steps=self.config.max_agent_steps,
            logger=logger
        )
        
        # Web Researcher - gets context-aware content tools
        web_tools = env_tools.copy()
        
        # Add context-aware content retriever tool if available
        if self.context_aware_tools and isinstance(self.context_aware_tools, list):
            for tool in self.context_aware_tools:
                if hasattr(tool, 'name') and tool.name == "retrieve_content":
                    web_tools.append(tool)
                    print("✅ Web researcher gets context-aware ContentRetrieverTool")
                    break
                elif isinstance(tool, str) and "content_retriever" in tool:
                    print(f"⚠️  Context tool is string: {tool}")
        else:
            print(f"⚠️  Context-aware tools not available or invalid: {type(self.context_aware_tools)}")
        
        specialists["web_researcher"] = ToolCallingAgent(
            name="web_researcher",
            description="Web researcher for current information and content processing",
            tools=web_tools,
            model=self.model,
            max_steps=self.config.max_agent_steps,
            planning_interval=self.config.planning_interval,
            add_base_tools=False,
            logger=logger
        )
        
        print(f"✅ Created {len(specialists)} specialist agents")
        return specialists

    def _create_shared_tools(self):
        """Create shared tool instance"""
        shared_tools = {}
        
        if CUSTOM_TOOLS_AVAILABLE:
            try:
                shared_tools['get_attachment'] = GetAttachmentTool()
                print("✅ GetAttachmentTool created successfully")
            except Exception as e:
                print(f"❌ GetAttachmentTool failed: {e}")
                
            try:
                shared_tools['content_retriever'] = ContentRetrieverTool()
                print("✅ ContentRetrieverTool created successfully")
            except Exception as e:
                print(f"❌ ContentRetrieverTool failed: {e}")
                
            print(f"🔧 Final shared_tools keys: {list(shared_tools.keys())}")
        else:
            print("⚠️  Custom tools not available - shared_tools will be empty")
        
        return shared_tools
    
    def _create_manager_agent(self):
        """Create manager agent with context-aware file access tools"""
        manager_tools = []
        
        # Manager gets file access tools (context-aware versions)
        if self.context_aware_tools and isinstance(self.context_aware_tools, list):
            for tool in self.context_aware_tools:
                if hasattr(tool, 'name') and tool.name == "get_attachment":
                    manager_tools.append(tool)
                    print("✅ Manager gets context-aware GetAttachmentTool")
                    break
        else:
            print(f"⚠️  Context-aware tools not available for manager: {type(self.context_aware_tools)}")
        
        # Setup logging
        logger = None
        if self.logging and hasattr(self.logging, 'logger'):
            logger = self.logging.logger
        
        # Manager is ToolCallingAgent with file access specialization
        manager = ToolCallingAgent(
            name="gaia_manager",
            description="GAIA task coordinator that manages file access and coordinates specialist agents",
            model=self.model,
            tools=manager_tools,
            managed_agents=list(self.specialists.values()),
            planning_interval=self.config.planning_interval,
            max_steps=self.config.max_agent_steps,
            logger=logger
        )
        
        print("✅ Manager agent created with context-aware file access tools")
        return manager
    
    def _build_workflow(self):
        """Build LangGraph workflow with smart routing"""
        builder = StateGraph(GAIAState)
        
        if self.config.enable_smart_routing:
            # Smart routing workflow
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("complexity_check", self._complexity_check_node)
            builder.add_node("one_shot_answering", self._one_shot_answering_node)
            builder.add_node("manager_execution", self._manager_execution_node)
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
                    "complex": "manager_execution"
                }
            )
            
            # Both paths converge to formatting
            builder.add_edge("one_shot_answering", "format_answer")
            builder.add_edge("manager_execution", "format_answer")
            builder.add_edge("format_answer", END)
        else:
            # Original linear workflow
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("manager_execution", self._manager_execution_node)
            builder.add_node("format_answer", self._format_answer_node)
            
            builder.add_edge(START, "read_question")
            builder.add_edge("read_question", "manager_execution")
            builder.add_edge("manager_execution", "format_answer")
            builder.add_edge("format_answer", END)
        
        return builder.compile()
    
# ============================================================================
# FIXED MEMORY ACCESS HELPERS
# ============================================================================

    def get_agent_memory_safely(agent) -> Dict:
        """
        Safely access agent memory in various SmolagAgent versions.
        
        Args:
            agent: SmolagAgent instance
            
        Returns:
            Dictionary representation of agent memory
        """
        try:
            # NEW METHOD: Direct memory attribute access
            if hasattr(agent, 'memory') and agent.memory is not None:
                # Handle AgentMemory object
                if hasattr(agent.memory, '__dict__'):
                    return agent.memory.__dict__
                # Handle dict-like memory
                elif hasattr(agent.memory, 'steps'):
                    return {'steps': agent.memory.steps}
                # Handle memory as dict
                elif isinstance(agent.memory, dict):
                    return agent.memory
            
            # FALLBACK: Use deprecated logs attribute
            if hasattr(agent, 'logs'):
                print("⚠️  Using deprecated 'logs' attribute - consider updating SmolagAgents")
                return {'steps': agent.logs}
            
            # No memory found
            print("⚠️  No memory found in agent")
            return {'steps': []}
            
        except Exception as e:
            print(f"⚠️  Error accessing agent memory: {e}")
            return {'steps': []}

    def get_agent_step_count_safely(agent) -> int:
        """
        Safely get step count from agent memory.
        
        Args:
            agent: SmolagAgent instance
            
        Returns:
            Number of steps executed by the agent
        """
        try:
            memory = get_agent_memory_safely(agent)
            steps = memory.get('steps', [])
            return len(steps) if steps else 0
        except Exception as e:
            print(f"⚠️  Error getting step count: {e}")
            return 0
    
    # ============================================================================
    # WORKFLOW NODES WITH CONTEXT
    # ============================================================================
    
    def _read_question_node(self, state: GAIAState):
        """Read question and setup context bridge"""
        task_id = state.get("task_id")
        question = state["question"]
        
        # Setup context bridge
        if self.config.enable_context_bridge:
            metadata = {
                "complexity": state.get("complexity"),
                "routing_path": "initializing"
            }
            ContextVariableFlow.set_task_context(task_id, question, metadata)
            
            if self.config.context_bridge_debug:
                print(f"🌉 Context bridge activated: {ContextVariableFlow.get_context_summary()}")
        
        if self.logging:
            self.logging.log_step("question_setup", f"Processing question: {question[:50]}...")
        
        # Initialize similar_examples
        similar_examples = []
        
        # Only do RAG retrieval for complex questions if routing is enabled
        if not self.config.enable_smart_routing or not self.config.skip_rag_for_simple:
            # Always do RAG (original behavior)
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
        else:
            print("⚡ Skipping RAG for fast routing decision")
        
        return {
            "similar_examples": similar_examples,
            "steps": state["steps"] + ["Question setup and context bridge activated"]
        }

    def _complexity_check_node(self, state: GAIAState):
        """Determine complexity and update context bridge"""
        question = state["question"]
        task_id = state.get("task_id", "")
        
        if self.logging:
            self.logging.log_step("complexity_analysis", f"Analyzing complexity for: {question[:50]}...")
        
        print(f"🧠 Analyzing complexity for: {question[:50]}...")
        
        # Quick pattern matching first
        if is_simple_math(question):
            complexity = "simple"
            reason = "Simple arithmetic detected"
        elif is_simple_fact(question):
            complexity = "simple" 
            reason = "Simple factual query detected"
        elif has_attachments(task_id):
            complexity = "complex"
            reason = "File attachments detected"
        elif needs_web_search(question):
            complexity = "complex"
            reason = "Current information needed"
        else:
            # LLM decides for edge cases
            if self.logging:
                self.logging.log_step("llm_complexity_check", "Using LLM for edge case assessment")
            
            complexity = self._llm_complexity_check(question)
            reason = "LLM complexity assessment"
        
        print(f"📊 Complexity: {complexity} ({reason})")
        
        # Update context bridge with complexity
        if self.config.enable_context_bridge:
            ContextVariableFlow.update_complexity(complexity)
        
        if self.logging:
            self.logging.set_complexity(complexity)
            self.logging.log_step("complexity_result", f"Final complexity: {complexity} - {reason}")
        
        # If complex perform RAG search with retriever
        if complexity == "complex" and self.config.skip_rag_for_simple and not state.get("similar_examples"):
            print("📚 Retrieving RAG examples for complex question...")
            
            if self.logging:
                self.logging.log_step("rag_retrieval", "Retrieving examples for complex question")
            
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
                    self.logging.log_step("rag_complete", f"Retrieved {len(similar_examples)} examples")
                    
            except Exception as e:
                print(f"⚠️  RAG retrieval error: {e}")
                if self.logging:
                    self.logging.log_step("rag_error", f"RAG retrieval failed: {str(e)}")
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
        """Direct LLM answering for simple questions with context bridge"""
        print("⚡ Using one-shot direct answering")
        
        # Update context bridge routing
        if self.config.enable_context_bridge:
            ContextVariableFlow.update_routing_path("one_shot_llm")
        
        if self.logging:
            self.logging.set_routing_path("one_shot_llm")
            self.logging.log_step("one_shot_start", "Starting direct LLM answering")
        
        # Build prompt for direct answering
        question = state["question"]
        
        if self.logging:
            self.logging.log_step("one_shot_prompt_prep", f"Preparing direct prompt for: {question[:30]}...")
        
        # Simple prompt for direct answering
        prompt = f"""You are a general AI assistant. Answer this question directly and concisely.

        Question: {question}

        Provide your answer in the format: FINAL ANSWER: [YOUR ANSWER]

        Remember:
        - Numbers: no commas, no units unless specified
        - Strings: no articles (the, a, an), concise wording
        - Keep it as brief as possible"""
        
        try:
            if self.logging:
                self.logging.log_step("one_shot_llm_call", "Making direct LLM call")
            
            response = llm_invoke_with_retry(self.orchestration_model, [HumanMessage(content=prompt)])
            
            if self.logging:
                response_preview = response.content[:50] + "..." if len(response.content) > 50 else response.content
                self.logging.log_step("one_shot_response", f"LLM response received: {response_preview}")
            
            # Validate response
            if not response.content or len(response.content.strip()) == 0:
                error_msg = "Empty response from LLM"
                if self.logging:
                    self.logging.log_step("one_shot_empty_response", error_msg)
                raise ValueError(error_msg)
            
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

    def _manager_execution_node(self, state: GAIAState):
        """Manager execution with context bridge coordination"""
        print("🎯 Using manager coordination with context bridge")
        
        # Update context bridge routing
        if self.config.enable_context_bridge:
            ContextVariableFlow.update_routing_path("manager_coordination")
        
        if self.logging:
            self.logging.set_routing_path("manager_coordination")
            self.logging.log_step("manager_start", "Starting manager coordination")
        
        question = state["question"]
        task_id = state.get("task_id", "")
        rag_examples = state.get("similar_examples", [])
        
        # Ensure context is properly set before manager execution
        if self.config.enable_context_bridge and task_id:
            # Refresh context to ensure it's active during manager execution
            ContextVariableFlow.set_task_context(
                task_id=task_id,
                question=question,
                metadata={
                    "complexity": state.get("complexity", "complex"),
                    "routing_path": "manager_coordination",
                    "execution_phase": "manager_delegation"
                }
            )
            print(f"🌉 Context refreshed for manager execution: {task_id}")
        
        # Prepare context for manager with curated delegation format
        context = self._prepare_manager_context(question, rag_examples, task_id)
        
        try:
            if self.logging:
                self.logging.log_step("manager_execution", "Executing manager with fixed context")
            
            # Run manager agent with proper context
            result = self.manager.run(context)
            
            # Check if result contains FINAL ANSWER - if so, stop here
            if "FINAL ANSWER:" in str(result).upper():
                print("✅ FINAL ANSWER detected - terminating manager execution")
                return {
                    "raw_answer": str(result),
                    "steps": state["steps"] + ["Manager found final answer and terminated"]
    }
            
            # Adopt new memory access method
            step_count = GAIAAgent.get_agent_step_count_safely(self.manager)
            
            if self.logging:
                self.logging.log_step("manager_complete", f"Manager execution completed - {step_count} steps")
            
            return {
                "raw_answer": str(result),
                "steps": state["steps"] + [
                    f"Manager coordination with fixed delegation complete - {step_count} steps"
                ]
            }
            
        except Exception as e:
            error_msg = f"Manager execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Enhanced error debugging
            if self.config.context_bridge_debug:
                context_summary = ContextVariableFlow.get_context_summary()
                print(f"🔍 Context during error: {context_summary}")
            
            if self.logging:
                self.logging.log_step("manager_error", error_msg)
            
            return {
                "raw_answer": f"Manager execution error: {str(e)}",
                "steps": state["steps"] + [error_msg]
            }
            
        finally:
            # Context bridge cleanup happens automatically via context variables
            if self.config.context_bridge_debug:
                context_summary = ContextVariableFlow.get_context_summary()
                print(f"🌉 Context during manager execution: {context_summary}")

    def _get_current_temporal_context(self) -> str:
        """Get current date/time context for temporal awareness"""
        now = datetime.now(timezone.utc)
        return f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%B %d, %Y')})"
    
    def _prepare_manager_context(self, question: str, rag_examples: List[Dict], task_id: str) -> str:
        """Prepare context for manager agent with curated delegation format"""
        if self.logging:
            self.logging.log_step("context_prep_start", f"Preparing manager context for task {task_id}")
        
        context_parts = [
            "GAIA TASK COORDINATOR",
            "=" * 50,
            "",
            f"TASK CONTEXT: {task_id}",
            f"Question: {question}",
            "",
            "🌉 Context bridge is ACTIVE - tools automatically access task context",
            "",
            "AVAILABLE SPECIALIST AGENTS:",
            "- data_analyst: Python code execution, calculations, statistics, data processing",
            "- web_researcher: Web search, current information, content processing",
            "",
            "⚠️  CRITICAL: Use the EXACT format below for agent delegation:",
            "",
            "AGENT DELEGATION FORMAT:",
            "To call data_analyst:",
            "data_analyst(\"Calculate the average of [1, 2, 3, 4, 5]\")",
            "",
            "To call web_researcher:", 
            "web_researcher(\"Search for current information about Mark Rutte\")",
            "",
            "❌ DO NOT use complex JSON or dictionary arguments",
            "❌ DO NOT pass task_context, current_date, or content_retrieval as arguments",
            "✅ DO use simple string instructions only",
            "",
            "COORDINATION WORKFLOW:",
            "1. If files needed: Use get_attachment() tool first",
            "2. Extract/process data from files yourself", 
            "3. Delegate to specialists using SIMPLE STRING INSTRUCTIONS:",
            "   • For calculations: data_analyst(\"your calculation request\")",
            "   • For research: web_researcher(\"your research request\")",
            "4. Synthesize specialist results and provide final answer",
            "",
            "EXAMPLES OF CORRECT DELEGATION:",
            "",
            "For data questions:",
            f'data_analyst("Analyze the data and calculate statistics for the question: {question}")',
            "",
            "For research questions:",
            f'web_researcher("Research current information about: {question}")',
            "",
            "For web search questions:", 
            f'web_researcher("Find the latest news and information about: {question}")',
        ]
        
        # Add RAG examples if available
        if rag_examples:
            if self.logging:
                self.logging.log_step("context_add_rag", f"Adding {len(rag_examples)} RAG examples to context")
            
            context_parts.extend([
                "",
                "SIMILAR GAIA EXAMPLES:",
                ""
            ])
            for i, example in enumerate(rag_examples[:2], 1):
                context_parts.extend([
                    f"Example {i}:",
                    f"Q: {example['question']}",
                    f"A: {example['answer']}",
                    ""
                ])
        
        # Add file access instructions if needed
        if task_id and CUSTOM_TOOLS_AVAILABLE:
            if self.logging:
                self.logging.log_step("context_add_file", f"Adding file access instructions for task {task_id}")
            
            context_parts.extend([
                "FILE ACCESS (if needed):",
                f"- Task ID: {task_id}",
                "- Use get_attachment() to access files",
                "- Context bridge automatically provides task_id",
                ""
            ])
        
        # Add final requirements
        context_parts.extend([
            "TERMINATION RULES:",
            "- Once you provide FINAL ANSWER, STOP immediately",
            "- Do NOT make additional tool calls after FINAL ANSWER",
            "- Do NOT continue planning after providing the answer",
            "FINAL ANSWER REQUIREMENTS:",
            "- Use format: FINAL ANSWER: [YOUR ANSWER]",
            "- Numbers: no commas, no units unless specified",
            "- Strings: no articles (the, a, an), concise wording",
            "- Lists: comma separated, apply above rules to each element",
            "",
            "Remember: Context bridge handles task_id automatically!",
            f"Your task: {task_id}",
            ""
        ])
        
        final_context = "\n".join(context_parts)
        
        if self.logging:
            context_length = len(final_context)
            self.logging.log_step("context_prep_complete", f"Manager context prepared: {context_length} characters")
        
        if self.config.context_bridge_debug:
            print(f"🔧 Manager context prepared with simplified delegation format")
        
        return final_context
    
    def _format_answer_node(self, state: GAIAState):
        """Final answer formatting and validation with context bridge cleanup"""
        raw_answer = state.get("raw_answer", "")
        
        if self.logging:
            self.logging.log_step("format_start", f"Formatting answer: {raw_answer[:50]}...")
        
        try:
            # Extract final answer
            if self.logging:
                self.logging.log_step("extract_answer", "Extracting final answer from raw response")
            
            formatted_answer = self._extract_final_answer(raw_answer)
            
            if self.logging:
                self.logging.log_step("apply_gaia_format", f"Applying GAIA formatting to: {formatted_answer}")
            
            formatted_answer = self._apply_gaia_formatting(formatted_answer)
            
            if self.logging:
                self.logging.log_step("format_complete", f"Final formatted answer: {formatted_answer}")
            
            return {
                "final_answer": formatted_answer,
                "steps": state["steps"] + ["Final answer formatting applied"]
            }
            
        except Exception as e:
            error_msg = f"Answer formatting error: {str(e)}"
            
            if self.logging:
                self.logging.log_step("format_error", error_msg)
            
            return {
                "final_answer": raw_answer.strip() if raw_answer else "No answer",
                "steps": state["steps"] + [error_msg]
            }
        finally:
            # Context cleanup
            if self.config.enable_context_bridge:
                if self.config.context_bridge_debug:
                    final_context = ContextVariableFlow.get_context_summary()
                    print(f"🌉 Final context before cleanup: {final_context}")
                
                ContextVariableFlow.clear_context()
                
                if self.config.context_bridge_debug:
                    print("🧹 Context bridge cleaned up")
            
    def _extract_final_answer(self, raw_answer: str) -> str:
        """Extract final answer from manager response"""
        if not raw_answer:
            if self.logging:
                self.logging.log_step("extract_empty", "Raw answer is empty")
            return "No answer"
        
        if self.logging:
            self.logging.log_step("extract_patterns", "Searching for FINAL ANSWER patterns")
        
        # Try to find FINAL ANSWER pattern
        patterns = [
            r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
            r"Final Answer:\s*(.+?)(?:\n|$)",
            r"Answer:\s*(.+?)(?:\n|$)"
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, raw_answer, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if self.logging:
                    self.logging.log_step("extract_success", f"Pattern {i+1} matched: {extracted}")
                return extracted
        
        # Fallback to last line
        lines = raw_answer.strip().split('\n')
        fallback = lines[-1].strip() if lines else "No answer"
        
        if self.logging:
            self.logging.log_step("extract_fallback", f"No pattern matched, using last line: {fallback}")
        
        return fallback

    def _apply_gaia_formatting(self, answer: str) -> str:
        """Apply GAIA formatting rules"""
        if not answer:
            if self.logging:
                self.logging.log_step("gaia_format_empty", "Answer is empty, returning default")
            return "No answer"
        
        original_answer = answer
        answer = answer.strip()
        
        if self.logging:
            self.logging.log_step("gaia_format_start", f"Original: '{original_answer}' -> Stripped: '{answer}'")
        
        # Remove common prefixes
        prefixes = [
            "the answer is", "answer:", "final answer:", "result:", "solution:",
            "the result is", "therefore", "so", "thus"
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes:
            if answer_lower.startswith(prefix):
                old_answer = answer
                answer = answer[len(prefix):].strip()
                answer_lower = answer.lower()
                if self.logging:
                    self.logging.log_step("gaia_format_prefix", f"Removed prefix '{prefix}': '{old_answer}' -> '{answer}'")
                break
        
        # Remove quotes and punctuation
        old_answer = answer
        answer = answer.strip('.,!?:;"\'')
        if old_answer != answer and self.logging:
            self.logging.log_step("gaia_format_punct", f"Removed punctuation: '{old_answer}' -> '{answer}'")
        
        # Handle numbers - remove commas
        if answer.replace('.', '').replace('-', '').replace(',', '').isdigit():
            old_answer = answer
            answer = answer.replace(',', '')
            if old_answer != answer and self.logging:
                self.logging.log_step("gaia_format_number", f"Removed commas from number: '{old_answer}' -> '{answer}'")
        
        if self.logging:
            self.logging.log_step("gaia_format_final", f"Final GAIA formatted answer: '{answer}'")
        
        return answer
    
    # ============================================================================
    # CONVENIENCE FUNCTION WITH CONTEXT
    # ============================================================================
    
    @with_context
    def process_question(self, question: str, task_id: str = None) -> Dict:
        """Core production method to process a single question with context"""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        initial_state = {
            "task_id": task_id,
            "question": question,
            "steps": []
        }
        
        # Start logging if available
        if self.logging:
            self.logging.start_task(task_id, model_used=self.config.model_name)
            self.logging.log_step("question_received", f"Processing question with context bridge: {question}")
        
        try:
            if self.logging:
                self.logging.log_step("workflow_start", "Starting LangGraph workflow with context bridge")
            
            result = self.workflow.invoke(initial_state)
            
            if self.logging:
                self.logging.log_step("workflow_complete", "Workflow completed successfully")
            
            # Add metadata with context bridge info
            result.update({
                "task_id": task_id,
                "question": question,
                "execution_successful": True,
                "context_bridge_used": self.config.enable_context_bridge
            })
            
            # Log completion
            if self.logging:
                manual_steps = len(self.logging.manual_steps)
                self.logging.log_question_result(
                    task_id=task_id,
                    question=question,
                    final_answer=result.get("final_answer", ""),
                    total_steps=manual_steps,
                    success=True
                )
                self.logging.log_step("execution_complete", f"Question execution with context bridge completed - {manual_steps} logged steps")
            
            return result
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            
            if self.logging:
                self.logging.log_step("execution_error", error_msg)
                self.logging.log_question_result(
                    task_id=task_id,
                    question=question,
                    final_answer="ERROR",
                    total_steps=len(self.logging.manual_steps) if self.logging else 0,
                    success=False
                )
            
            return {
                "task_id": task_id,
                "question": question,
                "final_answer": "Execution failed",
                "error": error_msg,
                "execution_successful": False,
                "context_bridge_used": self.config.enable_context_bridge
            }
        finally:
            # Ensure context cleanup
            if self.config.enable_context_bridge:
                ContextVariableFlow.clear_context()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 GAIA Manager Agent with Context")
    print("=" * 50)
    print("✅ Context integration available")
    print("✅ File access capability")
    print("✅ Routing with context awareness")
    print("")
    print("Use agent_interface.py for testing")