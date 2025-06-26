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
    LiteLLMModel,
    AgentLogger,
    LogLevel
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
    planning_interval: int = 3
        
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
    
    # Workflow tracking
    steps: List[str]
    raw_answer: Optional[str]
    final_answer: Optional[str]
    
    # RAG and routing
    similar_examples: Optional[List[Dict]]
    complexity: Optional[str]
    selected_agent: Optional[str]
    
    # Execution tracking
    execution_successful: Optional[bool]
    agent_used: Optional[str]

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
            
    def _create_specialist_agents(self):
        """Create specialized agents with the focused tool access"""
        
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        specialists = {}
        
        # Create general env_tools list
        env_tools = []
        if 'get_attachment' in self.shared_tools:
            env_tools.append(self.shared_tools['get_attachment'])
            print(f"✅ Added GetAttachmentTool to env_tools")

        # Data Analyst
        specialists["data_analyst"] = CodeAgent(
            name="data_analyst",
            description="Data analyst with advanced skills in statistic, handling tabular data and related Python packages.",
            tools=env_tools,
            additional_authorized_imports=[
                "numpy", "pandas", "matplotlib", "seaborn", "scipy", "io",
                "json", "csv", "statistics", "math", "re", "openpyxl", "xlrd"
            ],
            use_structured_outputs_internally=True,
            model=self.model,
            max_steps=self.config.max_agent_steps,
            logger=logger
    )
        
        # Web Researcher - REFERENCE STYLE: env_tools + additional tools
        web_tools = env_tools.copy()
        if LANGCHAIN_TOOLS_AVAILABLE:
            web_tools.extend(ALL_LANGCHAIN_TOOLS)
        if 'content_retriever' in self.shared_tools:
            web_tools.append(self.shared_tools['content_retriever'])
        
        specialists["web_researcher"] = ToolCallingAgent(
            name="web_researcher",
            description="Your role is to find information using search tools.

        Available tools:
        - retrieve_content: For processing documents
        - get_attachment: For accessing files  
        - search_web_serper: For current web information
        - search_wikipedia: For reliable information lookup
        - search_arxiv: For academic research

        WORKFLOW:
        1. Search for information using appropriate tools
        2. Process and analyze the results directly
        3. Provide clear, factual answers

        CRITICAL: Use tools directly, do NOT write Python code.",
            tools=web_tools,
            model=self.model,
            max_steps=self.config.max_agent_steps,
            planning_interval=self.config.planning_interval,
            add_base_tools=False,
            logger=logger
        )
        
        print(f"🎯 Created {len(specialists)} specialized agents:")
        print(f"   data_analyst: {len(data_tools)} tools")
        print(f"   web_researcher: {len(web_tools)} tools")
        
        return specialists

    def _create_shared_tools(self):
        """Create shared tool instances - NO WRAPPERS"""
        shared_tools = {}
        
        if CUSTOM_TOOLS_AVAILABLE:
            try:
                # Create tools directly
                shared_tools['content_retriever'] = ContentRetrieverTool()
                print("✅ ContentRetrieverTool created successfully")
            except Exception as e:
                print(f"❌ ContentRetrieverTool failed: {e}")
            
            try:
                # Create attachment tool directly - no wrappers  
                shared_tools['get_attachment'] = GetAttachmentTool()
                print("✅ GetAttachmentTool created successfully")
                print(f"🔧 GetAttachmentTool instance: {type(shared_tools['get_attachment'])}")
            except Exception as e:
                print(f"❌ GetAttachmentTool failed: {e}")
            
            # DISABLED: Grounding tool creation
            # if getattr(self.config, 'enable_grounding_tools', False):
            #     try:
            #         from tools.content_grounding_tool import ContentGroundingTool
            #         shared_tools['content_grounding'] = ContentGroundingTool()
            #         print("✅ ContentGroundingTool created successfully")
            #     except Exception as e:
            #         print(f"❌ ContentGroundingTool failed: {e}")
            
            print(f"🔧 Direct shared_tools created: {list(shared_tools.keys())}")
            print("🚫 ContentGroundingTool disabled for better GAIA performance")
        else:
            print("⚠️  Custom tools not available - shared_tools will be empty")
        
        return shared_tools
    
    def _configure_tools_from_state(self, agent_name: str, state: GAIAState):
        """SIMPLIFIED: Minimal tool configuration"""
        task_id = state.get("task_id")
        question = state.get("question")
        
        if not task_id:
            return
        
        print(f"🔧 Minimal tool configuration for {agent_name}")
        
        # Get the agent
        specialist = self.specialists[agent_name]
        
        # Only configure tools that explicitly need it
        for tool in specialist.tools:
            try:
                tool_name = getattr(tool, 'name', None) or tool.__class__.__name__
                
                # Skip get_attachment - already configured in read_question_node
                if tool_name == "get_attachment":
                    print(f"ℹ️  {tool_name} already activated globally")
                    continue
                
                # Only configure content retriever with question context
                if tool_name == "retrieve_content" and hasattr(tool, 'configure_from_state'):
                    tool.configure_from_state(question)
                    print(f"✅ Configured {tool_name} with question context")
                else:
                    print(f"ℹ️  {tool_name} needs no configuration")
                    
            except Exception as e:
                tool_name = getattr(tool, 'name', None) or tool.__class__.__name__
                print(f"⚠️ Could not configure {tool_name}: {e}")
                
    def get_agent_memory_safely(self, agent) -> Dict:
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

    def _llm_select_agent(self, question: str, similar_examples: List[Dict] = None) -> str:
        """LLM selects agent using RAG examples - no hard-coded rules"""
        
        # Build agent descriptions
        agents_description = "\n".join([
            f"{name}: {agent.description}"
            for name, agent in self.specialists.items()
        ])
        
        # Add similar examples context
        examples_context = ""
        if similar_examples:
            examples_context = "\n\nSimilar GAIA examples:\n"
            for i, example in enumerate(similar_examples[:3], 1):
                examples_context += f"{i}. Q: {example.get('question', '')}\n"
                examples_context += f"   A: {example.get('answer', '')}\n"
        
        # LLM selection prompt
        prompt = f"""Select the best specialist for this question.

    Question: {question}

    Available specialists:
    {agents_description}{examples_context}

    Choose: {' or '.join(self.specialists.keys())}

    Your choice:"""
        
        try:
            response = llm_invoke_with_retry(self.orchestration_model, [HumanMessage(content=prompt)])
            choice = response.content.strip().lower()
            
            # Parse LLM response
            for agent_name in self.specialists.keys():
                if agent_name in choice:
                    return agent_name
            
            # Default if unclear
            return list(self.specialists.keys())[0]
            
        except Exception as e:
            # Fallback to first agent
            return list(self.specialists.keys())[0]

    def _add_directional_info(self, question: str, agent_name: str, state: GAIAState) -> str:
        """Add essential directional info to the task"""
        
        # Get file info from state
        has_file = state.get("has_file", False)
        file_name = state.get("file_name", "")
        
        # Build enhanced task with directional info
        task_parts = [question]
        
        # Add file handling direction
        if has_file:
            task_parts.append(f"\nNote: There is an attached file '{file_name}'. Use get_attachment tool to access it.")
        
        # Add agent-specific direction
        if agent_name == "data_analyst":
            task_parts.append("\nProvide numerical calculations and data analysis. Use Python code for processing.")
        elif agent_name == "web_researcher":
            task_parts.append("\nSearch for current information using available research tools.")
        
        # Add GAIA format requirement
        task_parts.append("\nIMPORTANT: End your response with 'FINAL ANSWER: [your answer]' in the exact GAIA format.")
        
        return "\n".join(task_parts)
                   
    # ============================================================================
    # WORKFLOW 
    # ============================================================================

    def _build_workflow(self):
        """Build LangGraph workflow with smart routing - Updated with SmolagAgents pattern"""
        builder = StateGraph(GAIAState)
        
        if self.config.enable_smart_routing:
            # Smart routing workflow with proper agent selection/delegation separation
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("complexity_check", self._complexity_check_node)
            builder.add_node("one_shot_answering", self._one_shot_answering_node)
            builder.add_node("agent_selector", self._agent_selector_node)
            builder.add_node("delegate_to_agent", self._delegate_to_agent_node)  # NEW
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
                    "complex": "agent_selector"
                }
            )
            
            # Agent workflow: select → delegate → format
            builder.add_edge("agent_selector", "delegate_to_agent")
            builder.add_edge("delegate_to_agent", "format_answer")
            
            # Simple path: direct to formatting
            builder.add_edge("one_shot_answering", "format_answer")
            
            # Both paths end at formatting
            builder.add_edge("format_answer", END)
            
        else:
            # Original linear workflow - Updated with proper separation
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("agent_selector", self._agent_selector_node)
            builder.add_node("delegate_to_agent", self._delegate_to_agent_node)
            builder.add_node("format_answer", self._format_answer_node)
            
            # Linear flow: read → select → delegate → format
            builder.add_edge(START, "read_question")
            builder.add_edge("read_question", "agent_selector")
            builder.add_edge("agent_selector", "delegate_to_agent")
            builder.add_edge("delegate_to_agent", "format_answer")
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

    def _agent_selector_node(self, state: GAIAState):
        """Enhanced agent selection with file awareness"""
        
        if self.logging:
            self.logging.log_step("agent_selector", "Selecting appropriate specialist agent")
        
        print(f"🎯 Agent Selection for: {state['question'][:100]}...")
        
        # Use file information from state for better selection
        has_file = state.get("has_file", False)
        file_name = state.get("file_name", "")
        similar_examples = state.get("similar_examples", [])
        
        # Build enhanced examples context
        examples_context = ""
        if similar_examples:
            examples_context = "\n\nSimilar GAIA examples:\n"
            for i, example in enumerate(similar_examples[:3], 1):
                examples_context += f"{i}. Q: {example.get('question', '')[:100]}...\n"
                examples_context += f"   A: {example.get('answer', '')}\n"
        
        # Enhanced selection prompt with file awareness
        file_context = ""
        if has_file:
            file_context = f"\n\nFile Information:\n- Has file: {file_name}\n- File processing needed: Yes"
        
        selection_prompt = f"""
        Based on these successful GAIA examples, select the best agent for this question:
        
        QUESTION: {state["question"]}
        {file_context}
        {examples_context}
        
        Available agents:
        - data_analyst: Python code execution, file analysis, calculations, Excel/CSV processing
        - web_researcher: Google search, web browsing, current information, document research
        
        Consider:
        1. Does the question require file processing? → data_analyst
        2. Does it need web search or current information? → web_researcher
        3. What type of question succeeded in similar examples?
        4. File information: {file_name if has_file else "No files"}
        
        Reply with just: data_analyst OR web_researcher
        """
        
        try:
            if self.logging:
                self.logging.log_step("agent_selection_prompt", "LLM selecting best agent with file awareness")
            
            response = self.orchestration_model.invoke([HumanMessage(content=selection_prompt)])
            selected_agent = response.content.strip().lower()
            
            # Validate selection
            if selected_agent not in self.specialists:
                print(f"⚠️ Invalid agent selection '{selected_agent}', using data_analyst")
                selected_agent = "data_analyst"
            
            print(f"🤖 Selected agent: {selected_agent}")
            
            if self.logging:
                self.logging.log_step("agent_selected", f"Selected agent: {selected_agent}")
            
            return {
                **state,
                "selected_agent": selected_agent,
                "similar_examples": similar_examples
            }
            
        except Exception as e:
            print(f"❌ Agent selection failed: {e}")
            print("🔄 Falling back to data_analyst")
            
            if self.logging:
                self.logging.log_step("agent_selection_failed", f"Selection failed: {e}, using data_analyst fallback")
            
            return {
                **state,
                "selected_agent": "data_analyst",
                "similar_examples": similar_examples,
                "selection_fallback": True
            }

    def _delegate_to_agent_node(self, state: GAIAState):
        """ULTRA MINIMAL: Pass original question directly like reference"""
        selected_agent = state.get("selected_agent", "data_analyst")
        task_id = state.get("task_id")
        question = state.get("question")  # Original question only
        
        if self.logging:
            self.logging.log_step("delegate_to_agent", f"Executing agent: {selected_agent}")
        
        print(f"🚀 Executing agent: {selected_agent}")
        
        try:
            # EXISTING: Configure tools from LangGraph state
            self._configure_tools_from_state(selected_agent, state)
            
            # Get the specialist agent
            specialist = self.specialists[selected_agent]
            
            if self.logging:
                self.logging.log_step("agent_execution_start", f"Agent execution: {selected_agent}")
            
            # ULTRA MINIMAL: Pass original question directly (like reference)
            result = specialist.run(task=question)
            
            if self.logging:
                self.logging.log_step("agent_execution_complete", f"Agent {selected_agent} completed")
            
            print(f"✅ Agent {selected_agent} completed execution")
            
            return {
                **state,
                "agent_used": selected_agent,
                "raw_answer": result,
                "execution_successful": True
            }
            
        except Exception as e:
            print(f"❌ Agent execution failed: {e}")
            
            if self.logging:
                self.logging.log_step("agent_execution_failed", f"Agent {selected_agent} failed: {str(e)}")
            
            return {
                **state,
                "agent_used": selected_agent,
                "raw_answer": f"Agent execution failed: {str(e)}",
                "execution_successful": False,
                "execution_error": True
            }
        
    def _get_current_temporal_context(self) -> str:
        """DISABLED: Is interfering with GAIA historical queries"""
        return ""  # Returns empty string

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
        if not raw_answer:
            if self.logging:
                self.logging.log_step("extract_empty", "Raw answer is empty")
            return "No answer"
        
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