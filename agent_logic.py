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
    """GAIA agent using direct specialist execution with context sharing"""
    
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
        
        # Create context-aware tools
        if config.enable_context_bridge and self.shared_tools:
            self.context_aware_tools = create_context_aware_tools(self.shared_tools)
            print(f"🌉 Context-aware tools created: {len(self.context_aware_tools)} tools")
        else:
            self.context_aware_tools = []
        
        # Create specialist agents
        self.specialists = self._create_specialist_agents()
        
        # Build workflow (no manager needed)
        self.workflow = self._build_workflow()
        
        print("🚀 GAIA Agent initialized with direct specialist execution!")
    
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
        """Create specialized agents with focused tool distribution"""
        
        logger = self.logging.logger if self.logging and hasattr(self.logging, 'logger') else None
        specialists = {}

        # Helper function to get context-aware tools
        def get_context_tool(tool_name):
            if self.context_aware_tools and isinstance(self.context_aware_tools, list):
                for tool in self.context_aware_tools:
                    if hasattr(tool, 'name') and tool.name == tool_name:
                        return tool
            return None

        # Data Analyst
        data_tools = []
        
        # Only file access - no web search distractions
        if attachment_tool := get_context_tool("get_attachment"):
            data_tools.append(attachment_tool)
        if content_tool := get_context_tool("content_retriever"):
            data_tools.append(content_tool)
        
        specialists["data_analyst"] = CodeAgent(
            name="data_analyst",
            description="Specialized data analyst for Excel/CSV analysis, calculations, and file processing. Has both file access and content processing tools.",
            tools=data_tools,
            additional_authorized_imports=[
                "numpy", "pandas", "matplotlib", "seaborn", "scipy", "io",
                "json", "csv", "statistics", "math", "re", "openpyxl", "xlrd"
            ],
            use_structured_outputs_internally=True,
            model=self.model,
            max_steps=self.config.max_agent_steps,
            logger=logger
        )
        
        # Web Researcher - SPECIALIZED for information gathering and analysis
        web_tools = []
        
        # Add LangChain research tools - core specialty
        if LANGCHAIN_TOOLS_AVAILABLE:
            web_tools.extend(ALL_LANGCHAIN_TOOLS)
        
        # Add content processing tools
        if content_tool := get_context_tool("content_retriever"):
            web_tools.append(content_tool)
        
        # Add file access for document research
        if attachment_tool := get_context_tool("get_attachment"):
            web_tools.append(attachment_tool)

        specialists["web_researcher"] = ToolCallingAgent(
            name="web_researcher",
            description="Specialized web researcher for information gathering, content processing, and document analysis. Focuses on finding and processing information, delegates computational analysis to data_analyst.",
            tools=web_tools,  # Web search + content processing + file access
            model=self.model,
            max_steps=self.config.max_agent_steps,
            planning_interval=self.config.planning_interval,
            add_base_tools=False,
            logger=logger
        )
        
        print(f"🎯 Created {len(specialists)} specialized agents:")
        print(f"   data_analyst: {len(data_tools)} tools (computation focus)")
        print(f"   web_researcher: {len(web_tools)} tools (research focus)")
        
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
    
    def _build_workflow(self):
        """Build LangGraph workflow with smart routing"""
        builder = StateGraph(GAIAState)
        
        if self.config.enable_smart_routing:
            # Smart routing workflow
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("complexity_check", self._complexity_check_node)
            builder.add_node("one_shot_answering", self._one_shot_answering_node)
            builder.add_node("agent_selector", self._agent_selector_node)
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
            
            # Both paths converge to formatting
            builder.add_edge("one_shot_answering", "format_answer")
            builder.add_edge("agent_selector", "format_answer")
            builder.add_edge("format_answer", END)
        else:
            # Original linear workflow
            builder.add_node("read_question", self._read_question_node)
            builder.add_node("agent_selector", self._agent_selector_node)
            builder.add_node("format_answer", self._format_answer_node)
            
            builder.add_edge(START, "read_question")
            builder.add_edge("read_question", "agent_selector")
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
        prompt = f"""You are a general AI assistant. You must provide specific, factual answers.
        
        Question: {question}

        CRITICAL: Never use placeholder text like "[your answer]" or "[Title of...]". Always give the actual answer.

        When analyzing files:
        1. Use get_attachment tool to access file content
        2. Process the actual data thoroughly  
        3. Provide specific numerical or text answers

        Report your thoughts, and finish with: FINAL ANSWER: [YOUR SPECIFIC ANSWER]

        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list.
        - Numbers: no commas, no units ($ %) unless specified
        - Strings: no articles (the, a, an), no abbreviations, digits as text unless specified  
        - Lists: apply above rules to each element

        NEVER use placeholder text. Always give real, specific answers."""
        
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

    def _agent_selector_node(self, state: GAIAState):
        """Select and execute specialist agent using LLM + RAG examples"""
        
        question = state["question"]
        task_id = state.get("task_id", "")
        similar_examples = state.get("similar_examples", [])
        
        # Update context and logging
        if self.config.enable_context_bridge:
            ContextVariableFlow.update_routing_path("agent_selector")
        
        if self.logging:
            self.logging.set_routing_path("agent_selector")
            self.logging.log_step("agent_selector_start", "Starting agent selection")
        
        # Set context for file access
        if self.config.enable_context_bridge and task_id:
            ContextVariableFlow.set_task_context(
                task_id=task_id,
                question=question,
                metadata={
                    "complexity": state.get("complexity", "complex"),
                    "routing_path": "agent_selector"
                }
            )
        
        try:
            # LLM selects agent using RAG examples
            selected_agent = self._llm_select_agent(question, similar_examples)
            
            if self.logging:
                self.logging.log_step("agent_selected", f"Selected: {selected_agent}")
            
            # Execute selected agent directly
            specialist = self.specialists[selected_agent]
            result = specialist.run(question)
            
            if self.logging:
                self.logging.log_step("agent_complete", "Execution completed")
            
            step_count = 0  # Simplified - remove problematic step counting
            
            return {
                "raw_answer": str(result),
                "selected_agent": selected_agent,
                "execution_successful": True,
                "steps": state["steps"] + [f"Agent {selected_agent} executed successfully"]
            }
            
        except Exception as e:
            error_msg = f"Agent selection failed: {str(e)}"
            
            if self.logging:
                self.logging.log_step("agent_error", error_msg)
            
            return {
                "raw_answer": error_msg,
                "execution_successful": False,
                "steps": state["steps"] + [error_msg]
            }

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

    def _get_current_temporal_context(self) -> str:
        """Get current date/time context for temporal awareness"""
        now = datetime.now(timezone.utc)
        return f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%B %d, %Y')})"
    
    def _prepare_manager_context(self, question: str, rag_examples: List[Dict], task_id: str) -> str:
        """Ultra-simplified manager context to avoid JSON issues"""
        
        context_parts = [
            "GAIA TASK COORDINATOR",
            "=" * 50,
            "",
            f"Question: {question}",
            f"Task ID: {task_id}",
            " Context bridge is active - tools automatically access task context",
            "",
            " YOUR MISSION:",
            "Solve this question step by step and provide a final answer.",
            "",
            "AVAILABLE TOOLS and SPECIALISTS:",
            "- get_attachment to access files",
            "- data_analyst for calculations and data processing",
            "- web_researcher for current information and web search",
            "",
            "WORKFLOW:",
            "1. If files mentioned use get_attachment first",
            "2. Read and understand file content",
            "3. Analyze the problem step by step",
            "4. Get help from specialists if needed",
            "5. Provide final answer",
            "",
            "GETTING HELP:",
            "- For calculations ask data_analyst",
            "- For research ask web_researcher",
            "- Use simple natural language",
            "",
            "IMPORTANT NOTES:",
            "- Files are automatically linked to this task",
            "- Focus on solving the problem correctly",
            "- Work through the solution step by step",
        ]
        
        # Add RAG examples simply
        if rag_examples:
            context_parts.extend([
                "",
                " SIMILAR EXAMPLES:",
                f"Previous answer for similar question: {rag_examples[0].get('answer', 'Unknown')}",
                ""
            ])
        
        # Clean completion requirements
        context_parts.extend([
            "",
            "FINAL ANSWER REQUIREMENTS:",
            "",
            "Format: FINAL ANSWER: [your answer]",
            "",
            "Answer guidelines:",
            "- Numbers: Use integers when specified",
            "- Text: Be concise",
            "- No unnecessary words",
            "",
            "Examples:",
            "- FINAL ANSWER: 3",
            "- FINAL ANSWER: Time-Parking 2: Parallel Universe", 
            "- FINAL ANSWER: 17.056",
            "",
            "SUCCESS FACTORS:",
            "1. You MUST provide a FINAL ANSWER",
            "2. Stop after giving your answer",
            "3. Answer the question directly",
            "",
            f"Start solving task {task_id} now"
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