# agent_logic.py
# GAIA Agent System using SmolagAgents Manager Pattern

import os
import uuid
import re
import backoff
from typing import TypedDict, Optional, List, Dict
from dataclasses import dataclass

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
    """GAIA agent using SmolagAgents manager pattern with complexity check"""
    
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
        
        # Create specialist agents
        self.specialists = self._create_specialist_agents()
        
        # Create manager agent
        self.manager = self._create_manager_agent()
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        print("🚀 GAIA Manager Agent initialized!")
    
    def _initialize_retriever(self):
        """Initialize retriever"""
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
        """Create specialist agents with clean logging"""
        
        logger = None
        if self.logging and hasattr(self.logging, 'logger'):
            logger = self.logging.logger
    
        specialists = {}
        env_tools = []

        # Add LangChain research tools
        if LANGCHAIN_TOOLS_AVAILABLE:
            env_tools.extend(ALL_LANGCHAIN_TOOLS)
            print(f"✅ Added {len(ALL_LANGCHAIN_TOOLS)} LangChain tools")
        
        # Add custom GAIA tools if available
        if CUSTOM_TOOLS_AVAILABLE:
            env_tools.extend([
                GetAttachmentTool(),
                ContentRetrieverTool()
            ])
        
        # Data Analyst - CodeAgent for calculations and data processing
        specialists["data_analyst"] = CodeAgent(
            name="data_analyst",
            description="Data analyst with Python skills for calculations, statistics, and data processing",
            tools=env_tools,
            additional_authorized_imports=[
                "numpy", "pandas", "matplotlib", "seaborn", "scipy", 
                "json", "csv", "statistics", "math", "re"
            ],
            model=self.model,
            max_steps=self.config.max_agent_steps,
            logger=logger
        )
        
        # Web Researcher - ToolCallingAgent for search and current information
        web_tools = []

        # PRIORITY 1: LangChain search tools FIRST
        if LANGCHAIN_TOOLS_AVAILABLE:
            web_tools.extend(ALL_LANGCHAIN_TOOLS)  # Serper gets added FIRST
            print(f"✅ PRIORITY: {len(ALL_LANGCHAIN_TOOLS)} LangChain tools added FIRST")
            
        if CUSTOM_TOOLS_AVAILABLE:
            web_tools.extend([GetAttachmentTool(), ContentRetrieverTool()])
            print("✅ GAIA tools added as secondary priority")
        
        specialists["web_researcher"] = ToolCallingAgent(
            name="web_researcher",
            description="Web researcher using Serper API, Wikipedia, and ArXiv for comprehensive information gathering",
            tools=web_tools,
            model=self.model,
            max_steps=self.config.max_agent_steps,
            planning_interval=self.config.planning_interval,
            add_base_tools=False,
            logger=logger
        )
        
        print(f"✅ Created {len(specialists)} specialist agents")
        return specialists
    
    def _create_manager_agent(self):
        """Create manager agent with specialist coordination using ToolCallingAgent"""
        manager_tools = []
        
        # Manager gets context tools
        if CUSTOM_TOOLS_AVAILABLE:
            manager_tools.extend([
                GetAttachmentTool(),
                ContentRetrieverTool()
            ])
        
        # Setup logging
        logger = None
        if self.logging and hasattr(self.logging, 'logger'):
            logger = self.logging.logger
        
        # Switch from CodeAgent to ToolCallingAgent
        manager = ToolCallingAgent(
            name="gaia_manager",
            description="GAIA task coordinator that manages specialist agents and ensures answer formatting",
            model=self.model,
            tools=manager_tools,
            managed_agents=list(self.specialists.values()),
            planning_interval=self.config.planning_interval,
            max_steps=self.config.max_agent_steps,
            logger=logger
        )
        
        print("✅ Manager agent created with ToolCallingAgent for better coordination")
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
    # WORKFLOW NODES
    # ============================================================================
    
    def _read_question_node(self, state: GAIAState):
        """Read question and conditionally get RAG context"""
        if self.logging:
            self.logging.log_step("question_setup", f"Processing question: {state['question'][:50]}...")
        
        # Initialize similar_examples
        similar_examples = []
        
        # Only do RAG retrieval for complex questions if routing is enabled
        if not self.config.enable_smart_routing or not self.config.skip_rag_for_simple:
            # Always do RAG (original behavior)
            try:
                similar_docs = self.retriever.search(state["question"], k=self.config.rag_examples_count)
                
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
            "steps": state["steps"] + ["Question setup complete"]
        }

    def _complexity_check_node(self, state: GAIAState):
        """Determine if question needs specialist coordination - MAIN LOGGING HERE"""
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
        """Use LLM to determine complexity for edge cases - SIMPLIFIED"""
        
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
            
            # Only log the final result in the parent method
            return complexity
            
        except Exception as e:
            # Default to complex if LLM fails
            print(f"⚠️  LLM complexity check failed: {str(e)}, defaulting to complex")
            return "complex"

    def _route_by_complexity(self, state: GAIAState) -> str:
        """Routing function for conditional edges - NO LOGGING NEEDED"""
        # This is just a simple getter - no logging needed
        return state.get("complexity", "complex")
    
    def _one_shot_answering_node(self, state: GAIAState):
        """Direct LLM answering for simple questions with comprehensive logging"""
        print("⚡ Using one-shot direct answering")
        
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
                "steps": state["steps"] + ["One-shot direct answering completed"]
            }
            
        except Exception as e:
            error_msg = f"One-shot answering failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            if self.logging:
                self.logging.log_step("one_shot_error", error_msg)
            
            # Return error state but continue workflow
            return {
                "raw_answer": f"Error in one-shot answering: {error_msg}",
                "steps": state["steps"] + [error_msg]
            }
        
    def _manager_execution_node(self, state: GAIAState):
        """Execute manager agent with manual step logging"""
        task_id = state.get("task_id", str(uuid.uuid4()))
        
        # Set task_id in environment for tools to access
        os.environ["CURRENT_GAIA_TASK_ID"] = task_id
        
        # Start logging for this task
        if self.logging:
            self.logging.start_task(task_id, self.logging.current_complexity, self.logging.current_model_used)
            self.logging.set_routing_path("manager_coordination")
            self.logging.log_step("manager_start", "Starting manager coordination")
        
        # Prepare manager context
        manager_context = self._prepare_manager_context(
            question=state["question"],
            rag_examples=state.get("similar_examples", []),
            task_id=task_id
        )
        
        if self.logging:
            self.logging.log_step("context_prepared", f"Manager context prepared with {len(state.get('similar_examples', []))} examples")
        
        try:
            print(f"🤖 Executing manager agent with {len(self.specialists)} specialists")
            
            if self.logging:
                self.logging.log_step("manager_execution", "Running manager agent")
            
            result = self.manager.run(manager_context)
            
            if self.logging:
                self.logging.log_step("manager_complete", "Manager execution successful")
            
            # Get step count
            step_count = self.logging.step_counter if self.logging else 0
            
            return {
                "raw_answer": str(result),
                "steps": state["steps"] + [
                    f"Manager coordination complete - {step_count} steps"
                ]
            }
            
        except Exception as e:
            error_msg = f"Manager execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            if self.logging:
                self.logging.log_step("manager_error", error_msg)
            
            return {
                "raw_answer": error_msg,
                "steps": state["steps"] + [error_msg]
            }
            
        finally:
            # Clean up environment - this ALWAYS runs
            if "CURRENT_GAIA_TASK_ID" in os.environ:
                del os.environ["CURRENT_GAIA_TASK_ID"]

    def _prepare_manager_context(self, question: str, rag_examples: List[Dict], task_id: str) -> str:
        """Prepare context for manager agent with logging"""
        if self.logging:
            self.logging.log_step("context_prep_start", f"Preparing manager context for task {task_id}")
        
        context_parts = [
            "You are coordinating a GAIA benchmark question. You have access to specialist agents:",
            "",
            "Available specialists:",
            "- data_analyst: Python code execution, calculations, statistics, data processing",
            "- web_researcher: Web search, current information, fact verification",  
            "- document_processor: File analysis, multimedia processing, transcription",
            "",
            f"Question: {question}",
            ""
        ]
        
        # Add RAG examples
        if rag_examples:
            if self.logging:
                self.logging.log_step("context_add_rag", f"Adding {len(rag_examples)} RAG examples to context")
            
            context_parts.extend([
                "Similar examples from GAIA database:",
                ""
            ])
            for i, example in enumerate(rag_examples[:2], 1):
                context_parts.extend([
                    f"Example {i}:",
                    f"Q: {example['question']}",
                    f"A: {example['answer']}",
                    ""
                ])
        else:
            if self.logging:
                self.logging.log_step("context_no_rag", "No RAG examples to add")
        
        # Add file context if available
        if task_id:
            if self.logging:
                self.logging.log_step("context_add_file", f"Adding file context for task {task_id}")
            
            context_parts.extend([
                f"Task ID: {task_id}",
                "Use GetAttachmentTool to access any uploaded files if needed.",
                ""
            ])
        
        # Add coordination instructions
        context_parts.extend([
            "Coordination instructions:",
            "1. Analyze the question to determine which specialist(s) to use",
            "2. For web searches, provide focused search queries to avoid broad searches",
            "3. Delegate to appropriate specialist(s) with clear instructions",
            "4. Review specialist responses for relevance and completeness",
            "5. When you have the answer, use final_answer('your concise answer') to finish."
            
            "IMPORTANT: Use the final_answer tool to provide your response, not free text.",
            "FINAL ANSWER format requirements:",
            "- Should be a number OR as few words as possible OR comma separated list",
            "- Numbers: no commas, no units ($ %) unless specified",
            "- Strings: no articles (the, a, an), no abbreviations, digits as text unless specified",
            "- Lists: apply above rules to each element",
            "",
            "Provide your final answer in format: FINAL ANSWER: [YOUR ANSWER]"
        ])
        
        final_context = "\n".join(context_parts)
        
        if self.logging:
            context_length = len(final_context)
            self.logging.log_step("context_prep_complete", f"Manager context prepared: {context_length} characters")
        
        return final_context
    
    def _format_answer_node(self, state: GAIAState):
        """Final answer formatting and validation with logging"""
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
            
    def _extract_final_answer(self, raw_answer: str) -> str:
        """Extract final answer from manager response with logging"""
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
        """Apply GAIA formatting rules with logging"""
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
    # CONVENIENCE FUNCTION
    # ============================================================================
    
    def process_question(self, question: str, task_id: str = None) -> Dict:
        """Core production method to process a single question"""
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
            self.logging.log_step("question_received", f"Processing question: {question}")
        
        try:
            if self.logging:
                self.logging.log_step("workflow_start", "Starting LangGraph workflow execution")
            
            result = self.workflow.invoke(initial_state)
            
            if self.logging:
                self.logging.log_step("workflow_complete", "Workflow completed successfully")
            
            # Add basic metadata
            result.update({
                "task_id": task_id,
                "question": question,
                "execution_successful": True
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
                self.logging.log_step("execution_complete", f"Question execution completed with {manual_steps} logged steps")
            
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
                "execution_successful": False
            }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 GAIA Manager Agent Module")
    print("=" * 30)
    print("Use gaia_interface.py for testing")