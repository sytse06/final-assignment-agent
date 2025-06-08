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

# SmolagAgents imports
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    GoogleSearchTool,
    VisitWebpageTool,
)

# Import retriever system
from dev_retriever import load_gaia_retriever

# Import logging
from agent_logging import AgentLoggingSetup

# Import custom tools
try:
    from tools import GetAttachmentTool, ContentRetrieverTool
    CUSTOM_TOOLS_AVAILABLE = True
except ImportError:
    print("⚠️  Custom tools not available - using base tools only")
    CUSTOM_TOOLS_AVAILABLE = False

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
    """Retry logic for LLM calls"""
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
        """Initialize LiteLLMModel"""
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
                raise ValueError(f"Unknown provider: {self.config.model_provider}")
            
            return LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                temperature=self.config.temperature
            )
            
        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            raise
    
    def _create_specialist_agents(self):
        """Create specialist agents"""
        env_tools = []
        
        # Add custom GAIA tools if available
        if CUSTOM_TOOLS_AVAILABLE:
            env_tools.extend([
                GetAttachmentTool(),
                ContentRetrieverTool()
            ])
        
        # Setup logging callbacks
        step_callbacks = []
        logger = None
        if self.logging:
            step_callbacks = [self.logging.capture_step_log]
            logger = self.logging.logger
        
        specialists = {}
        
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
            logger=logger,
            step_callbacks=step_callbacks
        )
        
        # Web Researcher - ToolCallingAgent for search and current information
        web_tools = env_tools.copy()
        try:
            web_tools.append(GoogleSearchTool())
            print("✅ GoogleSearchTool added to web_researcher")
        except Exception:
            print("⚠️  GoogleSearchTool not available")
        
        try:
            web_tools.append(VisitWebpageTool())
            print("✅ VisitWebpageTool added to web_researcher")
        except Exception:
            print("⚠️  VisitWebpageTool not available")
        
        specialists["web_researcher"] = ToolCallingAgent(
            name="web_researcher",
            description="Web researcher for finding current information and verifying facts",
            tools=web_tools,
            model=self.model,
            max_steps=self.config.max_agent_steps,
            planning_interval=self.config.planning_interval,
            logger=logger,
            step_callbacks=step_callbacks
        )
        
        # Document Processor - ToolCallingAgent for file and multimedia processing
        specialists["document_processor"] = ToolCallingAgent(
            name="document_processor",
            description="Document processor for analyzing files, images, and multimedia content",
            tools=env_tools,
            add_base_tools=True,  # Includes transcriber for audio
            model=self.model,
            max_steps=self.config.max_agent_steps,
            logger=logger,
            step_callbacks=step_callbacks
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
        step_callbacks = []
        logger = None
        if self.logging:
            step_callbacks = [self.logging.capture_step_log]
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
            logger=logger,
            step_callbacks=step_callbacks
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
            self.logging.logger.log_task(
                content=state["question"].strip(),
                title="GAIA Question Processing",
                subtitle="Question Analysis and Smart Routing"
            )
        
        # Initialize similar_examples
        similar_examples = []
        
        # Only do RAG retrieval for complex questions if smart routing is enabled
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
        """Determine if question needs specialist coordination"""
        question = state["question"]
        task_id = state.get("task_id", "")
        
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
            complexity = self._llm_complexity_check(question)
            reason = "LLM complexity assessment"
        
        print(f"📊 Complexity: {complexity} ({reason})")
        
        # If complex and we skipped RAG earlier, do it now
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
            response = llm_invoke_with_retry(self.model, [HumanMessage(content=prompt)])
            result = response.content.strip().lower()
            return "simple" if "simple" in result else "complex"
        except Exception:
            # Default to complex if LLM fails
            return "complex"
    
    def _one_shot_answering_node(self, state: GAIAState):
        """Direct LLM answering for simple questions"""
        print("⚡ Using one-shot direct answering")
        
        # Simple prompt for direct answering
        prompt = f"""You are a general AI assistant. Answer this question directly and concisely.

        Question: {state["question"]}

        Provide your answer in the format: FINAL ANSWER: [YOUR ANSWER]

        Remember:
        - Numbers: no commas, no units unless specified
        - Strings: no articles (the, a, an), concise wording
        - Keep it as brief as possible"""
        
        try:
            response = llm_invoke_with_retry(self.model, [HumanMessage(content=prompt)])
            
            return {
                "raw_answer": response.content,
                "steps": state["steps"] + ["One-shot direct answering completed"]
            }
        except Exception as e:
            error_msg = f"One-shot answering failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                "raw_answer": f"Error: {error_msg}",
                "steps": state["steps"] + [error_msg]
            }
    
    def _route_by_complexity(self, state: GAIAState) -> str:
        """Routing function for conditional edges"""
        return state.get("complexity", "complex")
    
    def _manager_execution_node(self, state: GAIAState):
        """Execute manager agent with specialist coordination"""
        task_id = state.get("task_id", str(uuid.uuid4()))
        
        # Start logging for this task
        if self.logging:
            self.logging.start_task(task_id)
        
        # Prepare manager context
        manager_context = self._prepare_manager_context(
            question=state["question"],
            rag_examples=state.get("similar_examples", []),
            task_id=task_id
        )
        
        try:
            print(f"🤖 Executing manager agent with {len(self.specialists)} specialists")
            result = self.manager.run(manager_context)
            
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
            
            return {
                "raw_answer": error_msg,
                "steps": state["steps"] + [error_msg]
            }
    
    def _prepare_manager_context(self, question: str, rag_examples: List[Dict], task_id: str) -> str:
        """Prepare context for manager agent"""
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
        
        # Add RAG examples if available
        if rag_examples:
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
        
        # Add file context if available
        if task_id:
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
            "5. Format final answer according to GAIA requirements",
            "",
            "FINAL ANSWER format requirements:",
            "- Should be a number OR as few words as possible OR comma separated list",
            "- Numbers: no commas, no units ($ %) unless specified",
            "- Strings: no articles (the, a, an), no abbreviations, digits as text unless specified",
            "- Lists: apply above rules to each element",
            "",
            "Provide your final answer in format: FINAL ANSWER: [YOUR ANSWER]"
        ])
        
        return "\n".join(context_parts)
    
    def _format_answer_node(self, state: GAIAState):
        """Final answer formatting and validation"""
        raw_answer = state.get("raw_answer", "")
        
        try:
            formatted_answer = self._extract_final_answer(raw_answer)
            formatted_answer = self._apply_gaia_formatting(formatted_answer)
            
            return {
                "final_answer": formatted_answer,
                "steps": state["steps"] + ["Final answer formatting applied"]
            }
            
        except Exception as e:
            error_msg = f"Answer formatting error: {str(e)}"
            return {
                "final_answer": raw_answer.strip() if raw_answer else "No answer",
                "steps": state["steps"] + [error_msg]
            }
    
    def _extract_final_answer(self, raw_answer: str) -> str:
        """Extract final answer from manager response"""
        if not raw_answer:
            return "No answer"
        
        # Try to find FINAL ANSWER pattern
        patterns = [
            r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
            r"Final Answer:\s*(.+?)(?:\n|$)",
            r"Answer:\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_answer, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback to last line
        lines = raw_answer.strip().split('\n')
        return lines[-1].strip() if lines else "No answer"
    
    def _apply_gaia_formatting(self, answer: str) -> str:
        """Apply GAIA formatting rules"""
        if not answer:
            return "No answer"
        
        answer = answer.strip()
        
        # Remove common prefixes
        prefixes = [
            "the answer is", "answer:", "final answer:", "result:", "solution:",
            "the result is", "therefore", "so", "thus"
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                answer_lower = answer.lower()
        
        # Remove quotes and punctuation
        answer = answer.strip('.,!?:;"\'')
        
        # Handle numbers - remove commas
        if answer.replace('.', '').replace('-', '').replace(',', '').isdigit():
            answer = answer.replace(',', '')
        
        return answer
    
    # ============================================================================
    # PUBLIC INTERFACE
    # ============================================================================
    
    def run_single_question(self, question: str, task_id: str = None) -> Dict:
        """Execute single question using manager agent"""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        initial_state = {
            "task_id": task_id,
            "question": question,
            "steps": []
        }
        
        print(f"🔍 Processing: {question[:60]}...")
        
        try:
            result = self.workflow.invoke(initial_state)
            
            # Log to CSV if enabled
            if self.logging:
                self.logging.log_question_result(
                    task_id=task_id,
                    question=question,
                    final_answer=result.get("final_answer", ""),
                    total_steps=len(result.get("steps", [])),
                    success=True
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            if self.logging:
                self.logging.log_question_result(
                    task_id=task_id,
                    question=question,
                    final_answer="ERROR",
                    total_steps=0,
                    success=False
                )
            
            return {
                "task_id": task_id,
                "question": question,
                "final_answer": "Execution failed",
                "error": error_msg
            }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 GAIA Manager Agent Module")
    print("=" * 30)
    print("Use gaia_interface.py for testing capabilities")