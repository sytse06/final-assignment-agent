# agent_logic.py
# Core GAIA Agent System using SmolagAgents + LangGraph

import os
import uuid
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
    tool,
    AgentLogger,
    LogLevel,
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
    """Simple configuration for GAIA agent"""
    model_provider: str = "groq"
    model_name: str = "qwen-qwq-32b"
    temperature: float = 0.3
    
    # Retriever settings
    csv_file: str = "gaia_embeddings.csv"
    rag_examples_count: int = 3
    
    # Agent settings
    max_agent_steps: int = 15
    
    # Logging
    enable_csv_logging: bool = True
    step_log_file: str = "gaia_steps.csv"
    question_log_file: str = "gaia_questions.csv"
    debug_mode: bool = True

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class GAIAState(TypedDict):
    """Simple state for GAIA workflow"""
    task_id: Optional[str]
    question: str
    steps: List[str]
    raw_answer: Optional[str]
    final_answer: Optional[str]
    similar_examples: Optional[List[Dict]]

# ============================================================================
# LLM RETRY LOGIC
# ============================================================================

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, max_tries=3)
def llm_invoke_with_retry(llm, messages):
    """Simple retry logic for LLM calls"""
    return llm.invoke(messages)

# ============================================================================
# MAIN GAIA AGENT
# ============================================================================

class GAIAAgent:
    """Simple GAIA agent using SmolagAgents + LangGraph"""
    
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
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        print("🚀 Simple GAIA Agent initialized!")
    
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
    
    def _setup_tools(self):
        """Setup available tools"""
        tools = []
        
        # Add custom GAIA tools if available
        if CUSTOM_TOOLS_AVAILABLE:
            tools.extend([
                GetAttachmentTool(),
                ContentRetrieverTool()
            ])
            print("✅ Custom GAIA tools loaded")
        
        # Add web search tools
        try:
            tools.append(GoogleSearchTool())
            print("✅ GoogleSearchTool added")
        except Exception:
            print("⚠️  GoogleSearchTool not available")
        
        try:
            tools.append(VisitWebpageTool())
            print("✅ VisitWebpageTool added")
        except Exception:
            print("⚠️  VisitWebpageTool not available")
        
        return tools
    
    def _build_workflow(self):
        """Build simple LangGraph workflow"""
        builder = StateGraph(GAIAState)
        
        # Add nodes
        builder.add_node("read_question", self._read_question_node)
        builder.add_node("agent_execution", self._agent_execution_node)
        builder.add_node("format_answer", self._format_answer_node)
        
        # Simple linear flow
        builder.add_edge(START, "read_question")
        builder.add_edge("read_question", "agent_execution")
        builder.add_edge("agent_execution", "format_answer")
        builder.add_edge("format_answer", END)
        
        return builder.compile()
    
    # ============================================================================
    # WORKFLOW NODES
    # ============================================================================
    
    def _read_question_node(self, state: GAIAState):
        """Read and setup question with RAG context"""
        if self.logging:
            self.logging.logger.log_task(
                content=state["question"].strip(),
                title="GAIA Question Processing"
            )
        
        # RAG retrieval - find similar examples
        try:
            similar_docs = self.retriever.search(state["question"], k=self.config.rag_examples_count)
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
            similar_examples = []
        
        return {
            "similar_examples": similar_examples,
            "steps": state["steps"] + ["Question setup and RAG retrieval complete"]
        }
    
    def _agent_execution_node(self, state: GAIAState):
        """Execute SmolagAgent with tools"""
        task_id = state.get("task_id", str(uuid.uuid4()))
        
        # Start logging for this task
        if self.logging:
            self.logging.start_task(task_id)
        
        # Setup tools
        tools = self._setup_tools()
        
        # Create SmolagAgent
        step_callbacks = [self.logging.capture_step_log] if self.logging else []
        logger = self.logging.logger if self.logging else None
        
        agent = ToolCallingAgent(
            name="gaia_agent",
            model=self.model,
            tools=tools,
            add_base_tools=True,  # Includes Python interpreter, web search, transcriber
            max_steps=self.config.max_agent_steps,
            logger=logger,
            step_callbacks=step_callbacks
        )
        
        # Prepare question with RAG context
        question_with_context = self._prepare_question_context(
            state["question"], 
            state.get("similar_examples", [])
        )
        
        try:
            # Execute agent
            print(f"🤖 Executing SmolagAgent (max {self.config.max_agent_steps} steps)")
            result = agent.run(question_with_context)
            
            # Get step count
            step_count = self.logging.step_counter if self.logging else 0
            
            return {
                "raw_answer": str(result),
                "steps": state["steps"] + [
                    f"Agent execution complete - {step_count} steps"
                ]
            }
            
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                "raw_answer": error_msg,
                "steps": state["steps"] + [error_msg]
            }
    
    def _prepare_question_context(self, question: str, similar_examples: List[Dict]) -> str:
        """Prepare question with RAG context"""
        if not similar_examples:
            return question
        
        context_parts = [
            "Here are similar GAIA examples to guide your approach:",
            ""
        ]
        
        for i, example in enumerate(similar_examples[:2], 1):  # Limit to 2 examples
            context_parts.extend([
                f"Example {i}:",
                f"Question: {example['question']}",
                f"Answer: {example['answer']}",
                ""
            ])
        
        context_parts.extend([
            "Now solve this question following GAIA format requirements:",
            f"Question: {question}",
            "",
            "Provide your final answer in the format: FINAL ANSWER: [YOUR ANSWER]"
        ])
        
        return "\n".join(context_parts)
    
    def _format_answer_node(self, state: GAIAState):
        """Format answer according to GAIA rules"""
        raw_answer = state.get("raw_answer", "")
        
        try:
            # Extract FINAL ANSWER
            formatted_answer = self._extract_final_answer(raw_answer)
            
            # Apply GAIA formatting rules
            formatted_answer = self._apply_gaia_formatting(formatted_answer)
            
            return {
                "final_answer": formatted_answer,
                "steps": state["steps"] + ["Answer formatted according to GAIA rules"]
            }
            
        except Exception as e:
            error_msg = f"Answer formatting error: {str(e)}"
            return {
                "final_answer": raw_answer.strip() if raw_answer else "No answer",
                "steps": state["steps"] + [error_msg]
            }
    
    def _extract_final_answer(self, raw_answer: str) -> str:
        """Extract final answer from agent response"""
        if not raw_answer:
            return "No answer"
        
        import re
        
        # Try to find FINAL ANSWER pattern
        patterns = [
            r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
            r"Final Answer:\s*(.+?)(?:\n|$)",
            r"Answer:\s*(.+?)(?:\n|$)",
            r"The answer is:\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_answer, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, return last line
        lines = raw_answer.strip().split('\n')
        return lines[-1].strip() if lines else "No answer"
    
    def _apply_gaia_formatting(self, answer: str) -> str:
        """Apply GAIA benchmark formatting rules"""
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
        """Execute single question"""
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
# MAIN EXECUTION (Basic Test Only)
# ============================================================================

if __name__ == "__main__":
    print("🚀 Core GAIA Agent Module")
    print("=" * 30)
    print("Use gaia_interface.py for full testing capabilities")