# hybrid_gaia_agent.py
# Hybrid GAIA Architecture integrated with existing dev_retriever system

import json
import os
from typing import TypedDict, Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import uuid
from datetime import datetime

# Core dependencies
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# Import your existing retriever system
from dev_retriever import load_gaia_retriever, DevelopmentGAIARetriever

# ============================================================================
# CONFIGURATION USING YOUR EXISTING INFRASTRUCTURE
# ============================================================================

@dataclass
class HybridGAIAConfig:
    """Configuration that leverages your existing vector store"""
    # Model settings (multi-provider support)
    model_provider: str = "openrouter"  # openrouter, ollama
    primary_model: str = "qwen-2.5-coder-32b-instruct:free"
    temperature: float = 0.7
    
    # Use your existing retriever
    csv_file: str = "gaia_embeddings.csv"
    metadata_path: str = "metadata.jsonl"
    
    # RAG settings
    rag_examples_count: int = 3
    similarity_threshold: float = 0.7
    
    # Testing and development
    results_output_path: str = "gaia_results"
    debug_mode: bool = True
    save_intermediate_results: bool = True

# ============================================================================
# ENHANCED STATE MANAGEMENT
# ============================================================================

class HybridGAIAState(TypedDict):
    """State management combining LangGraph patterns with GAIA tracking"""
    # Core execution
    messages: List[BaseMessage]
    question: str
    task_id: str
    
    # RAG context (using your retriever)
    retriever_context: Optional[str]
    similar_examples: List[Dict]
    
    # Execution tracking
    selected_strategy: Optional[str]
    selected_agent: Optional[str]
    execution_steps: List[str]
    
    # Results
    raw_answer: Optional[str]
    final_answer: Optional[str]
    confidence_score: Optional[float]
    
    # Evaluation metadata
    ground_truth: Optional[str]
    level: Optional[int]
    execution_time: Optional[float]
    model_used: Optional[str]
    
    # Debug info
    debug_info: Dict[str, Any]

# ============================================================================
# METADATA INTEGRATION LAYER
# ============================================================================

class GAIAMetadataManager:
    """Manage GAIA metadata using your existing .jsonl format"""
    
    def __init__(self, config: HybridGAIAConfig):
        self.config = config
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> List[Dict]:
        """Load from your metadata.jsonl file"""
        try:
            with open(self.config.metadata_path, 'r', encoding='utf-8') as f:
                metadata = [json.loads(line) for line in f]
            print(f"✅ Loaded {len(metadata)} GAIA examples from metadata")
            return metadata
        except FileNotFoundError:
            print(f"❌ Metadata file not found: {self.config.metadata_path}")
            return []
    
    def get_test_sample(self, sample_size: Optional[int] = None) -> List[Dict]:
        """Get test sample for evaluation"""
        if sample_size is None:
            return self.metadata
        return self.metadata[:sample_size]
    
    def get_ground_truth(self, task_id: str) -> Optional[Dict]:
        """Get ground truth for a specific task"""
        for item in self.metadata:
            if item.get('task_id') == task_id:
                return item
        return None
    
    def analyze_tool_usage(self) -> Dict[str, int]:
        """Analyze tool usage patterns from metadata (for optimization)"""
        tool_frequencies = {}
        
        for item in self.metadata:
            question = item.get('Question', '').lower()
            
            # Tool detection patterns (same as your build_vectorstore.py logic)
            if any(word in question for word in ["calculate", "compute", "math", "number"]):
                tool_frequencies['calculator'] = tool_frequencies.get('calculator', 0) + 1
            if any(word in question for word in ["search", "find", "lookup", "google"]):
                tool_frequencies['web_search'] = tool_frequencies.get('web_search', 0) + 1
            if any(word in question for word in ["file", "attachment", "document", "image", "audio"]):
                tool_frequencies['file_processing'] = tool_frequencies.get('file_processing', 0) + 1
            if any(word in question for word in ["wikipedia", "wiki"]):
                tool_frequencies['wiki_search'] = tool_frequencies.get('wiki_search', 0) + 1
        
        return tool_frequencies

# ============================================================================
# HYBRID AGENT WITH YOUR EXISTING RETRIEVER
# ============================================================================

class HybridGAIAAgent:
    """Hybrid agent using your existing optimized retriever system"""
    
    def __init__(self, config: HybridGAIAConfig = None):
        if config is None:
            config = HybridGAIAConfig()
        
        self.config = config
        self.metadata_manager = GAIAMetadataManager(config)
        self.retriever = self._initialize_retriever()
        self.llm = self._initialize_model()
        self.workflow = self._build_workflow()
        
        # Performance tracking
        self.execution_results = []
        Path(config.results_output_path).mkdir(exist_ok=True)
        
        # Show tool usage analysis
        if config.debug_mode:
            tool_usage = self.metadata_manager.analyze_tool_usage()
            print("📊 Tool Usage Analysis from Metadata:")
            for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
                print(f"  ├── {tool}: {count} occurrences")
    
    def _initialize_retriever(self) -> DevelopmentGAIARetriever:
        """Initialize using your existing retriever system"""
        print("🔄 Setting up retriever using existing infrastructure...")
        retriever = load_gaia_retriever(self.config.csv_file)
        
        if retriever and retriever.is_ready():
            print("✅ Retriever ready!")
            return retriever
        else:
            raise RuntimeError("❌ Failed to initialize retriever. Check your vector store.")
    
    def _initialize_model(self):
        """Initialize LLM with multi-provider support"""
        if self.config.model_provider == "groq":
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
        elif self.config.model_provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.config.primary_model,
                temperature=self.config.temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.model_provider}")
    
    def _build_workflow(self):
        """Build the hybrid workflow"""
        builder = StateGraph(HybridGAIAState)
        
        # Add nodes
        builder.add_node("initialize", self._initialize_node)
        builder.add_node("rag_retrieval", self._rag_retrieval_node)
        builder.add_node("strategy_selection", self._strategy_selection_node)
        builder.add_node("smolag_execution", self._smolag_execution_node)
        builder.add_node("direct_llm_execution", self._direct_llm_execution_node)
        builder.add_node("answer_formatting", self._answer_formatting_node)
        builder.add_node("evaluation", self._evaluation_node)
        
        # Define workflow (RAG-first)
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "rag_retrieval")
        builder.add_edge("rag_retrieval", "strategy_selection")
        
        # Conditional routing
        builder.add_conditional_edges(
            "strategy_selection",
            self._route_execution,
            {
                "smolag_agent": "smolag_execution",
                "direct_llm": "direct_llm_execution"
            }
        )
        
        builder.add_edge("smolag_execution", "answer_formatting")
        builder.add_edge("direct_llm_execution", "answer_formatting")
        builder.add_edge("answer_formatting", "evaluation")
        builder.add_edge("evaluation", END)
        
        return builder.compile()
    
    # ============================================================================
    # WORKFLOW NODE IMPLEMENTATIONS
    # ============================================================================
    
    def _initialize_node(self, state: HybridGAIAState):
        """Initialize execution state"""
        start_time = datetime.now()
        
        return {
            "messages": [HumanMessage(content=state["question"])],
            "execution_steps": ["Execution initialized"],
            "debug_info": {
                "start_time": start_time.isoformat(),
                "model_provider": self.config.model_provider,
                "model_name": self.config.primary_model
            }
        }
    
    def _rag_retrieval_node(self, state: HybridGAIAState):
        """RAG retrieval using your existing retriever (exactly like student #1)"""
        question = state["question"]
        messages = state["messages"]
        
        # Use your retriever's exact method (same as student #1)
        rag_result = self.retriever.retriever_node(messages)
        
        # Extract the RAG context from the result
        if len(rag_result["messages"]) > len(messages):
            rag_message = rag_result["messages"][-1]
            retriever_context = rag_message.content
        else:
            retriever_context = "No similar examples found"
        
        # Also get raw similar examples for analysis
        similar_docs = self.retriever.search(question, k=self.config.rag_examples_count)
        similar_examples = []
        
        for doc in similar_docs:
            # Parse the Q&A format from your vector store
            content = doc.page_content
            if "Question :" in content and "Final answer :" in content:
                parts = content.split("Final answer :")
                if len(parts) == 2:
                    q_part = parts[0].replace("Question :", "").strip()
                    a_part = parts[1].strip()
                    similar_examples.append({
                        "question": q_part,
                        "answer": a_part,
                        "source": doc.metadata.get("source", "unknown")
                    })
        
        return {
            "messages": rag_result["messages"],  # Use exact result from your retriever
            "retriever_context": retriever_context,
            "similar_examples": similar_examples,
            "execution_steps": state["execution_steps"] + [
                f"RAG retrieval completed - found {len(similar_examples)} examples"
            ]
        }
    
    def _strategy_selection_node(self, state: HybridGAIAState):
        """Select execution strategy based on RAG context and complexity"""
        question = state["question"]
        similar_examples = state.get("similar_examples", [])
        
        # Analyze complexity using similar examples and question patterns
        complexity_analysis = self._analyze_question_complexity(question, similar_examples)
        
        # Decision logic
        if complexity_analysis["complexity_score"] > 0.6 or complexity_analysis["requires_tools"]:
            strategy = "smolag_agent"
            selected_agent = complexity_analysis["recommended_agent"]
        else:
            strategy = "direct_llm"
            selected_agent = None
        
        return {
            "selected_strategy": strategy,
            "selected_agent": selected_agent,
            "execution_steps": state["execution_steps"] + [
                f"Strategy: {strategy}" + (f" (agent: {selected_agent})" if selected_agent else ""),
                f"Complexity score: {complexity_analysis['complexity_score']:.2f}"
            ],
            "debug_info": {
                **state.get("debug_info", {}),
                "complexity_analysis": complexity_analysis
            }
        }
    
    def _analyze_question_complexity(self, question: str, similar_examples: List[Dict]) -> Dict:
        """Analyze question complexity for strategy selection"""
        complexity_indicators = []
        complexity_score = 0.0
        requires_tools = False
        
        question_lower = question.lower()
        
        # File attachment complexity
        if any(word in question_lower for word in ["file", "attachment", "image", "audio", "document", "spreadsheet"]):
            complexity_indicators.append("file_processing")
            complexity_score += 0.5
            requires_tools = True
        
        # Calculation complexity
        if any(word in question_lower for word in ["calculate", "compute", "sum", "average", "statistics", "math"]):
            complexity_indicators.append("calculations")
            complexity_score += 0.4
            requires_tools = True
        
        # Web search complexity
        if any(word in question_lower for word in ["current", "latest", "recent", "search", "find", "wikipedia"]):
            complexity_indicators.append("web_search")
            complexity_score += 0.3
            requires_tools = True
        
        # Multi-step complexity
        if len(question.split()) > 25 or question.count("?") > 1:
            complexity_indicators.append("multi_step")
            complexity_score += 0.2
        
        # Example-based complexity (use similar examples to inform decision)
        if similar_examples:
            # If similar examples suggest complex patterns, increase complexity
            for example in similar_examples:
                example_q = example.get("question", "").lower()
                if any(indicator in example_q for indicator in ["calculate", "file", "search", "complex"]):
                    complexity_score += 0.1
        
        # Recommend agent based on primary complexity
        agent_mapping = {
            "file_processing": "document_processor",
            "calculations": "data_analyst",
            "web_search": "web_researcher",
            "multi_step": "general_assistant"
        }
        
        recommended_agent = "general_assistant"  # default
        if complexity_indicators:
            recommended_agent = agent_mapping.get(complexity_indicators[0], "general_assistant")
        
        return {
            "complexity_score": min(complexity_score, 1.0),
            "complexity_indicators": complexity_indicators,
            "recommended_agent": recommended_agent,
            "requires_tools": requires_tools,
            "similar_examples_count": len(similar_examples)
        }
    
    def _smolag_execution_node(self, state: HybridGAIAState):
        """Execute using SmolagAgent (placeholder for your actual implementation)"""
        question = state["question"]
        agent = state["selected_agent"]
        context = state.get("retriever_context", "")
        
        # Placeholder - replace with your actual SmolagAgent integration
        execution_result = self._simulate_smolag_execution(question, agent, context)
        
        return {
            "raw_answer": execution_result["answer"],
            "execution_steps": state["execution_steps"] + execution_result["steps"],
            "debug_info": {
                **state.get("debug_info", {}),
                "agent_used": agent,
                "tools_used": execution_result["tools_used"]
            }
        }
    
    def _simulate_smolag_execution(self, question: str, agent: str, context: str) -> Dict:
        """Simulate SmolagAgent - replace with actual integration"""
        # This is where you'd integrate your actual SmolagAgents
        
        tools_map = {
            "data_analyst": ["calculator", "pandas", "numpy"],
            "web_researcher": ["web_search", "tavily_search"],
            "document_processor": ["file_reader", "speech_recognition"],
            "general_assistant": ["calculator", "web_search"]
        }
        
        # Simulate processing with context
        context_summary = context[:200] + "..." if len(context) > 200 else context
        
        return {
            "answer": f"[{agent}] Based on similar examples: {context_summary}\n\nAnswer: {question[:50]}...",
            "steps": [
                f"SmolagAgent {agent} initialized",
                f"RAG context processed: {len(context)} chars",
                f"Tools available: {tools_map.get(agent, [])}",
                "Task executed"
            ],
            "tools_used": tools_map.get(agent, [])
        }
    
    def _direct_llm_execution_node(self, state: HybridGAIAState):
        """Direct LLM execution with RAG context (like student #2)"""
        question = state["question"]
        messages = state["messages"]  # Already includes RAG context
        
        # Build GAIA-compliant prompt
        system_prompt = """You are a general AI assistant. Report your thoughts, and finish with: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list.
- Numbers: no commas, no units ($ %) unless specified
- Strings: no articles (the, a, an), no abbreviations, digits as text unless specified  
- Lists: apply above rules to each element"""
        
        final_messages = [SystemMessage(content=system_prompt)] + messages
        response = self.llm.invoke(final_messages)
        
        return {
            "raw_answer": response.content,
            "execution_steps": state["execution_steps"] + [
                "Direct LLM execution with RAG context"
            ]
        }
    
    def _answer_formatting_node(self, state: HybridGAIAState):
        """Format answer according to GAIA rules"""
        raw_answer = state.get("raw_answer", "")
        
        # Extract FINAL ANSWER if present
        if "FINAL ANSWER:" in raw_answer:
            formatted_answer = raw_answer.split("FINAL ANSWER:")[-1].strip()
        else:
            formatted_answer = self._apply_gaia_formatting(raw_answer)
        
        return {
            "final_answer": formatted_answer,
            "execution_steps": state["execution_steps"] + ["GAIA formatting applied"]
        }
    
    def _apply_gaia_formatting(self, raw_answer: str) -> str:
        """Apply GAIA benchmark formatting rules"""
        if not raw_answer:
            return "No answer"
        
        answer = raw_answer.strip()
        
        # Remove common prefixes
        prefixes = ["the answer is", "answer:", "final answer:", "result:", "solution:"]
        for prefix in prefixes:
            if answer.lower().startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Remove trailing punctuation for short answers
        if len(answer.split()) < 5:
            answer = answer.rstrip('.,!?')
        
        return answer
    
    def _evaluation_node(self, state: HybridGAIAState):
        """Evaluate result and compute metrics"""
        end_time = datetime.now()
        start_time_str = state.get("debug_info", {}).get("start_time")
        
        execution_time = None
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            execution_time = (end_time - start_time).total_seconds()
        
        # Calculate accuracy if ground truth available
        is_correct = None
        if state.get("ground_truth"):
            is_correct = self._evaluate_answer(state["final_answer"], state["ground_truth"])
        
        return {
            "execution_time": execution_time,
            "model_used": f"{self.config.model_provider}/{self.config.primary_model}",
            "execution_steps": state["execution_steps"] + ["Evaluation completed"],
            "debug_info": {
                **state.get("debug_info", {}),
                "end_time": end_time.isoformat(),
                "is_correct": is_correct,
                "execution_time": execution_time
            }
        }
    
    def _evaluate_answer(self, predicted: str, ground_truth: str) -> bool:
        """Evaluate answer accuracy"""
        if not predicted or not ground_truth:
            return False
        
        # Normalize for comparison
        pred_normalized = predicted.lower().strip()
        truth_normalized = ground_truth.lower().strip()
        
        return pred_normalized == truth_normalized
    
    def _route_execution(self, state: HybridGAIAState) -> str:
        """Route to appropriate execution strategy"""
        return state.get("selected_strategy", "direct_llm")
    
    # ============================================================================
    # INTERFACE METHODS
    # ============================================================================
    
    def run_single_question(self, question: str, task_id: str = None, ground_truth: str = None, level: int = None) -> Dict:
        """Run a single question through the hybrid agent"""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        initial_state = {
            "question": question,
            "task_id": task_id,
            "ground_truth": ground_truth,
            "level": level
        }
        
        if self.config.debug_mode:
            print(f"🔍 Processing: {question[:50]}...")
        
        result = self.workflow.invoke(initial_state)
        
        # Store result for analysis
        execution_record = {
            "task_id": task_id,
            "question": question,
            "final_answer": result.get("final_answer"),
            "ground_truth": ground_truth,
            "level": level,
            "is_correct": result.get("debug_info", {}).get("is_correct"),
            "execution_time": result.get("execution_time"),
            "strategy_used": result.get("selected_strategy"),
            "selected_agent": result.get("selected_agent"),
            "model_used": result.get("model_used"),
            "similar_examples_count": len(result.get("similar_examples", [])),
            "timestamp": datetime.now().isoformat()
        }
        
        self.execution_results.append(execution_record)
        
        if self.config.save_intermediate_results:
            self._save_result(execution_record)
        
        return result
    
    def run_batch_evaluation(self, sample_size: int = None) -> pd.DataFrame:
        """Run evaluation on GAIA dataset using your metadata"""
        test_data = self.metadata_manager.get_test_sample(sample_size)
        
        print(f"🧪 Running evaluation on {len(test_data)} questions...")
        print("=" * 50)
        
        results = []
        for i, example in enumerate(test_data):
            if self.config.debug_mode and (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(test_data)} ({(i+1)/len(test_data)*100:.1f}%)")
            
            result = self.run_single_question(
                question=example.get("Question", ""),
                task_id=example.get("task_id", f"eval_{i}"),
                ground_truth=example.get("Final answer", ""),
                level=example.get("Level", 1)
            )
            
            results.append(self.execution_results[-1])
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Calculate metrics
        accuracy = df['is_correct'].mean() if 'is_correct' in df and df['is_correct'].notna().any() else 0
        avg_execution_time = df['execution_time'].mean() if 'execution_time' in df else 0
        
        print(f"\n🎯 EVALUATION RESULTS")
        print("=" * 30)
        print(f"Total questions: {len(df)}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Average execution time: {avg_execution_time:.2f}s")
        
        # Strategy breakdown
        if 'strategy_used' in df:
            strategy_counts = df['strategy_used'].value_counts()
            print(f"\nStrategy usage:")
            for strategy, count in strategy_counts.items():
                strategy_accuracy = df[df['strategy_used'] == strategy]['is_correct'].mean()
                print(f"  ├── {strategy}: {count} questions, {strategy_accuracy:.3f} accuracy")
        
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
            f.write(json.dumps(result) + "\n")
    
    def close(self):
        """Clean shutdown"""
        if hasattr(self.retriever, 'close'):
            self.retriever.close()

# ============================================================================
# CONVENIENCE FUNCTIONS FOR NOTEBOOK USE
# ============================================================================

def create_hybrid_gaia_agent(config_overrides: Dict = None) -> HybridGAIAAgent:
    """Factory function for easy notebook usage"""
    config = HybridGAIAConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)
    
    return HybridGAIAAgent(config)

def quick_test_agent(question: str = "What is the capital of France?") -> Dict:
    """Quick test function for notebook experimentation"""
    agent = create_hybrid_gaia_agent()
    result = agent.run_single_question(question)
    agent.close()
    return result

# ============================================================================
# NOTEBOOK USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("🚀 Hybrid GAIA Agent - Using Existing Infrastructure")
    print("=" * 60)
    
    # Quick test
    print("🧪 Quick test...")
    result = quick_test_agent("Calculate 15% of 1000")
    print(f"Question: {result['question']}")
    print(f"Final Answer: {result['final_answer']}")
    print(f"Strategy: {result['selected_strategy']}")
    print(f"Execution Time: {result.get('execution_time', 0):.2f}s")

"""
# Notebook Usage Examples:

# Cell 1: Basic Setup
from hybrid_gaia_agent import create_hybrid_gaia_agent

agent = create_hybrid_gaia_agent({
    "model_provider": "groq",
    "primary_model": "qwen-qwq-32b",
    "debug_mode": True
})

# Cell 2: Test Single Question
result = agent.run_single_question("What is the population of Tokyo?")
print(f"Answer: {result['final_answer']}")
print(f"Strategy: {result['selected_strategy']}")

# Cell 3: Run Evaluation
results_df = agent.run_batch_evaluation(sample_size=20)

# Cell 4: Analyze Performance 
accuracy_by_level = results_df.groupby('level')['is_correct'].mean()
print(accuracy_by_level)

# Cell 5: Compare Models
models_to_test = [
    {"model_provider": "groq", "primary_model": "qwen-qwq-32b"},
    {"model_provider": "google", "primary_model": "gemini-2.0-flash"}
]

for model_config in models_to_test:
    agent = create_hybrid_gaia_agent(model_config)
    results = agent.run_batch_evaluation(sample_size=10)
    accuracy = results['is_correct'].mean()
    print(f"{model_config['model_provider']}: {accuracy:.3f}")
    agent.close()

# Cell 6: Strategy Analysis
strategy_performance = results_df.groupby('strategy_used').agg({
    'is_correct': 'mean',
    'execution_time': 'mean',
    'task_id': 'count'
}).round(3)
print(strategy_performance)
"""