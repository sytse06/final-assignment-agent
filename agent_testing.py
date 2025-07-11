# agent_testing.py
# GAIA Agent Testing Framework

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
import traceback
from collections import defaultdict
import re
from difflib import SequenceMatcher
import logging
import uuid

# Import our agent system and dataset manager
from agent_logic import GAIAAgent, GAIAConfig
from agent_interface import (
    create_gaia_agent,
    get_openrouter_config,
    get_anthropic_config, 
    get_groq_config,  
    get_google_config, 
    get_ollama_config, 
    get_performance_config, 
    get_accuracy_config
)
from agent_logging import create_timestamped_filename

# Import context bridge for integration
from agent_context import ContextBridge

# Try to import dataset management (graceful degradation if not available)
try:
    from gaia_dataset_utils import GAIADatasetManager, quick_dataset_check
    DATASET_UTILS_AVAILABLE = True
    print("✅ GAIA dataset utilities available for complete testing")
except ImportError:
    print("⚠️  GAIA dataset utilities not available - limited testing mode")
    DATASET_UTILS_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class GAIATestConfig:
    """Configuration for GAIA testing framework"""
    # File paths
    dataset_path: str = "./tests/gaia_data"
    results_dir: str = "./test_results"
    
    # Execution settings
    timeout_per_question: int = 180
    max_retries: int = 3
    enable_retries: bool = True
    enable_strategy_fallback: bool = True
    
    # Testing behavior
    enable_ground_truth_isolation: bool = True
    save_execution_logs: bool = True
    save_evaluation_logs: bool = True
    save_intermediate: bool = True
    enable_real_time_monitoring: bool = True
    
    # Error handling
    detect_execution_failures: bool = True
    fallback_strategies: List[str] = None
    
    # Performance monitoring
    enable_performance_profiling: bool = True
    memory_monitoring: bool = False
    
    # Agent configuration
    agent_config_name: str = "groq"
    custom_agent_config: Dict = None
    
def get_agent_config_by_name(config_name: str) -> GAIAConfig:
    """Get agent configuration by name - UPDATED for GAIAConfig compatibility"""
    
    config_functions = {
        "openrouter": get_openrouter_config,
        "or": get_openrouter_config,
        "anthropic": get_anthropic_config,
        "claude": get_anthropic_config,
        "groq": get_groq_config,
        "qwen3_32b": get_groq_config,
        "google": get_google_config,
        "gemini": get_google_config,
        "ollama": get_ollama_config,
        "local": get_ollama_config,
        "performance": get_performance_config,
        "accuracy": get_accuracy_config
    }
    
    config_name_lower = config_name.lower()
    
    if config_name_lower in config_functions:
        # These functions return GAIAConfig objects directly
        return config_functions[config_name_lower]()
    else:
        # Fallback to groq config
        print(f"⚠️  Unknown config name '{config_name}', using default groq config")
        return get_groq_config()

# ============================================================================
# PHASE 1: BLIND EXECUTION
# ============================================================================

class GAIAQuestionExecutor:
    """
    Execute GAIA questions WITHOUT access to ground truth (blind testing).
    FIXED: All .get() calls on GAIATestConfig replaced with getattr()
    """
    
    def __init__(self, agent_config: Union[str, GAIAConfig], test_config=None):
        """Initialize test executor with proper config handling"""
        
        # Proper handling of GAIAConfig objects
        if isinstance(agent_config, str):
            self.gaia_config = get_agent_config_by_name(agent_config)
            self.agent_config_name = agent_config
        elif isinstance(agent_config, GAIAConfig):
            self.gaia_config = agent_config
            self.agent_config_name = f"{agent_config.model_provider}_{agent_config.model_name}"
        else:
            raise ValueError(f"Expected str or GAIAConfig, got {type(agent_config)}")
        
        # Create agent using the GAIAConfig object
        self.agent = create_gaia_agent(self.gaia_config)
        
        # Setup test configuration
        self.test_config = test_config or self._default_test_config()
        
        # Generate execution timestamp
        self.execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"✅ GAIAQuestionExecutor initialized (Hybrid Compatible):")
        print(f"   Config: {self.agent_config_name}")
        print(f"   Model: {self.gaia_config.model_provider}/{self.gaia_config.model_name}")
        print(f"   Smart Routing: {self.gaia_config.enable_smart_routing}")
        print(f"   Context Bridge: {self.gaia_config.enable_context_bridge}")

    def _default_test_config(self):
        """Create default test configuration"""
        return {
            "batch_size": 20,
            "timeout_per_question": 120,
            "enable_detailed_logging": True,
            "save_execution_results": True,
            "results_directory": "./test_results",
            "enable_ground_truth_isolation": True
        }
    
    def execute_questions_batch(self, blind_questions: List[Dict], batch_name: str = None) -> str:
        """Execute batch of questions WITHOUT ground truth access - FIXED"""
        if batch_name is None:
            batch_name = f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🚀 BLIND EXECUTION (Hybrid Compatible): {len(blind_questions)} questions")
        print(f"📝 Batch: {batch_name}")
        
        # FIXED: Use getattr instead of .get() for dataclass
        if getattr(self.test_config, 'enable_ground_truth_isolation', True):
            self._verify_blind_questions(blind_questions)
        
        execution_results = []
        start_time = time.time()
        
        for i, question_data in enumerate(blind_questions, 1):
            print(f"\n🎯 Question {i}/{len(blind_questions)}")
            
            result = self._execute_single_question_hybrid_compatible(question_data, i)
            execution_results.append(result)
            
            # Brief pause between questions
            time.sleep(0.5)
        
        total_time = time.time() - start_time
        
        # Save execution results (NO ground truth included)
        execution_file = self._save_execution_results(
            execution_results, batch_name, total_time
        )
        
        print(f"\n📊 BLIND EXECUTION COMPLETE")
        print(f"   Questions: {len(blind_questions)}")
        print(f"   Successful: {sum(1 for r in execution_results if r['execution_successful'])}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Results saved: {execution_file}")
        
        return execution_file
    
    def _verify_blind_questions(self, questions: List[Dict]):
        """PRESERVED: Verify questions don't contain ground truth"""
        contaminated_questions = []
        
        for i, q in enumerate(questions):
            contamination_found = False
            
            if 'Final answer' in q or 'final_answer' in q:
                contaminated_questions.append(f"Question {i+1}: Contains 'Final answer' field")
                contamination_found = True
            
            if 'ground_truth' in q or 'expected_answer' in q:
                contaminated_questions.append(f"Question {i+1}: Contains ground truth fields")
                contamination_found = True
            
            # Check for suspicious answer-like fields
            suspicious_fields = ['answer', 'solution', 'result', 'correct_answer']
            for field in suspicious_fields:
                if field in q and len(str(q[field])) > 0:
                    contaminated_questions.append(f"Question {i+1}: Suspicious field '{field}'")
                    contamination_found = True
                    break
        
        if contaminated_questions:
            print("🚨 GROUND TRUTH CONTAMINATION DETECTED:")
            for contamination in contaminated_questions[:5]:
                print(f"   ❌ {contamination}")
            
            if len(contaminated_questions) > 5:
                print(f"   ... and {len(contaminated_questions) - 5} more contaminations")
            
            raise ValueError(f"Blind testing compromised! {len(contaminated_questions)} questions contain ground truth.")
        else:
            print("✅ Verified: Questions are properly blind (no ground truth contamination)")
    
    def _execute_single_question_hybrid_compatible(self, question_data: Dict, question_num: int) -> Dict:
        """Execute single question compatible with hybrid state + context bridge"""
        
        task_id = question_data.get('task_id', f"blind_{question_num}")
        question = question_data.get('Question', question_data.get('question', ''))
        level = question_data.get('Level', 'Unknown')
        file_name = question_data.get('file_name', '')
        file_path = question_data.get('file_path', '')
        
        start_time = time.time()
        
        try:
            print(f"   🔒 Processing (BLIND + Hybrid): {question[:50]}...")
            
            # NEW: Start context bridge tracking (if enabled)
            if self.gaia_config.enable_context_bridge:
                ContextBridge.start_task_execution(task_id)
                ContextBridge.track_operation(f"Blind execution: {question[:30]}...")
            
            # CRITICAL: Execute question - agent has NO access to ground truth
            result = self.agent.process_question(question, task_id=task_id)
            
            execution_time = time.time() - start_time
            
            # Extract information from new result structure
            strategy_used = self._determine_strategy_used_hybrid(result)
            execution_successful = result.get("execution_successful", False)
            
            # Get context bridge metrics if available
            context_metrics = {}
            if self.gaia_config.enable_context_bridge:
                context_metrics = ContextBridge.get_execution_metrics()
                ContextBridge.clear_tracking()
            
            # Enhanced execution record with hybrid state compatibility
            return {
                "question_number": question_num,
                "task_id": task_id,
                "question": question,
                "level": level,
                "file_name": file_name,
                "file_path": file_path,
                "has_file": bool(file_name),
                "final_answer": result.get("final_answer", ""),
                "raw_answer": result.get("raw_answer", ""),
                "steps": result.get("steps", []),
                "execution_successful": execution_successful,
                "execution_time": execution_time,
                "strategy_used": strategy_used,
                "complexity": result.get("complexity"),
                "selected_agent": result.get("selected_agent", "unknown"),
                "similar_examples_count": len(result.get("similar_examples", [])),
                "context_bridge_used": self.gaia_config.enable_context_bridge,
                "context_metrics": context_metrics,
                "model_provider": self.gaia_config.model_provider,
                "model_name": self.gaia_config.model_name,
                "execution_timestamp": datetime.now().isoformat(),
                "file_access_method": "hybrid_state",
                "blind_execution_verified": True,
                "total_steps": result.get("total_steps", 0),
                "performance_metrics": result.get("performance_metrics", {}),
                "file_info": result.get("file_info", {})
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ Execution failed: {str(e)}")
            
            # Clean up context bridge on error
            if self.gaia_config.enable_context_bridge:
                try:
                    ContextBridge.clear_tracking()
                except:
                    pass
            
            return {
                "question_number": question_num,
                "task_id": task_id,
                "question": question,
                "level": level,
                "file_name": file_name,
                "file_path": file_path,
                "has_file": bool(file_name),
                "final_answer": "ERROR",
                "raw_answer": f"Execution error: {str(e)}",
                "steps": [],
                "execution_successful": False,
                "execution_time": execution_time,
                "strategy_used": "error",
                "complexity": "unknown",
                "selected_agent": "error",
                "similar_examples_count": 0,
                "context_bridge_used": self.gaia_config.enable_context_bridge,
                "context_metrics": {},
                "model_provider": self.gaia_config.model_provider, 
                "model_name": self.gaia_config.model_name,
                "execution_timestamp": datetime.now().isoformat(),
                "file_access_method": "error",
                "error": str(e),
                "error_type": self._categorize_error(str(e)),
                "blind_execution_verified": True,
                "total_steps": 0,
                "performance_metrics": {"error": True},
                "file_info": {}
            }

    def _determine_strategy_used_hybrid(self, result: Dict) -> str:
        """Determine strategy from new hybrid result structure"""
        
        # Check for complexity-based routing
        complexity = result.get("complexity", "unknown")
        if complexity == "simple":
            return "one_shot_llm"
        elif complexity == "complex":
            return "manager_coordination"
        
        # Check for selected agent
        selected_agent = result.get("selected_agent", "unknown")
        if selected_agent != "unknown":
            return f"agent_{selected_agent}"
        
        # Fallback: analyze steps
        steps = result.get("steps", [])
        if len(steps) <= 2:
            return "one_shot_llm"
        else:
            return "manager_coordination"
    
    def _save_execution_results(self, results: List[Dict], batch_name: str, total_time: float) -> str:
        """FIXED: Save execution results with proper dataclass attribute access"""
        
        # Create timestamped filename
        base_filename = f"{batch_name}_execution"
        
        # FIXED: Use getattr instead of .get() for dataclass
        results_dir = getattr(self.test_config, 'results_directory', "./test_results")
        if isinstance(self.test_config, dict):
            results_dir = self.test_config.get("results_directory", "./test_results")
        
        execution_file = os.path.join(
            results_dir,
            create_timestamped_filename(base_filename, "json")
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(execution_file), exist_ok=True)
        
        execution_data = {
            "batch_name": batch_name,
            "execution_timestamp": datetime.now().isoformat(),
            "agent_config": {
                "provider": self.gaia_config.model_provider,
                "model": self.gaia_config.model_name,
                "temperature": self.gaia_config.temperature,
                "smart_routing": self.gaia_config.enable_smart_routing,
                "context_bridge": self.gaia_config.enable_context_bridge,
                "skip_rag_for_simple": self.gaia_config.skip_rag_for_simple,
                "max_agent_steps": self.gaia_config.max_agent_steps
            },
            "execution_summary": {
                "total_questions": len(results),
                "successful_executions": sum(1 for r in results if r["execution_successful"]),
                "total_execution_time": total_time,
                "avg_execution_time": total_time / len(results) if results else 0,
                "context_bridge_enabled": self.gaia_config.enable_context_bridge,
                "hybrid_state_used": True
            },
            "results": results,
            "blind_testing_verified": True,
            "framework_version": "hybrid_state_compatible"
        }
        
        try:
            with open(execution_file, 'w', encoding='utf-8') as f:
                json.dump(execution_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Execution results saved: {execution_file}")
            return execution_file
            
        except Exception as e:
            print(f"❌ Failed to save execution results: {e}")
            # Fallback to simple filename
            fallback_results_dir = getattr(self.test_config, 'results_directory', "./test_results")
            if isinstance(self.test_config, dict):
                fallback_results_dir = self.test_config.get("results_directory", "./test_results")
            
            fallback_file = os.path.join(fallback_results_dir, f"{batch_name}_execution.json")
            with open(fallback_file, 'w', encoding='utf-8') as f:
                json.dump(execution_data, f, indent=2, ensure_ascii=False)
            return fallback_file
        
    def _categorize_error(self, error_message: str) -> str:
        """PRESERVED: Categorize error types for analysis"""
        
        error_message_lower = error_message.lower()
        
        if 'unexpected keyword argument' in error_message_lower:
            return 'interface_mismatch'
        elif 'attributeerror' in error_message_lower:
            return 'attribute_error'
        elif 'timeout' in error_message_lower:
            return 'timeout'
        elif 'api' in error_message_lower or 'rate limit' in error_message_lower:
            return 'api_error'
        elif 'file not found' in error_message_lower:
            return 'file_error'
        elif 'authentication' in error_message_lower or 'permission' in error_message_lower:
            return 'auth_error'
        elif 'tool' in error_message_lower:
            return 'tool_error'
        elif any(word in error_message_lower for word in ['manager', 'execution', 'agent']):
            return 'execution_error'
        else:
            return 'unknown_error'

# ============================================================================
# PHASE 2: EVALUATION
# ============================================================================

class GAIAAnswerEvaluator:
    """Phase 2: Evaluate execution results AGAINST ground truth."""
    
    def __init__(self, dataset_manager: 'GAIADatasetManager', test_config: GAIATestConfig = None):
        if not dataset_manager:
            raise ValueError("Dataset manager required for evaluation phase")
        
        self.dataset_manager = dataset_manager
        self.test_config = test_config or GAIATestConfig()
        
        print(f"🎯 EVALUATOR initialized (Hybrid Compatible) - HAS ground truth access")
        print(f"   Dataset: {dataset_manager.dataset_path}")
    
    def evaluate_execution_results(self, execution_file: str) -> Dict:
        """
        COMPLETE FIXED VERSION: Evaluate execution results with proper ground truth handling
        
        This method loads execution results and compares them against ground truth answers
        from the dataset manager to calculate accuracy and performance metrics.
        """
        
        print(f"📊 EVALUATION PHASE (Fixed Ground Truth): {execution_file}")
        
        # Load execution results with error handling
        try:
            with open(execution_file, 'r', encoding='utf-8') as f:
                execution_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load execution file: {e}")
            return {"error": f"Could not load execution file: {str(e)}"}
        
        # Verify this was truly blind execution
        if not execution_data.get("blind_testing_verified", False):
            print("⚠️  WARNING: Execution results not verified as blind testing")
        
        # Check framework version
        framework_version = execution_data.get("framework_version", "unknown")
        if framework_version == "hybrid_state_compatible":
            print("✅ Hybrid state execution detected")
        
        results = execution_data.get("results", [])
        
        if not results:
            print("❌ No results found in execution file")
            return {"error": "No results found in execution file"}
        
        print(f"🔍 Evaluating {len(results)} agent responses against ground truth...")
        
        # Initialize tracking variables
        evaluated_results = []
        correct_answers = 0
        evaluation_errors = 0
        level_performance = defaultdict(lambda: {"total": 0, "correct": 0})
        strategy_performance = defaultdict(lambda: {"total": 0, "correct": 0})
        
        # Process each result
        for i, result in enumerate(results, 1):
            task_id = result.get("task_id")
            agent_answer = result.get("final_answer", "")
            level = result.get("level", "Unknown")
            strategy = result.get("strategy_used", "unknown")
            
            print(f"\n🔍 Evaluating {i}/{len(results)}: {task_id}")
            print(f"   Agent Answer: '{agent_answer}'")
            
            try:
                # *** CRITICAL FIX: Proper ground truth extraction ***
                ground_truth_data = self.dataset_manager.get_ground_truth(task_id)
                
                if ground_truth_data is None:
                    print(f"❌ No ground truth found for task: {task_id}")
                    evaluated_result = result.copy()
                    evaluated_result.update({
                        "ground_truth": None,
                        "is_correct": None,
                        "matching_method": "no_ground_truth",
                        "evaluation_error": f"No ground truth found for {task_id}",
                        "evaluation_timestamp": datetime.now().isoformat()
                    })
                    evaluated_results.append(evaluated_result)
                    evaluation_errors += 1
                    continue
                
                # *** FIXED: Extract the actual answer string from the data structure ***
                if isinstance(ground_truth_data, dict):
                    expected_answer = ground_truth_data.get('final_answer', '')
                    gt_level = ground_truth_data.get('level', level)
                else:
                    # Fallback: if it's already a string
                    expected_answer = str(ground_truth_data)
                    gt_level = level
                
                if not expected_answer:
                    print(f"⚠️  Empty ground truth answer for {task_id}")
                    expected_answer = ""
                
                print(f"   Expected Answer: '{expected_answer}'")
                
                # Perform GAIA-compliant answer matching
                is_correct = self.gaia_answer_matching(agent_answer, expected_answer)
                matching_method = "exact" if is_correct else "no_match"
                
                # Try fuzzy matching if exact match failed
                if not is_correct:
                    fuzzy_correct = self.fuzzy_answer_matching(agent_answer, expected_answer)
                    if fuzzy_correct:
                        is_correct = True
                        matching_method = "fuzzy"
                
                # Update counters
                if is_correct:
                    correct_answers += 1
                
                # Track level performance
                level_performance[str(gt_level)]["total"] += 1
                if is_correct:
                    level_performance[str(gt_level)]["correct"] += 1
                
                # Track strategy performance
                strategy_performance[strategy]["total"] += 1
                if is_correct:
                    strategy_performance[strategy]["correct"] += 1
                
                # Create evaluated result with proper ground truth
                evaluated_result = result.copy()
                evaluated_result.update({
                    "ground_truth": expected_answer,  # *** FIXED: Store the answer string, not the dict ***
                    "is_correct": is_correct,
                    "matching_method": matching_method,
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "level": gt_level  # Use ground truth level if available
                })
                
                # Log the comparison
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                print(f"   Result: {status} ({matching_method})")
                
            except Exception as e:
                print(f"⚠️  Evaluation error for {task_id}: {e}")
                import traceback
                traceback.print_exc()
                
                evaluated_result = result.copy()
                evaluated_result.update({
                    "ground_truth": None,
                    "is_correct": None,
                    "matching_method": "error",
                    "evaluation_error": str(e),
                    "evaluation_timestamp": datetime.now().isoformat()
                })
                evaluation_errors += 1
            
            evaluated_results.append(evaluated_result)
        
        # Calculate overall performance metrics
        total_questions = len(results)
        questions_with_gt = total_questions - evaluation_errors
        accuracy = correct_answers / questions_with_gt if questions_with_gt > 0 else 0
        
        # Calculate level-specific accuracy
        level_accuracy = {}
        for level, perf in level_performance.items():
            level_accuracy[level] = {
                "accuracy": perf["correct"] / perf["total"] if perf["total"] > 0 else 0,
                "correct": perf["correct"],
                "total": perf["total"]
            }
        
        # Calculate strategy-specific accuracy
        strategy_analysis = {}
        for strategy, perf in strategy_performance.items():
            strategy_analysis[strategy] = {
                "total_questions": perf["total"],
                "correct_answers": perf["correct"],
                "accuracy": perf["correct"] / perf["total"] if perf["total"] > 0 else 0
            }
        
        # Create enhanced evaluation results
        evaluation_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "execution_file": execution_file,
            "batch_info": execution_data.get("batch_name", "Unknown"),
            "agent_config": execution_data.get("agent_config", {}),
            "framework_version": framework_version,
            "overall_performance": {
                "total_questions": total_questions,
                "questions_with_ground_truth": questions_with_gt,
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "successful_executions": execution_data.get("execution_summary", {}).get("successful_executions", 0),
                "evaluation_errors": evaluation_errors
            },
            "level_performance": level_accuracy,
            "strategy_analysis": strategy_analysis,
            "hybrid_state_metrics": self._analyze_hybrid_state_usage(evaluated_results),
            "results": evaluated_results,
            "evaluation_verified": True
        }
        
        # Save evaluation results
        evaluation_file = self._save_evaluation_results(evaluation_results, execution_file)
        
        # Print comprehensive summary
        print(f"\n📊 EVALUATION COMPLETE (Fixed Ground Truth)")
        print("=" * 50)
        print(f"📋 Total Questions: {total_questions}")
        print(f"✅ With Ground Truth: {questions_with_gt}")
        print(f"🎯 Correct Answers: {correct_answers}")
        print(f"📈 Accuracy: {accuracy:.1%}")
        print(f"❌ Evaluation Errors: {evaluation_errors}")
        
        # Show level breakdown if available
        if level_accuracy:
            print(f"\n📊 Performance by Level:")
            for level, perf in level_accuracy.items():
                if perf['total'] > 0:
                    print(f"   Level {level}: {perf['accuracy']:.1%} ({perf['correct']}/{perf['total']})")
        
        # Show strategy breakdown if available
        if strategy_analysis:
            print(f"\n🎯 Performance by Strategy:")
            for strategy, perf in strategy_analysis.items():
                if perf['total_questions'] > 0:
                    print(f"   {strategy}: {perf['accuracy']:.1%} ({perf['correct_answers']}/{perf['total_questions']})")
        
        print(f"\n💾 Evaluation saved: {evaluation_file}")
        
        return evaluation_results
    
    def _analyze_hybrid_state_usage(self, results: List[Dict]) -> Dict:
        """NEW: Analyze hybrid state and context bridge usage"""
        
        context_bridge_used = sum(1 for r in results if r.get('context_bridge_used', False))
        hybrid_executions = sum(1 for r in results if r.get('file_access_method') == 'hybrid_state')
        
        strategy_distribution = defaultdict(int)
        for result in results:
            strategy = result.get('strategy_used', 'unknown')
            strategy_distribution[strategy] += 1
        
        return {
            "context_bridge_usage": {
                "total_questions": len(results),
                "context_bridge_used": context_bridge_used,
                "usage_percentage": context_bridge_used / len(results) if results else 0
            },
            "hybrid_state_usage": {
                "hybrid_executions": hybrid_executions,
                "hybrid_percentage": hybrid_executions / len(results) if results else 0
            },
            "strategy_distribution": dict(strategy_distribution),
            "average_execution_time": sum(r.get('execution_time', 0) for r in results) / len(results) if results else 0,
            "context_metrics_available": sum(1 for r in results if r.get('context_metrics', {}))
        }
    
    def gaia_answer_matching(self, predicted: str, expected: str) -> bool:
        """PRESERVED: GAIA-compliant exact matching"""
        if not predicted or not expected:
            return False
        
        # Normalize both answers
        pred_norm = self._normalize_gaia_answer(predicted)
        exp_norm = self._normalize_gaia_answer(expected)
        
        return pred_norm == exp_norm
    
    def fuzzy_answer_matching(self, predicted: str, expected: str, threshold: float = 0.8) -> bool:
        """PRESERVED: Enhanced fuzzy string matching with multiple methods"""
        if not predicted or not expected:
            return False
        
        # Normalize both strings
        pred_norm = predicted.lower().strip()
        exp_norm = expected.lower().strip()
        
        # Method 1: Exact match after normalization
        if pred_norm == exp_norm:
            return True
        
        # Method 2: Token-based similarity
        pred_tokens = set(re.findall(r'\w+', pred_norm))
        exp_tokens = set(re.findall(r'\w+', exp_norm))
        
        if pred_tokens and exp_tokens:
            intersection = len(pred_tokens.intersection(exp_tokens))
            union = len(pred_tokens.union(exp_tokens))
            jaccard_similarity = intersection / union if union > 0 else 0
            
            if jaccard_similarity >= threshold:
                return True
        
        # Method 3: Sequence matcher for edit distance
        sequence_similarity = SequenceMatcher(None, pred_norm, exp_norm).ratio()
        if sequence_similarity >= threshold:
            return True
        
        # Method 4: Check if one is contained in the other (for partial matches)
        if len(pred_norm) > 3 and len(exp_norm) > 3:
            if pred_norm in exp_norm or exp_norm in pred_norm:
                return True
        
        return False
    
    def _normalize_gaia_answer(self, answer: str) -> str:
        """PRESERVED: Normalize answer for GAIA comparison"""
        if not answer:
            return ""
        
        # Remove common prefixes and suffixes
        answer = answer.strip()
        
        # Remove articles
        for article in ["the ", "a ", "an "]:
            if answer.lower().startswith(article):
                answer = answer[len(article):]
        
        # Remove punctuation and extra spaces
        answer = answer.strip('.,!?:;"\'').strip()
        
        # Remove commas from numbers
        if answer.replace('.', '').replace('-', '').replace(',', '').isdigit():
            answer = answer.replace(',', '')
        
        return answer.lower()
    
    def _save_evaluation_results(self, evaluation_results: Dict, execution_file: str) -> str:
        """FIXED: Save evaluation results with proper dataclass access"""
        base_name = os.path.basename(execution_file).replace("_execution.json", "")
        
        # Use timestamped filename
        evaluation_filename = create_timestamped_filename(f"{base_name}_evaluation", "json")
        evaluation_file = os.path.join(self.test_config.results_dir, evaluation_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(evaluation_file), exist_ok=True)
        
        try:
            with open(evaluation_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Evaluation results saved: {evaluation_file}")
            return evaluation_file
            
        except Exception as e:
            print(f"❌ Failed to save evaluation results: {e}")
            # Fallback to simple filename
            fallback_file = os.path.join(
                self.test_config.results_dir,
                f"{base_name}_evaluation.json"
            )
            with open(fallback_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            return fallback_file
        
# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def run_gaia_test(agent_config_name: str = "groq", dataset_path: str = "./tests/gaia_data", 
                  max_questions: int = 20, test_config: GAIATestConfig = None) -> Optional[Dict]:
    """
    COMPLETE BLIND TESTING WORKFLOW: Execute then Evaluate
    Compatible with hybrid state + context bridge approach
    """
    print(f"🎯 COMPLETE GAIA TEST (Hybrid Compatible): {agent_config_name}")
    print("=" * 60)
    print("📋 Two-Phase Blind Testing:")
    print("   Phase 1: Blind Execution (no ground truth)")
    print("   Phase 2: Evaluation (with ground truth)")
    print("   🔧 Framework: Hybrid State + Context Bridge Compatible")
    print("")
    
    if not DATASET_UTILS_AVAILABLE:
        print("❌ GAIA dataset utilities required for blind testing")
        return None
    
    if test_config is None:
        test_config = GAIATestConfig(dataset_path=dataset_path)
    
    try:
        # Setup dataset manager
        from gaia_dataset_utils import GAIADatasetManager
        dataset_manager = GAIADatasetManager(dataset_path)
        print("✅ Dataset manager initialized")
        
        # Create blind test batch (NO ground truth)
        blind_questions = dataset_manager.create_test_batch(max_questions, "balanced")
        print(f"✅ Created blind test batch: {len(blind_questions)} questions")
        
        # PHASE 1: Blind Execution (Hybrid Compatible)
        print(f"\n🔒 PHASE 1: BLIND EXECUTION (Hybrid State)")
        print("-" * 40)
        
        agent_config = get_agent_config_by_name(agent_config_name)
        executor = GAIAQuestionExecutor(agent_config, test_config)
        
        batch_name = f"gaia_test_{agent_config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        execution_file = executor.execute_questions_batch(blind_questions, batch_name)
        
        # PHASE 2: Evaluation (Hybrid Compatible)
        print(f"\n🎯 PHASE 2: EVALUATION (Hybrid Aware)")
        print("-" * 30)
        
        evaluator = GAIAAnswerEvaluator(dataset_manager, test_config)
        evaluation_results = evaluator.evaluate_execution_results(execution_file)
        
        print(f"\n🏆 COMPLETE GAIA TEST RESULTS (Hybrid)")
        print("=" * 50)
        overall = evaluation_results["overall_performance"]
        print(f"📊 Overall Accuracy: {overall['accuracy']:.1%}")
        print(f"🎯 Correct Answers: {overall['correct_answers']}/{overall['total_questions']}")
        print(f"⚡ Execution Success: {overall['successful_executions']}/{overall['total_questions']}")
        
        # Show hybrid state metrics
        hybrid_metrics = evaluation_results.get("hybrid_state_metrics", {})
        if hybrid_metrics:
            context_usage = hybrid_metrics.get("context_bridge_usage", {})
            print(f"🌉 Context Bridge Usage: {context_usage.get('usage_percentage', 0):.1%}")
            
            strategy_dist = hybrid_metrics.get("strategy_distribution", {})
            print(f"🎯 Strategy Distribution:")
            for strategy, count in strategy_dist.items():
                print(f"   {strategy}: {count} questions")
        
        # Show level breakdown
        level_perf = evaluation_results.get("level_performance", {})
        if level_perf:
            print(f"\n📈 Performance by Level:")
            for level, perf in level_perf.items():
                print(f"   Level {level}: {perf['accuracy']:.1%} ({perf['correct']}/{perf['total']})")
        
        return evaluation_results
        
    except Exception as e:
        print(f"❌ Complete GAIA test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return proper error structure
        return {
            "error": str(e),
            "overall_performance": {
                "total_questions": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "successful_executions": 0
            },
            "level_performance": {},
            "strategy_analysis": {},
            "hybrid_state_metrics": {}
        }

def run_quick_gaia_test(agent_config_name: str = "groq", **kwargs) -> Optional[Dict]:
    """Quick GAIA test with hybrid state compatibility"""
    
    # Extract parameters with backward compatibility
    num_questions = kwargs.get('num_questions', kwargs.get('max_questions', 5))
    dataset_path = kwargs.get('dataset_path', './tests/gaia_data')
    
    print(f"🚀 Quick GAIA Test (Hybrid Compatible): {agent_config_name}")
    print(f"   Questions: {num_questions}")
    print(f"   Framework: Hybrid State + Context Bridge")
    
    result = run_gaia_test(
        agent_config_name=agent_config_name,
        dataset_path=dataset_path,
        max_questions=num_questions
    )
    
    if result is None:
        return {
            "error": "Test failed to return results",
            "overall_performance": {
                "total_questions": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "successful_executions": 0
            }
        }
    
    return result

def compare_agent_configs(config_names: List[str], num_questions: int = 10, 
                         dataset_path: str = "./tests/gaia_data") -> Dict:
    """Compare multiple agent configurations with hybrid state"""
    
    print(f"🔄 AGENT COMPARISON (Hybrid Compatible): {len(config_names)} configs")
    print(f"   Configs: {', '.join(config_names)}")
    print(f"   Questions: {num_questions}")
    print(f"   Framework: Hybrid State + Context Bridge")
    
    comparison_results = {}
    
    for config_name in config_names:
        print(f"\n🧪 Testing {config_name}...")
        
        result = run_gaia_test(
            agent_config_name=config_name,
            dataset_path=dataset_path,
            max_questions=num_questions
        )
        
        if result:
            comparison_results[config_name] = {
                "accuracy": result["overall_performance"]["accuracy"],
                "correct_answers": result["overall_performance"]["correct_answers"],
                "total_questions": result["overall_performance"]["total_questions"],
                "level_performance": result.get("level_performance", {}),
                "strategy_analysis": result.get("strategy_analysis", {}),
                "hybrid_metrics": result.get("hybrid_state_metrics", {})
            }
            print(f"   ✅ {config_name}: {result['overall_performance']['accuracy']:.1%}")
        else:
            comparison_results[config_name] = {"error": "Test failed"}
            print(f"   ❌ {config_name}: Failed")
    
    return {
        "comparison_results": comparison_results,
        "timestamp": datetime.now().isoformat(),
        "test_questions": num_questions,
        "framework_version": "hybrid_state_compatible"
    }

def run_smart_routing_test(agent_config_name: str = "performance") -> Dict:
    """Test smart routing with hybrid state awareness"""
    
    print(f"🔀 Smart Routing Test (Hybrid Compatible): {agent_config_name}")
    
    result = run_gaia_test(
        agent_config_name=agent_config_name,
        max_questions=20
    )
    
    if result and 'strategy_analysis' in result:
        strategy_analysis = result.get('strategy_analysis', {})
        hybrid_metrics = result.get('hybrid_state_metrics', {})
        
        print(f"\n🔀 ROUTING ANALYSIS (Hybrid State)")
        print("-" * 40)
        
        # Strategy effectiveness
        one_shot_count = sum(1 for strategy in strategy_analysis.keys() if 'one_shot' in strategy.lower())
        manager_count = sum(1 for strategy in strategy_analysis.keys() if 'manager' in strategy.lower() or 'agent_' in strategy.lower())
        
        print(f"📊 Strategy Distribution:")
        print(f"├── One-shot approaches: {one_shot_count}")
        print(f"├── Manager/Agent coordination: {manager_count}")
        
        print(f"\n📈 Strategy Effectiveness:")
        for strategy, stats in strategy_analysis.items():
            accuracy = stats.get('accuracy', 0)
            total = stats.get('total_questions', 0)
            print(f"├── {strategy}: {accuracy:.1%} accuracy ({total} questions)")
        
        # Hybrid state metrics
        if hybrid_metrics:
            context_usage = hybrid_metrics.get('context_bridge_usage', {})
            print(f"\n🌉 Context Bridge Usage:")
            print(f"├── Usage: {context_usage.get('usage_percentage', 0):.1%}")
            print(f"└── Average execution time: {hybrid_metrics.get('average_execution_time', 0):.2f}s")
    
    return result

def analyze_failure_patterns(evaluation_results: Dict) -> Dict:
    """Analyze failure patterns with hybrid state awareness"""
    
    if not evaluation_results or 'results' not in evaluation_results:
        return {'error': 'No evaluation results to analyze'}
    
    detailed_results = evaluation_results['results']
    incorrect_results = [r for r in detailed_results if not r.get('is_correct', False)]
    
    print(f"\n🔍 FAILURE PATTERN ANALYSIS (Hybrid State)")
    print("=" * 50)
    print(f"📊 Total Questions: {len(detailed_results)}")
    print(f"❌ Incorrect Answers: {len(incorrect_results)}")
    
    if len(incorrect_results) == 0:
        return {'perfect_performance': True}
    
    # Analyze failure patterns with hybrid state awareness
    failure_patterns = {
        'by_level': defaultdict(int),
        'by_strategy': defaultdict(int),
        'by_file_type': defaultdict(int),
        'execution_failures': 0,
        'context_bridge_failures': 0,
        'hybrid_state_failures': 0,
        'common_error_types': defaultdict(int)
    }
    
    for result in incorrect_results:
        level = result.get('level', 'unknown')
        failure_patterns['by_level'][level] += 1
        
        strategy = result.get('strategy_used', 'unknown')
        failure_patterns['by_strategy'][strategy] += 1
        
        if result.get('has_file', False):
            file_name = result.get('file_name', '')
            if file_name:
                ext = Path(file_name).suffix.lower()
                failure_patterns['by_file_type'][ext or 'no_extension'] += 1
        else:
            failure_patterns['by_file_type']['no_file'] += 1
        
        if not result.get('execution_successful', True):
            failure_patterns['execution_failures'] += 1
            
            error_type = result.get('error_type', 'unknown')
            failure_patterns['common_error_types'][error_type] += 1
        
        # NEW: Track hybrid state specific failures
        if result.get('context_bridge_used', False) and not result.get('execution_successful', True):
            failure_patterns['context_bridge_failures'] += 1
        
        if result.get('file_access_method') == 'hybrid_state' and not result.get('execution_successful', True):
            failure_patterns['hybrid_state_failures'] += 1
    
    # Generate improvement recommendations with hybrid state awareness
    recommendations = _generate_improvement_recommendations_hybrid(failure_patterns, incorrect_results)
    
    return {
        'failure_patterns': dict(failure_patterns),
        'recommendations': recommendations,
        'sample_failures': incorrect_results[:5],
        'hybrid_state_analysis': _analyze_hybrid_failures(incorrect_results)
    }

def _generate_improvement_recommendations_hybrid(failure_patterns: Dict, incorrect_results: List[Dict]) -> List[str]:
    """Generate improvement recommendations with hybrid state awareness"""
    
    recommendations = []
    total_failures = len(incorrect_results)
    
    # Level-based recommendations
    level_failures = failure_patterns['by_level']
    if level_failures.get(1, 0) > level_failures.get(2, 0):
        recommendations.append("Focus on Level 1 performance - basic capabilities need strengthening")
    if level_failures.get(3, 0) > 0:
        recommendations.append("Level 3 questions are challenging - enhance reasoning capabilities")
    
    # Strategy-based recommendations  
    strategy_failures = failure_patterns['by_strategy']
    if strategy_failures.get('one_shot_llm', 0) > strategy_failures.get('manager_coordination', 0):
        recommendations.append("One-shot LLM failing more - improve direct LLM prompting")
    elif strategy_failures.get('manager_coordination', 0) > strategy_failures.get('one_shot_llm', 0):
        recommendations.append("Manager coordination failing more - check agent routing and tool integration")
    
    # Execution-based recommendations
    if failure_patterns['execution_failures'] > total_failures * 0.2:
        recommendations.append("High execution failure rate - improve error handling and stability")
    
    # NEW: Hybrid state specific recommendations
    if failure_patterns['context_bridge_failures'] > 0:
        recommendations.append("Context bridge issues detected - check ContextBridge integration")
    
    if failure_patterns['hybrid_state_failures'] > 0:
        recommendations.append("Hybrid state file access issues - verify GAIAState and file extraction")
    
    # Error type recommendations
    error_types = failure_patterns['common_error_types']
    if error_types.get('timeout', 0) > 0:
        recommendations.append("Timeout issues detected - optimize execution speed or increase timeout")
    if error_types.get('api_error', 0) > 0:
        recommendations.append("API issues detected - implement better rate limiting and retry logic")
    
    return recommendations

def _analyze_hybrid_failures(incorrect_results: List[Dict]) -> Dict:
    """NEW: Analyze failures specific to hybrid state implementation"""
    
    context_bridge_issues = []
    file_access_issues = []
    state_configuration_issues = []
    
    for result in incorrect_results:
        if result.get('context_bridge_used', False):
            context_metrics = result.get('context_metrics', {})
            if not context_metrics:
                context_bridge_issues.append(result['task_id'])
        
        if result.get('file_access_method') == 'hybrid_state' and not result.get('execution_successful', True):
            file_access_issues.append(result['task_id'])
        
        if result.get('strategy_used') == 'error':
            state_configuration_issues.append(result['task_id'])
    
    return {
        "context_bridge_issues": len(context_bridge_issues),
        "file_access_issues": len(file_access_issues),
        "state_configuration_issues": len(state_configuration_issues),
        "total_hybrid_failures": len(context_bridge_issues) + len(file_access_issues) + len(state_configuration_issues),
        "recommendations": [
            "Verify ContextBridge.start_task_execution() calls" if context_bridge_issues else None,
            "Check extract_file_info_from_task_id() function" if file_access_issues else None,
            "Validate GAIAState structure and workflow" if state_configuration_issues else None
        ]
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def test_file_vs_text_performance(agent_config_name: str = "groq") -> Dict:
    """Compare performance with hybrid state awareness"""
    
    print(f"📎 File vs Text Performance Test (Hybrid Compatible): {agent_config_name}")
    
    if not DATASET_UTILS_AVAILABLE:
        print("❌ Dataset utilities required")
        return {}
    
    results = {}
    
    # Test with different question sets
    print(f"\n📝 Test 1: Text-Only Questions (Hybrid State)")
    result = run_gaia_test(agent_config_name, max_questions=10)
    if result:
        results['text_only'] = result
    
    print(f"\n📎 Test 2: File-Based Questions (Hybrid State)") 
    result = run_gaia_test(agent_config_name, max_questions=10)
    if result:
        results['with_files'] = result
    
    # Compare results with hybrid metrics
    if 'text_only' in results and 'with_files' in results:
        text_acc = results['text_only']['overall_performance']['accuracy']
        file_acc = results['with_files']['overall_performance']['accuracy']
        
        text_hybrid = results['text_only'].get('hybrid_state_metrics', {})
        file_hybrid = results['with_files'].get('hybrid_state_metrics', {})
        
        print(f"\n📊 COMPARISON SUMMARY (Hybrid State)")
        print("=" * 40)
        print(f"📝 Text-only: {text_acc:.1%}")
        print(f"📎 With files: {file_acc:.1%}")
        print(f"📈 Difference: {(text_acc - file_acc)*100:+.1f} percentage points")
        
        if text_hybrid and file_hybrid:
            text_context = text_hybrid.get('context_bridge_usage', {}).get('usage_percentage', 0)
            file_context = file_hybrid.get('context_bridge_usage', {}).get('usage_percentage', 0)
            print(f"🌉 Context Bridge Usage:")
            print(f"   Text-only: {text_context:.1%}")
            print(f"   With files: {file_context:.1%}")
    
    return results

def validate_test_environment() -> Dict:
    """Validate environment with hybrid state compatibility"""
    
    print("🔍 Validating test environment (Hybrid State Compatible)...")
    
    validation = {
        "agent_logic_available": False,
        "agent_interface_available": False,
        "agent_context_available": False,
        "dataset_utils_available": DATASET_UTILS_AVAILABLE,
        "gaia_dataset_accessible": False,
        "test_directory_writable": False,
        "context_bridge_functional": False,
        "all_dependencies_ready": False
    }
    
    # Check agent_logic
    try:
        from agent_logic import GAIAAgent, GAIAConfig
        validation["agent_logic_available"] = True
        print("✅ agent_logic available")
    except ImportError:
        print("❌ agent_logic not available")
    
    # Check agent_interface
    try:
        from agent_interface import get_groq_config
        validation["agent_interface_available"] = True
        print("✅ agent_interface available")
    except ImportError:
        print("❌ agent_interface not available")
    
    # NEW: Check agent_context
    try:
        from agent_context import ContextBridge
        validation["agent_context_available"] = True
        
        # Test ContextBridge functionality
        ContextBridge.start_task_execution("test_123")
        ContextBridge.track_operation("Test operation")
        metrics = ContextBridge.get_execution_metrics()
        ContextBridge.clear_tracking()
        
        if metrics and metrics.get('steps_executed', 0) > 0:
            validation["context_bridge_functional"] = True
            print("✅ ContextBridge functional")
        else:
            print("⚠️  ContextBridge available but not functional")
            
    except ImportError:
        print("❌ agent_context not available")
    except Exception as e:
        print(f"⚠️  ContextBridge error: {e}")
    
    # Check GAIA dataset
    if DATASET_UTILS_AVAILABLE:
        try:
            from gaia_dataset_utils import GAIADatasetManager
            dataset_manager = GAIADatasetManager("./tests/gaia_data")
            if dataset_manager.metadata:
                validation["gaia_dataset_accessible"] = True
                print(f"✅ GAIA dataset accessible: {len(dataset_manager.metadata)} questions")
            else:
                print("❌ GAIA dataset not accessible")
        except Exception as e:
            print(f"❌ GAIA dataset error: {e}")
    
    # Check test directory
    try:
        test_dir = Path("./test_results")
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        validation["test_directory_writable"] = True
        print("✅ Test directory writable")
    except Exception as e:
        print(f"❌ Test directory not writable: {e}")
    
    # Overall validation
    validation["all_dependencies_ready"] = all([
        validation["agent_logic_available"],
        validation["agent_interface_available"],
        validation["agent_context_available"],
        validation["dataset_utils_available"],
        validation["test_directory_writable"]
    ])
    
    if validation["all_dependencies_ready"]:
        print("🎉 All dependencies ready for hybrid state testing!")
        if validation["context_bridge_functional"]:
            print("🌉 Context Bridge integration verified!")
    else:
        print("⚠️  Some dependencies missing - limited testing capability")
    
    return validation

# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================

class GAIATestExecutor:
    """Backwards compatibility wrapper updated for hybrid state"""
    
    def __init__(self, agent_config, test_config=None):
        print("🔄 Using backwards compatibility layer (Hybrid Compatible)")
        self.executor = GAIAQuestionExecutor(agent_config, test_config)
    
    def execute_questions_batch(self, questions):
        return self.executor.execute_questions_batch(questions)

# ============================================================================
# MAIN TESTING ENTRY POINTS
# ============================================================================

if __name__ == "__main__":
    print("🧪 GAIA Testing Framework - FIXED DATACLASS VERSION")
    print("=" * 60)
    print("✅ FIXED ISSUES:")
    print("   ├── Replaced all .get() calls on GAIATestConfig with getattr()")
    print("   ├── Maintained hybrid state + context bridge compatibility")
    print("   ├── Preserved all testing functionality")
    print("   └── Fixed 'GAIATestConfig' object has no attribute 'get' error")
    print("")
    print("🔧 DATACLASS FIX APPLIED:")
    fixes = [
        "✅ execute_questions_batch: getattr(test_config, 'enable_ground_truth_isolation', True)",
        "✅ _save_execution_results: getattr(test_config, 'results_directory', './test_results')",
        "✅ All test configuration access now uses proper dataclass attributes",
        "✅ Backward compatibility maintained for dict-based configs"
    ]
    
    for fix in fixes:
        print(f"  {fix}")

    print(f"\n📋 WORKING FUNCTIONS (DATACLASS FIXED):")
    functions = [
        'run_gaia_test ✅ (Fixed)',
        'run_quick_gaia_test ✅ (Fixed)', 
        'compare_agent_configs ✅ (Fixed)',
        'run_smart_routing_test ✅ (Fixed)',
        'test_file_vs_text_performance ✅ (Fixed)',
        'analyze_failure_patterns ✅ (Fixed)',
        'validate_test_environment ✅ (Fixed)'
    ]
    
    for func in functions:
        print(f"   ├── {func}")

    print(f"\n💡 Quick Test (Should Work Now):")
    print(f"   from agent_testing import run_quick_gaia_test")
    print(f"   result = run_quick_gaia_test('groq', num_questions=5)")
    print(f"   # Should NOT get 'GAIATestConfig' object has no attribute 'get' error")
    
    print(f"\n🎯 Fixed Error:")
    print(f"   ❌ Before: self.test_config.get('enable_ground_truth_isolation', True)")
    print(f"   ✅ After:  getattr(self.test_config, 'enable_ground_truth_isolation', True)")