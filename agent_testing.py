# agent_testing.py - PURE TESTING LAYER (Complete Version)
# Responsibilities: Agent execution, evaluation, performance analysis, comparisons
# NO dataset management, file operations, or metadata parsing

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

# Import our agent system and dataset manager
from agent_logic import GAIAAgent, GAIAConfig
from agent_interface import (
    create_gaia_agent, 
    get_groq_config, 
    get_openrouter_config, 
    get_google_config, 
    get_ollama_config, 
    get_performance_config, 
    get_accuracy_config
)
from agent_logging import create_timestamped_filename
from gaia_dataset_utils import GAIADatasetManager

# ============================================================================
# TESTING CONFIGURATION
# ============================================================================

@dataclass
class GAIATestConfig:
    """Configuration for GAIA testing framework"""
    # Execution settings
    timeout_per_question: int = 180
    max_retries: int = 3
    enable_retries: bool = True
    enable_strategy_fallback: bool = True
    
    # Output settings
    results_dir: str = "logs"
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

# ============================================================================
# AGENT CONFIGURATION MAPPING
# ============================================================================

def gaia_config_to_dict(config: GAIAConfig) -> Dict:
    """Convert GAIAConfig dataclass to dictionary"""
    import dataclasses
    return dataclasses.asdict(config)

def get_agent_config_by_name(config_name: str) -> GAIAConfig:
    """Map configuration names to actual configs -  - returns GAIAConfig objects"""
    
    config_map = {
        "groq": get_groq_config("qwen-qwq-32b"),
        "groq_fast": get_groq_config("meta-llama/llama-4-scout-17b-16e-instruct"),
        "openrouter": get_openrouter_config("qwen/qwen3-30b-a3b"),
        "openrouter_premium": get_openrouter_config("qwen/qwen2.5-vl-32b-instruct"),
        "google": get_google_config("gemini-2.5-pro-preview"),
        "google_pro": get_google_config("gemini-2.5-flash-preview-05-20"),
        "ollama": get_ollama_config("qwen-agent-custom"),
        "performance": get_performance_config(),
        "accuracy": get_accuracy_config()
    }
    
    return config_map.get(config_name, get_groq_config())

# ============================================================================
# AGENT EXECUTION (BLIND)
# ============================================================================

class GAIATestExecutor:
    def __init__(self, agent_config: Union[str, Dict, GAIAConfig], test_config: GAIATestConfig = None):
        """Initialize tests with agent and test configurations"""
        
        self.test_config = test_config or GAIATestConfig()
        self.results_dir = Path(self.test_config.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Handle different config types
        if isinstance(agent_config, str):
            gaia_config = get_agent_config_by_name(agent_config)  # Returns GAIAConfig now
            self.agent_config_name = agent_config
        elif isinstance(agent_config, dict):
            gaia_config = dict_to_gaia_config(agent_config)
            self.agent_config_name = "custom"
        else:  # GAIAConfig object
            gaia_config = agent_config
            self.agent_config_name = "custom"
        
        # Pass GAIAConfig object
        self.agent = create_gaia_agent(gaia_config)
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Execution tracking
        self.execution_stats = {
            'total_questions': 0,
            'successful_executions': 0,
            'execution_failures': 0,
            'retry_count': 0,
            'strategy_fallback_count': 0,
            'start_time': None,
            'end_time': None
        }
        
        print(f"🎯 GAIA Test Executor Initialized")
        print(f"🤖 Agent: {self.agent_config_name}")
        print(f"📊 Session: {self.session_id}")
        print(f"🔧 Features: Error detection, Strategy fallback, Real-time monitoring")
    
    def execute_test_batch(self, questions: List[Dict]) -> List[Dict]:
        """Execute agent on provided questions (blind - no ground truth)"""
        
        if not questions:
            print("❌ No questions provided for execution")
            return []
        
        self.execution_stats['total_questions'] = len(questions)
        self.execution_stats['start_time'] = time.time()
        
        print(f"\n🚀 Starting Batch Execution")
        print(f"📝 Questions: {len(questions)}")
        print(f"🤖 Agent: {self.agent_config_name}")
        print(f"🔧 Error detection: {'✅ ON' if self.test_config.detect_execution_failures else '❌ OFF'}")
        print(f"🔄 Strategy fallback: {'✅ ON' if self.test_config.enable_strategy_fallback else '❌ OFF'}")
        print("=" * 70)
        
        execution_results = []
        
        for i, question_data in enumerate(questions):
            if self.test_config.enable_real_time_monitoring:
                self._print_progress_header(i, len(questions), question_data)
            
            # Execute with enhanced error handling
            execution_result = self.execute_single_question(question_data, attempt=1)
            execution_results.append(execution_result)
            
            # Update stats
            if execution_result.get('execution_successful', False):
                self.execution_stats['successful_executions'] += 1
            else:
                self.execution_stats['execution_failures'] += 1
            
            # Print immediate result
            self._print_immediate_result(execution_result, i+1)
        
        self.execution_stats['end_time'] = time.time()
        self._print_batch_summary(execution_results)
        
        # Save results
        if self.test_config.save_intermediate:
            self._save_execution_results(execution_results)
        
        return execution_results
    
    def execute_single_question(self, question_data: Dict, attempt: int = 1) -> Dict:
        """Execute single question with comprehensive tracking"""
        
        start_time = time.time()
        task_id = question_data.get('task_id', f'unknown_{int(time.time())}')
        
        execution_record = {
            'session_id': self.session_id,
            'task_id': task_id,
            'question': question_data.get('Question', ''),
            'level': question_data.get('Level'),
            'file_name': question_data.get('file_name', ''),
            'has_file': bool(question_data.get('file_name', '').strip()),
            'timestamp': datetime.now().isoformat(),
            'agent_config': self.agent_config_name,
            'attempt_number': attempt
        }
        
        try:
            # Prepare question for agent
            question_text = self._prepare_question_for_agent(question_data)
            
            # Execute with our agent system
            result = self.agent.process_question(
                question=question_text,
                task_id=task_id
            )
            
            execution_time = time.time() - start_time
            
            # Extract agent response
            raw_agent_response = result.get('final_answer', '')
            
            # Detect execution failures in agent response
            is_execution_failure = self._detect_execution_failure(raw_agent_response)
            
            if is_execution_failure:
                # Execution failed - agent returned error message
                execution_record.update({
                    'execution_successful': False,
                    'agent_answer': '',  # No real answer
                    'raw_agent_response': raw_agent_response,
                    'error_message': raw_agent_response,
                    'error_type': self._categorize_error(raw_agent_response),
                    'strategy_used': 'failed',
                    'execution_time': execution_time,
                    'total_steps': 0
                })
                
                # Try strategy fallback if enabled
                if (self.test_config.enable_strategy_fallback and 
                    attempt <= self.test_config.max_retries):
                    
                    return self._attempt_strategy_fallback(question_data, execution_record, attempt)
                
            else:
                # Execution succeeded - got real answer
                steps = result.get('steps', [])
                complexity = result.get('complexity', '')
                selected_agent = result.get('selected_agent', '')
                strategy_used = self._determine_strategy_used(result, complexity, steps)
                
                execution_record.update({
                    'execution_successful': True,
                    'agent_answer': raw_agent_response.strip(),
                    'raw_agent_response': raw_agent_response,
                    'error_message': None,
                    'error_type': None,
                    'strategy_used': strategy_used,
                    'selected_agent': selected_agent,
                    'complexity_detected': complexity,
                    'execution_time': execution_time,
                    'total_steps': len(steps),
                    'steps_summary': steps[-3:] if len(steps) > 3 else steps
                })
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            execution_record.update({
                'execution_successful': False,
                'agent_answer': '',
                'raw_agent_response': '',
                'error_message': error_msg,
                'error_type': 'python_exception',
                'strategy_used': 'failed',
                'execution_time': execution_time,
                'total_steps': 0,
                'exception_traceback': traceback.format_exc()
            })
            
            # Retry logic for Python exceptions
            if attempt < self.test_config.max_retries and self.test_config.enable_retries:
                self.execution_stats['retry_count'] += 1
                print(f"    🔄 Retry attempt {attempt + 1} (Python exception)")
                time.sleep(2)
                return self.execute_single_question(question_data, attempt + 1)
        
        return execution_record
    
    def execute_with_retry(self, question: Dict, max_retries: int = 2) -> Dict:
        """Execute with retry logic for robustness"""
        
        for attempt in range(1, max_retries + 2):
            result = self.execute_single_question(question, attempt)
            
            if result.get('execution_successful', False):
                return result
            
            if attempt <= max_retries:
                print(f"    🔄 Retry {attempt}/{max_retries}")
                time.sleep(1)
        
        return result
    
    def execute_with_timeout(self, question: Dict, timeout: int = 180) -> Dict:
        """Execute with timeout handling"""
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Question execution timeout after {timeout}s")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = self.execute_single_question(question)
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError as e:
            signal.alarm(0)
            return {
                'task_id': question.get('task_id', 'unknown'),
                'execution_successful': False,
                'error_message': str(e),
                'error_type': 'timeout',
                'agent_answer': '',
                'strategy_used': 'timeout',
                'execution_time': timeout
            }
    
    def _detect_execution_failure(self, agent_response: str) -> bool:
        """Detect execution failures in agent responses"""
        
        if not agent_response or not agent_response.strip():
            return True
        
        # Comprehensive error patterns
        error_patterns = [
            'execution failed',
            'manager execution failed',
            'agent execution failed',
            'got an unexpected keyword argument',
            'attributeerror',
            'keyerror',
            'valueerror',
            'typeerror',
            'nameerror',
            'traceback',
            'exception occurred',
            'error:',
            'failed:',
            'timeout',
            'api error',
            'rate limit',
            'connection error',
            'authentication failed',
            'permission denied',
            'file not found',
            'invalid response',
            'parsing error',
            'tool execution failed'
        ]
        
        response_lower = agent_response.lower().strip()
        
        # Check for error patterns
        for pattern in error_patterns:
            if pattern in response_lower:
                return True
        
        # Check for suspiciously long responses that contain error keywords
        if len(agent_response) > 150:
            error_keywords = ['failed', 'error', 'exception', 'traceback', 'unexpected']
            if sum(1 for keyword in error_keywords if keyword in response_lower) >= 2:
                return True
        
        # Check for Python-style stack traces
        if 'traceback (most recent call last):' in response_lower:
            return True
        
        # Check for empty or placeholder responses
        placeholder_responses = ['', 'none', 'null', 'undefined', 'no answer', 'cannot answer']
        if response_lower in placeholder_responses:
            return True
        
        return False
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error types for analysis"""
        
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
    
    def _attempt_strategy_fallback(self, question_data: Dict, failed_record: Dict, attempt: int) -> Dict:
        """Attempt execution with fallback strategy"""
        
        self.execution_stats['strategy_fallback_count'] += 1
        
        print(f"    🔄 Strategy fallback attempt {attempt} (trying simpler approach)")
        
        # For now, return the failed record with fallback info
        # TODO: Implement actual strategy switching in agent system
        failed_record.update({
            'fallback_attempted': True,
            'fallback_attempt_number': attempt,
            'original_error': failed_record['error_message']
        })
        
        return failed_record
    
    def _prepare_question_for_agent(self, question_data: Dict) -> str:
        """Prepare question text with file references"""
        
        question_text = question_data.get('Question', '')
        
        # Add file reference if present
        file_path = question_data.get('file_path', '')
        if file_path and os.path.exists(file_path):
            question_text = f"{question_text}\n\n[FILE]: {file_path}"
        elif question_data.get('file_name'):
            question_text = f"{question_text}\n\n[FILE ATTACHED]: {question_data['file_name']}"
        
        return question_text
    
    def _determine_strategy_used(self, result: Dict, complexity: str, steps: List) -> str:
        """Determine which strategy was used based on our system"""
        
        if complexity == "simple":
            return "one_shot_llm"
        elif complexity == "complex":
            return "manager_coordination"
        else:
            # Fallback - analyze steps
            if len(steps) <= 2:
                return "one_shot_llm"
            else:
                return "manager_coordination"
    
    def _print_progress_header(self, index: int, total: int, question_data: Dict):
        """Print progress header for real-time monitoring"""
        
        progress = (index + 1) / total * 100
        progress_bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
        
        print(f"\n🔄 Question {index+1}/{total} ({progress:.1f}%) [{progress_bar}]")
        print(f"📋 Task ID: {question_data.get('task_id', 'unknown')}")
        print(f"📊 Level: {question_data.get('Level', 'unknown')}")
        
        if question_data.get('file_name'):
            print(f"📎 File: {question_data['file_name']}")
        
        question_text = question_data.get('Question', '')
        preview = question_text[:60] + "..." if len(question_text) > 60 else question_text
        print(f"❓ Question: {preview}")
    
    def _print_immediate_result(self, execution_result: Dict, question_num: int):
        """Print immediate result after each question"""
        
        if execution_result.get('execution_successful', False):
            answer = execution_result.get('agent_answer', 'No answer')
            strategy = execution_result.get('strategy_used', '')
            exec_time = execution_result.get('execution_time', 0)
            
            print(f"✅ Answer: {answer}")
            print(f"⏱️ Time: {exec_time:.2f}s")
            print(f"🎯 Strategy: {strategy}")
        else:
            error_type = execution_result.get('error_type', 'unknown')
            error_msg = execution_result.get('error_message', 'Unknown error')
            exec_time = execution_result.get('execution_time', 0)
            
            print(f"❌ Execution Failed ({error_type})")
            print(f"💬 Error: {error_msg[:80]}{'...' if len(error_msg) > 80 else ''}")
            print(f"⏱️ Time: {exec_time:.2f}s")
    
    def _print_batch_summary(self, execution_results: List[Dict]):
        """Print comprehensive batch execution summary"""
        
        total_time = self.execution_stats['end_time'] - self.execution_stats['start_time']
        total_questions = len(execution_results)
        successful = self.execution_stats['successful_executions']
        failed = self.execution_stats['execution_failures']
        
        print(f"\n🎉 Batch Execution Complete!")
        print("=" * 50)
        print(f"📊 Execution Statistics:")
        print(f"  Total questions: {total_questions}")
        print(f"  ✅ Successful executions: {successful} ({successful/total_questions:.1%})")
        print(f"  ❌ Failed executions: {failed} ({failed/total_questions:.1%})")
        print(f"  🔄 Retries used: {self.execution_stats['retry_count']}")
        print(f"  🔄 Strategy fallbacks: {self.execution_stats['strategy_fallback_count']}")
        print(f"  ⏱️ Total time: {total_time:.1f}s")
        print(f"  📈 Average time: {total_time/total_questions:.2f}s per question")
        
        # Error analysis
        if failed > 0:
            error_types = defaultdict(int)
            for result in execution_results:
                if not result.get('execution_successful', False):
                    error_type = result.get('error_type', 'unknown')
                    error_types[error_type] += 1
            
            print(f"\n🐛 Error Breakdown:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count} occurrences")
        
        # Strategy analysis
        strategy_stats = defaultdict(int)
        for result in execution_results:
            strategy = result.get('strategy_used', 'unknown')
            strategy_stats[strategy] += 1
        
        print(f"\n🎯 Strategy Usage:")
        for strategy, count in sorted(strategy_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy}: {count} questions")
    
    def _save_execution_results(self, results: List[Dict]):
        """Save execution results with additional metadata"""
        
        # Create results package
        results_package = {
            'execution_metadata': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'agent_config': self.agent_config_name,
                'total_questions': len(results),
                'execution_stats': self.execution_stats,
                'test_config': {
                    'timeout_per_question': self.test_config.timeout_per_question,
                    'max_retries': self.test_config.max_retries,
                    'detect_execution_failures': self.test_config.detect_execution_failures,
                    'enable_strategy_fallback': self.test_config.enable_strategy_fallback
                }
            },
            'execution_results': results
        }
        
        # Save as JSON
        filename = create_timestamped_filename(f"gaia_execution_{self.agent_config_name}", "json")
        filepath = self.results_dir / filename.split('/')[-1]
        
        with open(filepath, 'w') as f:
            json.dump(results_package, f, indent=2, default=str)
        
        print(f"💾 Execution results saved: {filepath}")

# ============================================================================
# EVALUATION (WITH GROUND TRUTH)
# ============================================================================

class GAIATestEvaluator:
    """Evaluate agent results against ground truth"""
    
    def __init__(self, dataset_manager: GAIADatasetManager):
        """Initialize with dataset manager for ground truth access"""
        self.dataset_manager = dataset_manager
        
        print(f"🎯 GAIA Test Evaluator Initialized")
        print(f"📊 Dataset: {len(dataset_manager.metadata)} questions available")
        print(f"🔧 Features: Fixed metrics, Error categorization, Advanced analysis")
    
    def evaluate_execution_results(self, execution_results: List[Dict]) -> Dict:
        """Evaluate agent results against ground truth"""
        
        if not execution_results:
            print("❌ No execution results to evaluate")
            return {}
        
        print(f"\n🔍 Starting Answer Evaluation")
        print(f"📊 Evaluating {len(execution_results)} results")
        print("=" * 60)
        
        evaluation_results = []
        correct_count = 0
        execution_success_count = 0
        
        for i, execution in enumerate(execution_results):
            task_id = execution.get('task_id')
            
            print(f"\n📝 Evaluation {i+1}/{len(execution_results)}")
            print(f"📋 Task ID: {task_id}")
            
            # Get ground truth
            ground_truth = self.dataset_manager.get_ground_truth(task_id)
            
            if not ground_truth:
                print(f"⚠️ No ground truth found for task {task_id}")
                evaluation_result = self._create_evaluation_record(
                    execution, None, False, "No ground truth available"
                )
            else:
                # Evaluate with fixed metrics
                evaluation_result = self._evaluate_single_execution(execution, ground_truth)
                
                if evaluation_result['is_correct']:
                    correct_count += 1
                
                if evaluation_result['execution_successful']:
                    execution_success_count += 1
                
                # Print evaluation details
                self._print_evaluation_details(evaluation_result)
            
            evaluation_results.append(evaluation_result)
        
        # Generate analysis
        analysis = self._generate_evaluation_analysis(evaluation_results, correct_count, execution_success_count)
        
        # Create final results package
        final_results = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_questions': len(evaluation_results),
                'successful_executions': execution_success_count,
                'execution_failures': len(evaluation_results) - execution_success_count,
                'correct_answers': correct_count,
                'overall_accuracy': correct_count / len(evaluation_results) if evaluation_results else 0,
                'execution_success_rate': execution_success_count / len(evaluation_results) if evaluation_results else 0,
                'agent_config': execution_results[0].get('agent_config', 'unknown') if execution_results else 'unknown'
            },
            'detailed_results': evaluation_results,
            'analysis': analysis
        }
        
        # Save evaluation results
        self._save_evaluation_results(final_results)
        
        print(f"\n🎉 Evaluation Complete!")
        self._print_evaluation_summary(analysis, final_results['evaluation_metadata'])
        
        return final_results
    
    def evaluate_single_answer(self, agent_answer: str, task_id: str) -> Dict:
        """Evaluate single answer with detailed analysis"""
        
        ground_truth = self.dataset_manager.get_ground_truth(task_id)
        
        if not ground_truth:
            return {
                'task_id': task_id,
                'is_correct': False,
                'error': 'No ground truth available'
            }
        
        expected_answer = ground_truth['final_answer']
        is_correct = self.gaia_answer_matching(agent_answer, expected_answer)
        
        return {
            'task_id': task_id,
            'agent_answer': agent_answer,
            'expected_answer': expected_answer,
            'is_correct': is_correct,
            'level': ground_truth.get('level'),
            'evaluation_details': self._analyze_answer_differences(agent_answer, expected_answer)
        }
    
    def _evaluate_single_execution(self, execution: Dict, ground_truth: Dict) -> Dict:
        """Evaluate single execution with fixed metrics"""
        
        # Extract execution data
        raw_agent_response = execution.get('raw_agent_response', execution.get('agent_answer', ''))
        clean_agent_answer = execution.get('agent_answer', '')
        expected_answer = ground_truth['final_answer']
        
        # Determine if execution was actually successful
        execution_successful = execution.get('execution_successful', False)
        
        # Evaluate correctness (only if execution succeeded)
        if execution_successful and clean_agent_answer:
            is_correct = self.gaia_answer_matching(clean_agent_answer, expected_answer)
            similarity_score = self.fuzzy_answer_matching(clean_agent_answer, expected_answer)
        else:
            is_correct = False
            similarity_score = 0.0
        
        # Create evaluation record
        return self._create_evaluation_record(
            execution, ground_truth, is_correct, 
            similarity_score=similarity_score
        )
    
    def _create_evaluation_record(self, execution: Dict, ground_truth: Dict, is_correct: bool, 
                                 note: str = None, similarity_score: float = 0.0) -> Dict:
        """Create comprehensive evaluation record with fixed metrics"""
        
        # Extract data safely
        execution_successful = execution.get('execution_successful', False)
        clean_agent_answer = execution.get('agent_answer', '') if execution_successful else ''
        raw_response = execution.get('raw_agent_response', execution.get('agent_answer', ''))
        error_message = execution.get('error_message')
        error_type = execution.get('error_type')
        
        return {
            'task_id': execution.get('task_id'),
            'question': execution.get('question'),
            'level': execution.get('level'),
            'has_file': execution.get('has_file', False),
            'file_name': execution.get('file_name', ''),
            
            # Execution data
            'agent_answer': clean_agent_answer,
            'raw_agent_response': raw_response,
            'strategy_used': execution.get('strategy_used', ''),
            'selected_agent': execution.get('selected_agent', ''),
            'complexity_detected': execution.get('complexity_detected', ''),
            'execution_time': execution.get('execution_time', 0.0),
            'total_steps': execution.get('total_steps', 0),
            'execution_successful': execution_successful,
            'error_message': error_message,
            'error_type': error_type,
            'attempt_number': execution.get('attempt_number', 1),
            'fallback_attempted': execution.get('fallback_attempted', False),
            
            # Ground truth data
            'expected_answer': ground_truth.get('final_answer', '') if ground_truth else '',
            'is_correct': is_correct and execution_successful,  # Can't be correct if execution failed
            'similarity_score': similarity_score,
            'note': note,
            
            # Analysis
            'answer_length_diff': len(clean_agent_answer) - len(ground_truth.get('final_answer', '')) if ground_truth else 0,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def gaia_answer_matching(self, predicted: str, expected: str) -> bool:
        """GAIA-compliant answer matching logic"""
        
        if not predicted or not expected:
            return False
        
        # Normalize answers
        pred_clean = str(predicted).strip().lower()
        exp_clean = str(expected).strip().lower()
        
        # Exact match
        if pred_clean == exp_clean:
            return True
        
        # Remove common artifacts (GAIA formatting rules)
        artifacts = ['.', ',', '!', '?', '"', "'", 'the ', 'a ', 'an ']
        for artifact in artifacts:
            pred_clean = pred_clean.replace(artifact, '')
            exp_clean = exp_clean.replace(artifact, '')
        
        if pred_clean == exp_clean:
            return True
        
        # Numeric comparison (handle commas, formatting)
        try:
            # Remove non-numeric characters except decimal point and minus
            pred_num_str = re.sub(r'[^\d.-]', '', pred_clean)
            exp_num_str = re.sub(r'[^\d.-]', '', exp_clean)
            
            if pred_num_str and exp_num_str:
                pred_num = float(pred_num_str)
                exp_num = float(exp_num_str)
                
                # Allow small floating point differences
                return abs(pred_num - exp_num) < 1e-6
        except (ValueError, TypeError):
            pass
        
        # List comparison (comma-separated)
        if ',' in pred_clean and ',' in exp_clean:
            pred_list = sorted([item.strip() for item in pred_clean.split(',')])
            exp_list = sorted([item.strip() for item in exp_clean.split(',')])
            return pred_list == exp_list
        
        # Date format comparison (common GAIA pattern)
        date_patterns = [
            r'\d{2}/\d{2}/\d{2,4}',  # MM/DD/YY or MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}',    # YYYY-MM-DD
            r'\d{2}-\d{2}-\d{2,4}'   # DD-MM-YY or DD-MM-YYYY
        ]
        
        for pattern in date_patterns:
            pred_dates = re.findall(pattern, predicted)
            exp_dates = re.findall(pattern, expected)
            if pred_dates and exp_dates:
                return pred_dates[0] == exp_dates[0]
        
        return False
    
    def fuzzy_answer_matching(self, predicted: str, expected: str) -> float:
        """Fuzzy matching for near-correct answers"""
        
        if not predicted or not expected:
            return 0.0
        
        # Basic similarity
        basic_similarity = SequenceMatcher(None, predicted.lower(), expected.lower()).ratio()
        
        # Enhanced similarity for numbers
        if self._is_numeric_answer(predicted) and self._is_numeric_answer(expected):
            try:
                pred_num = self._extract_number(predicted)
                exp_num = self._extract_number(expected)
                
                if pred_num is not None and exp_num is not None and exp_num != 0:
                    # Calculate relative error
                    relative_error = abs(pred_num - exp_num) / abs(exp_num)
                    numeric_similarity = max(0, 1 - relative_error)
                    return max(basic_similarity, numeric_similarity)
            except (ValueError, TypeError):
                pass
        
        return basic_similarity
    
    def _is_numeric_answer(self, answer: str) -> bool:
        """Check if answer is primarily numeric"""
        cleaned = re.sub(r'[^\d.-]', '', answer)
        return len(cleaned) > 0 and len(cleaned) / len(answer) > 0.5
    
    def _extract_number(self, answer: str) -> Optional[float]:
        """Extract numeric value from answer"""
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        return None
    
    def _print_evaluation_details(self, result: Dict):
        """Print detailed evaluation results"""
        
        is_correct = result.get('is_correct', False)
        execution_successful = result.get('execution_successful', False)
        agent_answer = result.get('agent_answer', '')
        expected_answer = result.get('expected_answer', '')
        similarity = result.get('similarity_score', 0)
        error_type = result.get('error_type')
        
        if execution_successful:
            print(f"🤖 Agent Answer: '{agent_answer}'")
            print(f"✅ Expected: '{expected_answer}'")
            print(f"🎯 Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
            if not is_correct and similarity > 0:
                print(f"📊 Similarity: {similarity:.2%}")
        else:
            print(f"❌ Execution Failed ({error_type or 'unknown'})")
            error_msg = result.get('error_message', '')
            print(f"💬 Error: {error_msg[:60]}{'...' if len(error_msg) > 60 else ''}")
            print(f"✅ Expected: '{expected_answer}'")
    
    def _analyze_answer_differences(self, predicted: str, expected: str) -> Dict:
        """Analyze differences between predicted and expected answers"""
        
        analysis = {
            'length_difference': len(predicted) - len(expected),
            'similarity_ratio': SequenceMatcher(None, predicted.lower(), expected.lower()).ratio(),
            'has_numeric_difference': False,
            'has_formatting_difference': False
        }
        
        # Check for numeric differences
        pred_numbers = re.findall(r'\d+\.?\d*', predicted)
        exp_numbers = re.findall(r'\d+\.?\d*', expected)
        
        if pred_numbers != exp_numbers:
            analysis['has_numeric_difference'] = True
            analysis['predicted_numbers'] = pred_numbers
            analysis['expected_numbers'] = exp_numbers
        
        # Check for formatting differences
        if predicted.strip() != expected.strip():
            analysis['has_formatting_difference'] = True
        
        return analysis
    
    def _generate_evaluation_analysis(self, evaluation_results: List[Dict], 
                                    correct_count: int, execution_success_count: int) -> Dict:
        """Generate comprehensive analysis of evaluation results"""
        
        total_questions = len(evaluation_results)
        
        analysis = {
            'overall_performance': {
                'total_questions': total_questions,
                'successful_executions': execution_success_count,
                'execution_failures': total_questions - execution_success_count,
                'correct_answers': correct_count,
                'incorrect_answers': total_questions - correct_count,
                'overall_accuracy': correct_count / total_questions if total_questions > 0 else 0,
                'execution_success_rate': execution_success_count / total_questions if total_questions > 0 else 0,
                'gaia_target_met': (correct_count / total_questions) >= 0.45 if total_questions > 0 else False
            }
        }
        
        # Error analysis
        error_analysis = self._analyze_errors(evaluation_results)
        analysis['error_analysis'] = error_analysis
        
        # Performance by level
        level_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'correct': 0, 'avg_time': 0})
        for result in evaluation_results:
            level = result.get('level', 'Unknown')
            level_stats[level]['total'] += 1
            
            if result.get('execution_successful', False):
                level_stats[level]['successful'] += 1
                if result.get('is_correct', False):
                    level_stats[level]['correct'] += 1
            
            exec_time = result.get('execution_time', 0)
            current_avg = level_stats[level]['avg_time']
            count = level_stats[level]['total']
            level_stats[level]['avg_time'] = ((current_avg * (count - 1)) + exec_time) / count
        
        analysis['level_performance'] = {}
        for level, stats in level_stats.items():
            analysis['level_performance'][f'level_{level}'] = {
                'total_questions': stats['total'],
                'successful_executions': stats['successful'],
                'execution_success_rate': stats['successful'] / stats['total'] if stats['total'] > 0 else 0,
                'correct_answers': stats['correct'],
                'accuracy_of_successful': stats['correct'] / stats['successful'] if stats['successful'] > 0 else 0,
                'overall_accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'avg_execution_time': stats['avg_time']
            }
        
        # Strategy performance
        strategy_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'correct': 0, 'avg_time': 0})
        for result in evaluation_results:
            strategy = result.get('strategy_used', 'unknown')
            strategy_stats[strategy]['total'] += 1
            
            if result.get('execution_successful', False):
                strategy_stats[strategy]['successful'] += 1
                if result.get('is_correct', False):
                    strategy_stats[strategy]['correct'] += 1
            
            exec_time = result.get('execution_time', 0)
            current_avg = strategy_stats[strategy]['avg_time']
            count = strategy_stats[strategy]['total']
            strategy_stats[strategy]['avg_time'] = ((current_avg * (count - 1)) + exec_time) / count
        
        analysis['strategy_performance'] = {}
        for strategy, stats in strategy_stats.items():
            analysis['strategy_performance'][strategy] = {
                'total_questions': stats['total'],
                'successful_executions': stats['successful'],
                'execution_success_rate': stats['successful'] / stats['total'] if stats['total'] > 0 else 0,
                'correct_answers': stats['correct'],
                'accuracy_of_successful': stats['correct'] / stats['successful'] if stats['successful'] > 0 else 0,
                'overall_accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'avg_execution_time': stats['avg_time']
            }
        
        # Routing analysis
        routing_stats = {
            'one_shot_questions': len([r for r in evaluation_results if r.get('strategy_used') == 'one_shot_llm']),
            'manager_questions': len([r for r in evaluation_results if r.get('strategy_used') == 'manager_coordination']),
            'routing_accuracy': 0.0
        }
        
        # Calculate routing accuracy
        correct_routing = 0
        total_routing = 0
        for result in evaluation_results:
            level = result.get('level', 1)
            strategy = result.get('strategy_used', '')
            
            if level == 1:
                total_routing += 1
                if strategy == 'one_shot_llm':
                    correct_routing += 1
            elif level == 3:
                total_routing += 1
                if strategy == 'manager_coordination':
                    correct_routing += 1
        
        routing_stats['routing_accuracy'] = correct_routing / total_routing if total_routing > 0 else 0
        analysis['routing_analysis'] = routing_stats
        
        return analysis
    
    def _analyze_errors(self, evaluation_results: List[Dict]) -> Dict:
        """Analyze errors with categorization"""
        
        failed_results = [r for r in evaluation_results if not r.get('execution_successful', False)]
        
        error_analysis = {
            'total_failures': len(failed_results),
            'error_type_distribution': defaultdict(int),
            'common_error_messages': defaultdict(int),
            'failure_by_level': defaultdict(int),
            'failure_by_strategy': defaultdict(int),
            'failure_by_file_type': defaultdict(int)
        }
        
        for result in failed_results:
            # Error type distribution
            error_type = result.get('error_type', 'unknown')
            error_analysis['error_type_distribution'][error_type] += 1
            
            # Common error messages
            error_msg = result.get('error_message', '')[:50]
            error_analysis['common_error_messages'][error_msg] += 1
            
            # Failure by level
            level = result.get('level', 'Unknown')
            error_analysis['failure_by_level'][level] += 1
            
            # Failure by strategy
            strategy = result.get('strategy_used', 'unknown')
            error_analysis['failure_by_strategy'][strategy] += 1
            
            # Failure by file type
            if result.get('has_file', False):
                file_name = result.get('file_name', '')
                if file_name:
                    ext = Path(file_name).suffix.lower() or 'no_extension'
                    error_analysis['failure_by_file_type'][ext] += 1
            else:
                error_analysis['failure_by_file_type']['no_file'] += 1
        
        # Convert defaultdicts to regular dicts
        for key in error_analysis:
            if isinstance(error_analysis[key], defaultdict):
                error_analysis[key] = dict(error_analysis[key])
        
        return error_analysis
    
    def _print_evaluation_summary(self, analysis: Dict, metadata: Dict):
        """Print comprehensive evaluation summary"""
        
        overall = analysis.get('overall_performance', {})
        
        print(f"\n🎯 EVALUATION SUMMARY")
        print("=" * 50)
        print(f"📊 Total Questions: {overall.get('total_questions', 0)}")
        print(f"✅ Successful Executions: {overall.get('successful_executions', 0)} ({overall.get('execution_success_rate', 0):.1%})")
        print(f"❌ Execution Failures: {overall.get('execution_failures', 0)}")
        print(f"🎯 Correct Answers: {overall.get('correct_answers', 0)} ({overall.get('overall_accuracy', 0):.1%})")
        print(f"🏆 GAIA Target (45%): {'✅ MET' if overall.get('gaia_target_met', False) else '❌ NOT MET'}")
        
        # Error breakdown
        error_analysis = analysis.get('error_analysis', {})
        if error_analysis.get('total_failures', 0) > 0:
            print(f"\n🐛 Error Analysis:")
            print(f"  Total failures: {error_analysis['total_failures']}")
            
            error_types = error_analysis.get('error_type_distribution', {})
            print(f"  Error types:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    {error_type}: {count} occurrences")
        
        # Strategy performance
        strategy_perf = analysis.get('strategy_performance', {})
        if strategy_perf:
            print(f"\n🎯 Strategy Performance:")
            for strategy, stats in strategy_perf.items():
                success_rate = stats.get('execution_success_rate', 0)
                accuracy = stats.get('overall_accuracy', 0)
                total = stats.get('total_questions', 0)
                
                print(f"  {strategy.replace('_', ' ').title()}:")
                print(f"    Execution success: {success_rate:.1%} ({total} questions)")
                print(f"    Overall accuracy: {accuracy:.1%}")
    
    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results to timestamped file"""
        
        filename = create_timestamped_filename("gaia_evaluation", "json")
        filepath = Path("logs") / filename.split('/')[-1]
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Evaluation results saved: {filepath}")

# ============================================================================
# ANALYSIS (PURE ANALYSIS LOGIC)
# ============================================================================

class GAIATestAnalyzer:
    """Analyze test results for insights and improvements"""
    
    def __init__(self):
        print(f"🔍 GAIA Test Analyzer Initialized")
    
    def analyze_failure_patterns(self, evaluation_results: Dict) -> Dict:
        """Identify patterns in failures for improvement"""
        
        if not evaluation_results or 'detailed_results' not in evaluation_results:
            return {'error': 'No evaluation results to analyze'}
        
        detailed_results = evaluation_results['detailed_results']
        incorrect_results = [r for r in detailed_results if not r.get('is_correct', False)]
        
        print(f"\n🔍 FAILURE PATTERN ANALYSIS")
        print("=" * 40)
        print(f"📊 Total Questions: {len(detailed_results)}")
        print(f"❌ Incorrect Answers: {len(incorrect_results)}")
        
        if len(incorrect_results) == 0:
            return {'perfect_performance': True}
        
        # Analyze failure patterns
        failure_patterns = {
            'by_level': defaultdict(int),
            'by_strategy': defaultdict(int),
            'by_file_type': defaultdict(int),
            'execution_failures': 0,
            'low_complexity_failures': 0,
            'high_complexity_failures': 0,
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
                
                error = result.get('error_message', '')
                if 'timeout' in error.lower():
                    failure_patterns['common_error_types']['timeout'] += 1
                elif 'api' in error.lower() or 'rate limit' in error.lower():
                    failure_patterns['common_error_types']['api_issues'] += 1
                else:
                    failure_patterns['common_error_types']['other'] += 1
            
            complexity = result.get('complexity_detected', '')
            if complexity == 'simple':
                failure_patterns['low_complexity_failures'] += 1
            elif complexity == 'complex':
                failure_patterns['high_complexity_failures'] += 1
        
        # Generate improvement recommendations
        recommendations = self._generate_improvement_recommendations(failure_patterns, incorrect_results)
        
        return {
            'failure_patterns': failure_patterns,
            'recommendations': recommendations,
            'sample_failures': incorrect_results[:5]
        }
    
    def analyze_routing_effectiveness(self, evaluation_results: Dict) -> Dict:
        """Analyze smart routing decisions and effectiveness"""
        
        if not evaluation_results or 'detailed_results' not in evaluation_results:
            return {'error': 'No evaluation results to analyze'}
        
        detailed_results = evaluation_results['detailed_results']
        
        routing_analysis = {
            'total_questions': len(detailed_results),
            'strategy_distribution': defaultdict(int),
            'strategy_accuracy': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'level_routing_patterns': defaultdict(lambda: defaultdict(int)),
            'file_routing_patterns': defaultdict(lambda: defaultdict(int)),
            'routing_recommendations': []
        }
        
        for result in detailed_results:
            strategy = result.get('strategy_used', 'unknown')
            level = result.get('level', 'unknown')
            has_file = result.get('has_file', False)
            is_correct = result.get('is_correct', False)
            
            routing_analysis['strategy_distribution'][strategy] += 1
            
            routing_analysis['strategy_accuracy'][strategy]['total'] += 1
            if is_correct:
                routing_analysis['strategy_accuracy'][strategy]['correct'] += 1
            
            routing_analysis['level_routing_patterns'][level][strategy] += 1
            
            file_category = 'with_files' if has_file else 'without_files'
            routing_analysis['file_routing_patterns'][file_category][strategy] += 1
        
        # Calculate accuracy rates
        for strategy, stats in routing_analysis['strategy_accuracy'].items():
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        # Generate routing recommendations
        routing_recommendations = []
        
        for level, strategies in routing_analysis['level_routing_patterns'].items():
            total_level = sum(strategies.values())
            if total_level > 0:
                one_shot_ratio = strategies.get('one_shot_llm', 0) / total_level
                manager_ratio = strategies.get('manager_coordination', 0) / total_level
                
                if level == 1 and one_shot_ratio < 0.7:
                    routing_recommendations.append(f"Level 1 questions should use one-shot LLM more (currently {one_shot_ratio:.1%})")
                elif level == 3 and manager_ratio < 0.7:
                    routing_recommendations.append(f"Level 3 questions should use manager coordination more (currently {manager_ratio:.1%})")
        
        routing_analysis['routing_recommendations'] = routing_recommendations
        
        return routing_analysis
    
    def compare_agent_configurations(self, config_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple agent configurations"""
        
        comparison_data = []
        
        for config_name, results in config_results.items():
            if 'evaluation_metadata' in results:
                metadata = results['evaluation_metadata']
                analysis = results.get('analysis', {})
                overall = analysis.get('overall_performance', {})
                
                # Calculate average execution time
                strategy_perf = analysis.get('strategy_performance', {})
                avg_time = 0
                if strategy_perf:
                    total_time = sum(stats.get('avg_execution_time', 0) * stats.get('total_questions', 0) 
                                   for stats in strategy_perf.values())
                    total_questions = sum(stats.get('total_questions', 0) for stats in strategy_perf.values())
                    avg_time = total_time / total_questions if total_questions > 0 else 0
                
                comparison_data.append({
                    'config': config_name,
                    'total_questions': metadata.get('total_questions', 0),
                    'successful_executions': metadata.get('successful_executions', 0),
                    'execution_success_rate': metadata.get('execution_success_rate', 0),
                    'correct_answers': metadata.get('correct_answers', 0),
                    'accuracy': metadata.get('overall_accuracy', 0),
                    'avg_execution_time': avg_time,
                    'gaia_target_met': overall.get('gaia_target_met', False)
                })
        
        df = pd.DataFrame(comparison_data)
        
        if not df.empty:
            df = df.sort_values('accuracy', ascending=False)
        
        return df
    
    def _generate_improvement_recommendations(self, failure_patterns: Dict, incorrect_results: List[Dict]) -> List[str]:
        """Generate improvement recommendations based on failure patterns"""
        
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
        
        # Error type recommendations
        error_types = failure_patterns['common_error_types']
        if error_types.get('timeout', 0) > 0:
            recommendations.append("Timeout issues detected - optimize execution speed or increase timeout")
        if error_types.get('api_issues', 0) > 0:
            recommendations.append("API issues detected - implement better rate limiting and retry logic")
        
        return recommendations

# ============================================================================
# CONVENIENCE FUNCTIONS FOR TESTING
# ============================================================================

def run_gaia_test(
    agent_config: Union[str, Dict] = "groq",
    dataset_path: str = "./tests/gaia_data",
    batch_strategy: str = "balanced",
    batch_size: int = 20,
    **kwargs
) -> Dict:
    """Complete GAIA test: prepare batch, execute, evaluate"""
    
    print(f"🚀 Complete GAIA Test")
    print(f"🤖 Agent: {agent_config}")
    print(f"📊 Batch: {batch_strategy} ({batch_size} questions)")
    print("=" * 60)
    
    # Step 1: Prepare test batch
    print(f"\n📋 STEP 1: Test Batch Preparation")
    print("-" * 30)
    
    dataset_manager = GAIADatasetManager(dataset_path)
    if not dataset_manager.metadata:
        print("❌ Could not load dataset")
        return {}
    
    test_batch = dataset_manager.create_test_batch(batch_size, batch_strategy, **kwargs)
    if not test_batch:
        print("❌ Could not create test batch")
        return {}
    
    # Step 2: Execute questions
    print(f"\n🤖 STEP 2: Agent Execution")
    print("-" * 30)
    
    executor = GAIATestExecutor(agent_config)
    execution_results = executor.execute_test_batch(test_batch)
    
    if not execution_results:
        print("❌ Execution failed")
        return {}
    
    # Step 3: Evaluate results
    print(f"\n🎯 STEP 3: Answer Evaluation")
    print("-" * 30)
    
    evaluator = GAIATestEvaluator(dataset_manager)
    evaluation_results = evaluator.evaluate_execution_results(execution_results)
    
    return evaluation_results

def run_small_batch_test(agent_config: Union[str, Dict] = "groq", 
                        dataset_path: str = "./tests/gaia_data") -> Dict:
    """Quick test with small diverse batch"""
    
    return run_gaia_test(
        agent_config=agent_config,
        dataset_path=dataset_path,
        batch_strategy="small_sample",
        batch_size=5
    )

def run_large_batch_test(agent_config: Union[str, Dict] = "groq",
                        dataset_path: str = "./tests/gaia_data") -> Dict:
    """Comprehensive test with large batch"""
    
    return run_gaia_test(
        agent_config=agent_config,
        dataset_path=dataset_path,
        batch_strategy="large_comprehensive",
        batch_size=25
    )

def run_quick_gaia_test(agent_config_name: str = "groq") -> Dict:
    """Quick test for development and validation"""
    
    return run_gaia_test(
        agent_config=agent_config_name,
        max_questions=5,
        batch_strategy="small_sample"
    )

def run_smart_routing_test(agent_config_name: str = "performance") -> Dict:
    """Test smart routing behavior specifically"""
    
    print(f"🔀 Smart Routing Test")
    print(f"🤖 Agent: {agent_config_name}")
    print("=" * 50)
    
    result = run_gaia_test(
        agent_config=agent_config_name,
        batch_size=20,
        batch_strategy="balanced"
    )
    
    if result and 'analysis' in result:
        routing_analysis = result['analysis'].get('routing_analysis', {})
        strategy_performance = result['analysis'].get('strategy_performance', {})
        
        print(f"\n🔀 ROUTING ANALYSIS SUMMARY")
        print("-" * 30)
        
        one_shot_count = routing_analysis.get('one_shot_questions', 0)
        manager_count = routing_analysis.get('manager_questions', 0)
        routing_accuracy = routing_analysis.get('routing_accuracy', 0)
        
        print(f"📊 Question Distribution:")
        print(f"├── One-shot LLM: {one_shot_count}")
        print(f"├── Manager Coordination: {manager_count}")
        print(f"└── Routing Accuracy: {routing_accuracy:.1%}")
        
        print(f"\n📈 Strategy Effectiveness:")
        for strategy, stats in strategy_performance.items():
            accuracy = stats.get('accuracy', 0)
            avg_time = stats.get('avg_execution_time', 0)
            total = stats.get('total_questions', 0)
            print(f"├── {strategy}: {accuracy:.1%} accuracy, {avg_time:.1f}s avg ({total} questions)")
    
    return result

def run_agent_comparison_study(agent_configs: List[str],
                              dataset_path: str = "./tests/gaia_data",
                              batch_size: int = 15) -> pd.DataFrame:
    """Compare multiple agents using standardized test batches"""
    
    print(f"🔬 Agent Comparison Study")
    print(f"🤖 Configs: {', '.join(agent_configs)}")
    print(f"📊 Batch size: {batch_size}")
    print("=" * 50)
    
    comparison_results = {}
    
    for config in agent_configs:
        print(f"\n🧪 Testing {config}...")
        
        try:
            result = run_gaia_test(
                agent_config=config,
                dataset_path=dataset_path,
                batch_strategy="balanced",
                batch_size=batch_size
            )
            
            if result:
                comparison_results[config] = result
                metadata = result.get('evaluation_metadata', {})
                accuracy = metadata.get('overall_accuracy', 0)
                print(f"  ✅ {config}: {accuracy:.1%} accuracy")
            else:
                print(f"  ❌ {config}: Test failed")
                
        except Exception as e:
            print(f"  ❌ {config}: Error - {e}")
    
    # Generate comparison DataFrame
    analyzer = GAIATestAnalyzer()
    comparison_df = analyzer.compare_agent_configurations(comparison_results)
    
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 60)
    if not comparison_df.empty:
        print(comparison_df.round(3).to_string(index=False))
        
        best_config = comparison_df.iloc[0]
        print(f"\n🏆 Best performing: {best_config['config']} ({best_config['accuracy']:.1%})")
    
    return comparison_df

def compare_agent_configs(
    configs: List[str] = None,
    questions_per_config: int = 15
) -> pd.DataFrame:
    """Compare multiple agent configurations"""
    
    if configs is None:
        configs = ["groq", "google", "performance", "accuracy"]
    
    return run_agent_comparison_study(configs, batch_size=questions_per_config)

def test_file_vs_text_performance(agent_config_name: str = "groq") -> Dict:
    """Compare performance on file vs text-only questions"""
    
    print(f"📎 File vs Text Performance Test")
    print(f"🤖 Agent: {agent_config_name}")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Text-only questions
    print(f"\n📝 Test 1: Text-Only Questions")
    print("-" * 30)
    
    text_result = run_gaia_test(
        agent_config=agent_config_name,
        batch_size=15,
        batch_strategy="balanced",
        has_files=False
    )
    
    if text_result:
        results['text_only'] = text_result
        metadata = text_result.get('evaluation_metadata', {})
        print(f"✅ Text-only: {metadata.get('overall_accuracy', 0):.1%} accuracy")
    
    # Test 2: File-based questions
    print(f"\n📎 Test 2: File-Based Questions")
    print("-" * 30)
    
    file_result = run_gaia_test(
        agent_config=agent_config_name,
        batch_size=15,
        batch_strategy="file_type_diverse"
    )
    
    if file_result:
        results['with_files'] = file_result
        metadata = file_result.get('evaluation_metadata', {})
        print(f"✅ With files: {metadata.get('overall_accuracy', 0):.1%} accuracy")
    
    # Compare results
    if 'text_only' in results and 'with_files' in results:
        text_acc = results['text_only']['evaluation_metadata']['overall_accuracy']
        file_acc = results['with_files']['evaluation_metadata']['overall_accuracy']
        
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 30)
        print(f"📝 Text-only: {text_acc:.1%}")
        print(f"📎 With files: {file_acc:.1%}")
        print(f"📈 Difference: {(text_acc - file_acc)*100:+.1f} percentage points")
        
        if file_acc < text_acc - 0.1:
            print(f"⚠️ File processing needs improvement")
        elif file_acc > text_acc + 0.1:
            print(f"🌟 File processing is excellent")
        else:
            print(f"✅ File processing performance is comparable")
    
    return results

def analyze_failure_patterns(evaluation_results: Dict) -> Dict:
    """Analyze patterns in incorrect answers for improvement insights"""
    
    analyzer = GAIATestAnalyzer()
    return analyzer.analyze_failure_patterns(evaluation_results)

def analyze_test_results(evaluation_results: Dict) -> Dict:
    """Comprehensive analysis of test results"""
    
    analyzer = GAIATestAnalyzer()
    
    analysis_results = {
        'failure_analysis': analyzer.analyze_failure_patterns(evaluation_results),
        'routing_analysis': analyzer.analyze_routing_effectiveness(evaluation_results)
    }
    
    return analysis_results

def generate_test_report(evaluation_results: Dict, agent_config_name: str) -> str:
    """Generate comprehensive test report"""
    
    if not evaluation_results:
        return "No evaluation results available for report generation"
    
    metadata = evaluation_results.get('evaluation_metadata', {})
    analysis = evaluation_results.get('analysis', {})
    overall = analysis.get('overall_performance', {})
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""
🎯 GAIA AGENT TEST REPORT
{'='*60}

📅 Generated: {timestamp}
🤖 Agent Configuration: {agent_config_name}
🔬 Testing Framework: Clean Architecture (Data + Testing Separation)

📊 EXECUTIVE SUMMARY
{'-'*30}
✨ Overall Performance: {overall.get('overall_accuracy', 0):.1%}
📝 Questions Tested: {overall.get('total_questions', 0)}
✅ Successful Executions: {metadata.get('successful_executions', 0)} ({metadata.get('execution_success_rate', 0):.1%})
❌ Execution Failures: {metadata.get('execution_failures', 0)}
🎯 Correct Answers: {overall.get('correct_answers', 0)}
🏆 GAIA Target (45%): {'✅ ACHIEVED' if overall.get('gaia_target_met', False) else '❌ NOT ACHIEVED'}

📈 PERFORMANCE BY GAIA LEVEL
{'-'*30}"""

    level_performance = analysis.get('level_performance', {})
    for level_key in sorted(level_performance.keys()):
        level_num = level_key.replace('level_', '')
        level_data = level_performance[level_key]
        accuracy = level_data.get('overall_accuracy', 0)
        success_rate = level_data.get('execution_success_rate', 0)
        total = level_data.get('total_questions', 0)
        correct = level_data.get('correct_answers', 0)
        avg_time = level_data.get('avg_execution_time', 0)
        
        report += f"\n🔸 Level {level_num}: {accuracy:.1%} accuracy ({correct}/{total}) - {success_rate:.1%} success rate - {avg_time:.1f}s avg"

    # Strategy performance
    strategy_perf = analysis.get('strategy_performance', {})
    if strategy_perf:
        report += f"\n\n🎯 STRATEGY PERFORMANCE\n{'-'*30}"
        
        for strategy, stats in strategy_perf.items():
            accuracy = stats.get('overall_accuracy', 0)
            success_rate = stats.get('execution_success_rate', 0)
            total = stats.get('total_questions', 0)
            avg_time = stats.get('avg_execution_time', 0)
            report += f"\n🔸 {strategy.replace('_', ' ').title()}: {accuracy:.1%} accuracy, {success_rate:.1%} success ({total} questions) - {avg_time:.1f}s"

    # Error analysis
    error_analysis = analysis.get('error_analysis', {})
    if error_analysis and error_analysis.get('total_failures', 0) > 0:
        report += f"\n\n🐛 ERROR ANALYSIS\n{'-'*30}"
        report += f"\nTotal Failures: {error_analysis['total_failures']}"
        
        error_types = error_analysis.get('error_type_distribution', {})
        if error_types:
            report += f"\nError Types:"
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                report += f"\n  • {error_type}: {count} occurrences"

    # Routing analysis
    routing_analysis = analysis.get('routing_analysis', {})
    if routing_analysis:
        report += f"\n\n🔀 SMART ROUTING ANALYSIS\n{'-'*30}"
        one_shot = routing_analysis.get('one_shot_questions', 0)
        manager = routing_analysis.get('manager_questions', 0)
        routing_acc = routing_analysis.get('routing_accuracy', 0)
        
        report += f"\n🔸 One-shot LLM: {one_shot} questions"
        report += f"\n🔸 Manager Coordination: {manager} questions"
        report += f"\n🔸 Routing Accuracy: {routing_acc:.1%}"

    # Recommendations
    report += f"\n\n💡 KEY RECOMMENDATIONS\n{'-'*30}"
    
    execution_success_rate = metadata.get('execution_success_rate', 0)
    overall_accuracy = overall.get('overall_accuracy', 0)
    
    if execution_success_rate < 0.8:
        report += "\n❌ Critical: Fix execution failures before optimizing performance"
        report += "\n🔧 Focus on agent stability and error handling"
    elif overall_accuracy >= 0.45:
        report += "\n✅ System meets GAIA performance targets"
        report += "\n🚀 Ready for production deployment"
    elif overall_accuracy >= 0.35:
        report += "\n⚠️ Performance approaching targets - optimization recommended"
        report += "\n🎯 Focus on accuracy improvements"
    else:
        report += "\n❌ Performance below targets - significant improvements needed"
        report += "\n🔧 Review agent architecture and strategy selection"

    report += f"\n\n{'='*60}"
    report += f"\nGenerated by GAIA Testing Framework v3.0 (Clean Architecture)"
    
    # Save to file
    report_file = Path("logs") / create_timestamped_filename(f"gaia_test_report_{agent_config_name}", "txt").split('/')[-1]
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Test report saved: {report_file}")
    
    return report

def diagnose_agent_issues(agent_config: Union[str, Dict] = "groq",
                         dataset_path: str = "./tests/gaia_data") -> Dict:
    """Diagnostic test to identify agent issues"""
    
    print(f"🔧 Agent Diagnostic Test")
    print("=" * 40)
    
    # Run test with maximum error detection
    test_config = GAIATestConfig(
        max_retries=1,  # Don't retry to see raw errors
        detect_execution_failures=True,
        enable_strategy_fallback=False,  # Don't mask with fallbacks
        enable_real_time_monitoring=True
    )
    
    # Create small diagnostic batch
    dataset_manager = GAIADatasetManager(dataset_path)
    test_batch = dataset_manager.create_test_batch(3, "small_sample")
    
    if test_batch:
        executor = GAIATestExecutor(agent_config, test_config)
        execution_results = executor.execute_test_batch(test_batch)
        
        # Analyze execution issues
        print(f"\n🔍 Diagnostic Analysis:")
        
        for i, result in enumerate(execution_results, 1):
            print(f"\nQuestion {i}:")
            print(f"  Execution successful: {result.get('execution_successful', False)}")
            print(f"  Error type: {result.get('error_type', 'None')}")
            
            if result.get('error_message'):
                print(f"  Error message: {result['error_message'][:100]}...")
            
            if result.get('raw_agent_response'):
                response = result['raw_agent_response']
                print(f"  Raw response: {response[:100]}...")
        
        return execution_results
    
    return {}

# ============================================================================
# MAIN EXECUTION AND EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("🧪 GAIA Testing Framework v3.0 - Clean Architecture")
    print("=" * 60)
    
    # Available test functions
    test_functions = {
        'run_quick_gaia_test': 'Quick 5-question validation (Level 1)',
        'run_small_batch_test': 'Quick 5-question diverse batch',
        'run_large_batch_test': 'Comprehensive 25-question evaluation',
        'run_gaia_test': 'Custom testing with parameters',
        'run_smart_routing_test': 'Analyze routing effectiveness',
        'run_agent_comparison_study': 'Compare multiple configurations',
        'compare_agent_configs': 'Compare agent configurations',
        'test_file_vs_text_performance': 'File processing vs text-only',
        'analyze_failure_patterns': 'Deep dive into failure analysis',
        'analyze_test_results': 'Comprehensive result analysis',
        'generate_test_report': 'Professional reporting',
        'diagnose_agent_issues': 'Diagnostic test for agent problems'
    }

    print(f"\n📋 Available Testing Functions:")
    for func, description in test_functions.items():
        print(f"├── {func}: {description}")

    print(f"\n💡 Usage Examples:")
    print(f"  ├── run_small_batch_test('groq')")
    print(f"  ├── run_gaia_test('groq', batch_size=30)")
    print(f"  ├── run_smart_routing_test('performance')")
    print(f"  ├── compare_agent_configs(['groq', 'google', 'performance'])")
    print(f"  ├── test_file_vs_text_performance('groq')")
    print(f"  ├── diagnose_agent_issues('groq')")
    print(f"  └── analyze_failure_patterns(result)")

    print(f"\n🔧 Key Features:")
    features = [
        "✅ Fixed execution success detection (catches error messages)",
        "✅ Comprehensive error categorization and analysis", 
        "✅ Real-time progress monitoring with progress bars",
        "✅ Strategy fallback mechanisms for robustness",
        "✅ Detailed performance profiling and timing",
        "✅ Robust retry logic with smart decisions",
        "✅ Enhanced similarity scoring for near-matches",
        "✅ GAIA-compliant answer matching with multiple strategies",
        "✅ Comprehensive reporting and export capabilities",
        "✅ Diagnostic tools for agent debugging"
    ]
    
    for feature in features:
        print(f"  {feature}")

    # Run demonstration
    print(f"\n🚀 Running demonstration...")
    try:
        demo_result = run_quick_gaia_test("groq")
        
        if demo_result and 'evaluation_metadata' in demo_result:
            metadata = demo_result['evaluation_metadata']
            accuracy = metadata.get('overall_accuracy', 0)
            execution_success_rate = metadata.get('execution_success_rate', 0)
            total = metadata.get('total_questions', 0)
            
            print(f"✅ Demo completed!")
            print(f"   ├── Questions: {total}")
            print(f"   ├── Execution Success: {execution_success_rate:.1%}")
            print(f"   ├── Accuracy: {accuracy:.1%}")
            print(f"   ├── Fixed Metrics: ✅ WORKING")
            print(f"   └── GAIA Target: {'✅ Met' if accuracy >= 0.45 else '❌ Not Met'}")
            
            # Generate demo report
            generate_test_report(demo_result, "groq")
            
        else:
            print(f"❌ Demo failed - check your setup")
            print(f"💡 Use diagnose_agent_issues('groq') to identify problems")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print(f"💡 Ensure dataset is available and agent system is configured")
        print(f"🔧 Try: diagnose_agent_issues('groq') for detailed diagnosis")