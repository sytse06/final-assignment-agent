# agent_testing.py
# Two-step GAIA testing framework adapted for refactored agent system

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
import uuid
import re
from difflib import SequenceMatcher

# Import our current agent system
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
from agent_logging import create_timestamped_filename, get_latest_log_file

# ============================================================================
# TESTING CONFIGURATION
# ============================================================================

@dataclass
class GAIATestConfig:
    """Configuration for two-step GAIA testing"""
    # Execution settings
    max_questions: int = 50
    timeout_per_question: int = 180
    save_intermediate: bool = True
    
    # Output settings
    results_dir: str = "logs"
    
    # Safety settings
    enable_retries: bool = True
    max_retries: int = 2
    
    # Content filtering
    include_file_questions: bool = True
    include_image_questions: bool = True
    target_levels: List[int] = None  # None = all levels
    
    # Agent configuration
    agent_config_name: str = "groq"
    custom_agent_config: Dict = None

# ============================================================================
# AGENT CONFIGURATION MAPPING
# ============================================================================

def get_agent_config_by_name(config_name: str) -> Dict:
    """Map configuration names to actual configs"""
    
    config_map = {
        "groq": get_groq_config("qwen-qwq-32b"),
        "groq_fast": get_groq_config("llama-3.3-70b-versatile"),
        "openrouter": get_openrouter_config("qwen/qwen-2.5-coder-32b-instruct:free"),
        "openrouter_premium": get_openrouter_config("deepseek/deepseek-chat"),
        "google": get_google_config("gemini-2.0-flash-preview"),
        "google_pro": get_google_config("gemini-1.5-pro-002"),
        "ollama": get_ollama_config("qwen2.5-coder:32b"),
        "performance": {**get_groq_config(), **get_performance_config()},
        "accuracy": {**get_groq_config(), **get_accuracy_config()}
    }
    
    return config_map.get(config_name, get_groq_config())

# ============================================================================
# STEP 1: QUESTION EXECUTION (BLIND)
# ============================================================================

class GAIAQuestionExecutor:
    """Step 1: Execute questions without knowing the ground truth"""
    
    def __init__(self, config: GAIATestConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.execution_log = []
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"🎯 GAIA Question Executor Initialized")
        print(f"📊 Session: {self.session_id}")
        print(f"📁 Results: {self.results_dir}")
    
    def load_questions_blind(self, metadata_path: str = "metadata.jsonl") -> List[Dict]:
        """Load questions WITHOUT ground truth answers"""
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                all_questions = [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"❌ Metadata file not found: {metadata_path}")
            return self._generate_sample_questions()
        
        # Filter questions based on config
        filtered_questions = []
        
        for question_data in all_questions:
            # Level filtering
            if self.config.target_levels:
                if question_data.get('Level') not in self.config.target_levels:
                    continue
            
            # File question filtering
            has_file = bool(question_data.get('file_name', '').strip())
            if has_file and not self.config.include_file_questions:
                continue
            
            # Image question filtering  
            file_name = question_data.get('file_name', '').lower()
            is_image = file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
            if is_image and not self.config.include_image_questions:
                continue
            
            # Create blind question (remove ground truth)
            blind_question = {
                'task_id': question_data.get('task_id'),
                'question': question_data.get('Question'),
                'level': question_data.get('Level'),
                'file_name': question_data.get('file_name', ''),
                'file_path': question_data.get('file_path', ''),
                'has_file': has_file,
                'is_image_question': is_image,
                # Deliberately exclude 'Final answer'
            }
            
            filtered_questions.append(blind_question)
        
        # Limit to max questions
        selected_questions = filtered_questions[:self.config.max_questions]
        
        print(f"📋 Loaded {len(selected_questions)} questions for execution")
        self._print_question_summary(selected_questions)
        
        return selected_questions
    
    def _print_question_summary(self, questions: List[Dict]):
        """Print summary of question distribution"""
        
        # Level distribution
        level_counts = defaultdict(int)
        file_counts = {'with_files': 0, 'without_files': 0}
        image_count = 0
        
        for q in questions:
            level_counts[q['level']] += 1
            if q['has_file']:
                file_counts['with_files'] += 1
            else:
                file_counts['without_files'] += 1
            if q['is_image_question']:
                image_count += 1
        
        print(f"📊 Question Distribution:")
        print(f"├── By Level:")
        for level in sorted(level_counts.keys()):
            print(f"│   ├── Level {level}: {level_counts[level]} questions")
        print(f"├── File Attachments:")
        print(f"│   ├── With files: {file_counts['with_files']}")
        print(f"│   └── Without files: {file_counts['without_files']}")
        print(f"└── Image questions: {image_count}")
    
    def execute_questions_batch(self, questions: List[Dict] = None) -> str:
        """Step 1: Execute all questions and save results"""
        
        if questions is None:
            questions = self.load_questions_blind()
        
        if not questions:
            print("❌ No questions to execute")
            return None
        
        print(f"\n🚀 Starting Question Execution")
        print(f"📝 Agent Config: {self.config.agent_config_name}")
        print(f"📊 Questions: {len(questions)}")
        print("=" * 60)
        
        # Create agent using our current system
        try:
            # Get configuration
            if self.config.custom_agent_config:
                agent_config = self.config.custom_agent_config
            else:
                agent_config = get_agent_config_by_name(self.config.agent_config_name)
            
            # Create agent
            agent = create_gaia_agent(agent_config)
            print(f"✅ Agent created: {self.config.agent_config_name}")
            
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            traceback.print_exc()
            return None
        
        # Execution log file with timestamp
        execution_file = self.results_dir / create_timestamped_filename("gaia_executions", "jsonl").split('/')[-1]
        
        try:
            with open(execution_file, 'w') as f:
                
                for i, question_data in enumerate(questions):
                    print(f"\n🔄 Question {i+1}/{len(questions)}")
                    print(f"📋 Task ID: {question_data['task_id']}")
                    print(f"📊 Level: {question_data['level']}")
                    
                    if question_data['has_file']:
                        print(f"📎 File: {question_data['file_name']}")
                    
                    # Show question (truncated for readability)
                    question_text = question_data['question']
                    preview = question_text[:100] + "..." if len(question_text) > 100 else question_text
                    print(f"❓ Question: {preview}")
                    
                    execution_result = self._execute_single_question(
                        agent, question_data, attempt=1
                    )
                    
                    # Save immediately
                    f.write(json.dumps(execution_result, default=str) + '\n')
                    f.flush()
                    
                    # Print immediate result
                    if execution_result.get('success', False):
                        answer = execution_result.get('agent_answer', 'No answer')
                        duration = execution_result.get('execution_time', 0)
                        strategy = execution_result.get('strategy_used', '')
                        print(f"✅ Answer: {answer}")
                        print(f"⏱️  Time: {duration:.2f}s")
                        if strategy:
                            print(f"🎯 Strategy: {strategy}")
                    else:
                        error = execution_result.get('error', 'Unknown error')
                        print(f"❌ Error: {error}")
                    
                    # Add to session log
                    self.execution_log.append(execution_result)
        
        except Exception as e:
            print(f"❌ Execution failed: {e}")
            traceback.print_exc()
            return None
        
        print(f"\n🎉 Execution Complete!")
        print(f"📁 Results saved: {execution_file}")
        print(f"📊 Total executed: {len(self.execution_log)}")
        
        return str(execution_file)
    
    def _execute_single_question(self, agent: GAIAAgent, question_data: Dict, attempt: int = 1) -> Dict:
        """Execute single question with comprehensive tracking"""
        
        start_time = time.time()
        task_id = question_data['task_id']
        
        execution_record = {
            'session_id': self.session_id,
            'task_id': task_id,
            'question': question_data['question'],
            'level': question_data['level'],
            'has_file': question_data['has_file'],
            'file_name': question_data.get('file_name', ''),
            'file_path': question_data.get('file_path', ''),
            'is_image_question': question_data['is_image_question'],
            'attempt': attempt,
            'timestamp': datetime.now().isoformat(),
            'agent_config': self.config.agent_config_name,
        }
        
        try:
            # Execute with our current agent system
            result = agent.run_single_question(
                question=question_data['question'],
                task_id=task_id
            )
            
            execution_time = time.time() - start_time
            
            # Extract agent response from our current system
            agent_answer = result.get('final_answer', '')
            raw_answer = result.get('raw_answer', '')
            steps = result.get('steps', [])
            complexity = result.get('complexity', '')
            
            # Determine strategy used based on our routing system
            strategy_used = self._determine_strategy_used(result, complexity)
            
            execution_record.update({
                'success': True,
                'agent_answer': agent_answer,
                'raw_answer': raw_answer,
                'strategy_used': strategy_used,
                'complexity_detected': complexity,
                'execution_time': execution_time,
                'total_steps': len(steps),
                'steps_summary': steps[-3:] if len(steps) > 3 else steps,  # Last 3 steps
                'error': None
            })
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            execution_record.update({
                'success': False,
                'error': error_msg,
                'execution_time': execution_time,
                'agent_answer': '',
                'strategy_used': 'failed',
                'complexity_detected': '',
                'total_steps': 0
            })
            
            # Retry logic
            if attempt < self.config.max_retries and self.config.enable_retries:
                print(f"    🔄 Retry attempt {attempt + 1}")
                return self._execute_single_question(agent, question_data, attempt + 1)
        
        return execution_record
    
    def _determine_strategy_used(self, result: Dict, complexity: str) -> str:
        """Determine which strategy was used based on our system"""
        
        # Our system uses routing based on complexity
        if complexity == "simple":
            return "one_shot_llm"
        elif complexity == "complex":
            return "manager_coordination"
        else:
            # Fallback - analyze steps to determine
            steps = result.get('steps', [])
            if len(steps) <= 2:
                return "one_shot_llm"
            else:
                return "manager_coordination"
    
    def _generate_sample_questions(self) -> List[Dict]:
        """Generate sample questions if metadata not available"""
        sample_questions = [
            {
                'task_id': f'sample_{i}',
                'question': f'Sample question {i}: Calculate {10 + i} * {20 + i}',
                'level': 1 if i < 3 else 2,
                'file_name': '',
                'file_path': '',
                'has_file': False,
                'is_image_question': False
            }
            for i in range(5)
        ]
        
        print("⚠️ Using sample questions - metadata.jsonl not found")
        return sample_questions

# ============================================================================
# STEP 2: ANSWER EVALUATION (WITH GROUND TRUTH)
# ============================================================================

class GAIAAnswerEvaluator:
    """Step 2: Evaluate executed answers against ground truth"""
    
    def __init__(self, config: GAIATestConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        
        print(f"🎯 GAIA Answer Evaluator Initialized")
        print(f"📁 Results directory: {self.results_dir}")
    
    def load_ground_truth(self, metadata_path: str = "metadata.jsonl") -> Dict[str, Dict]:
        """Load ground truth answers by task_id"""
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                all_data = [json.loads(line) for line in f]
            
            ground_truth = {}
            for item in all_data:
                task_id = item.get('task_id')
                if task_id:
                    ground_truth[task_id] = {
                        'final_answer': item.get('Final answer', ''),
                        'level': item.get('Level'),
                        'question': item.get('Question', ''),
                        'annotator_metadata': item.get('Annotator Metadata', {}),
                        'file_name': item.get('file_name', ''),
                        'file_path': item.get('file_path', '')
                    }
            
            print(f"✅ Loaded ground truth for {len(ground_truth)} questions")
            return ground_truth
            
        except FileNotFoundError:
            print(f"❌ Ground truth file not found: {metadata_path}")
            return {}
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            return {}
    
    def evaluate_execution_results(self, 
                                 execution_file: str,
                                 ground_truth: Dict[str, Dict] = None) -> Dict:
        """Step 2: Evaluate all execution results against ground truth"""
        
        if ground_truth is None:
            ground_truth = self.load_ground_truth()
        
        if not ground_truth:
            print("❌ No ground truth available for evaluation")
            return {}
        
        # Load execution results
        try:
            execution_results = []
            with open(execution_file, 'r') as f:
                for line in f:
                    execution_results.append(json.loads(line))
            
            print(f"📊 Evaluating {len(execution_results)} execution results")
            
        except Exception as e:
            print(f"❌ Error loading execution results: {e}")
            return {}
        
        print(f"\n🔍 Starting Answer Evaluation")
        print("=" * 50)
        
        evaluation_results = []
        correct_count = 0
        
        for i, execution in enumerate(execution_results):
            task_id = execution.get('task_id')
            agent_answer = execution.get('agent_answer', '')
            
            print(f"\n📝 Evaluation {i+1}/{len(execution_results)}")
            print(f"📋 Task ID: {task_id}")
            
            if task_id not in ground_truth:
                print(f"⚠️ No ground truth found for task {task_id}")
                evaluation_result = self._create_evaluation_record(
                    execution, None, False, "No ground truth available"
                )
            else:
                gt_data = ground_truth[task_id]
                expected_answer = gt_data['final_answer']
                
                print(f"🤖 Agent Answer: '{agent_answer}'")
                print(f"✅ Expected: '{expected_answer}'")
                
                # Evaluate correctness
                is_correct = self._evaluate_gaia_answer_match(agent_answer, expected_answer)
                correct_count += is_correct
                
                print(f"🎯 Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                
                evaluation_result = self._create_evaluation_record(
                    execution, gt_data, is_correct
                )
            
            evaluation_results.append(evaluation_result)
        
        # Generate comprehensive analysis
        analysis = self._generate_evaluation_analysis(evaluation_results, correct_count)
        
        # Save evaluation results with timestamp
        eval_file = self.results_dir / create_timestamped_filename("gaia_evaluation", "json").split('/')[-1]
        
        final_results = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'execution_file': execution_file,
                'total_questions': len(evaluation_results),
                'correct_answers': correct_count,
                'overall_accuracy': correct_count / len(evaluation_results) if evaluation_results else 0,
                'agent_config': self.config.agent_config_name
            },
            'detailed_results': evaluation_results,
            'analysis': analysis
        }
        
        with open(eval_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n🎉 Evaluation Complete!")
        print(f"📁 Results saved: {eval_file}")
        self._print_evaluation_summary(analysis)
        
        return final_results
    
    def _create_evaluation_record(self, 
                                execution: Dict, 
                                ground_truth: Dict, 
                                is_correct: bool,
                                note: str = None) -> Dict:
        """Create comprehensive evaluation record"""
        
        return {
            'task_id': execution.get('task_id'),
            'question': execution.get('question'),
            'level': execution.get('level'),
            'has_file': execution.get('has_file', False),
            'is_image_question': execution.get('is_image_question', False),
            'file_name': execution.get('file_name', ''),
            
            # Execution data
            'agent_answer': execution.get('agent_answer', ''),
            'raw_answer': execution.get('raw_answer', ''),
            'strategy_used': execution.get('strategy_used', ''),
            'complexity_detected': execution.get('complexity_detected', ''),
            'execution_time': execution.get('execution_time', 0.0),
            'total_steps': execution.get('total_steps', 0),
            'execution_successful': execution.get('success', False),
            'error': execution.get('error'),
            
            # Ground truth data
            'expected_answer': ground_truth.get('final_answer', '') if ground_truth else '',
            'is_correct': is_correct,
            'note': note,
            
            # Evaluation metadata
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def _evaluate_gaia_answer_match(self, predicted: str, expected: str) -> bool:
        """GAIA-style answer matching with multiple strategies"""
        
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
        
        # Partial match for very close answers (but be conservative)
        if len(exp_clean) > 5:
            similarity = SequenceMatcher(None, pred_clean, exp_clean).ratio()
            return similarity > 0.95  # Very high threshold
        
        return False
    
    def _generate_evaluation_analysis(self, evaluation_results: List[Dict], correct_count: int) -> Dict:
        """Generate comprehensive analysis of evaluation results"""
        
        if not evaluation_results:
            return {}
        
        total_questions = len(evaluation_results)
        overall_accuracy = correct_count / total_questions
        
        analysis = {
            'overall_performance': {
                'total_questions': total_questions,
                'correct_answers': correct_count,
                'incorrect_answers': total_questions - correct_count,
                'overall_accuracy': overall_accuracy,
                'gaia_target_met': overall_accuracy >= 0.45  # GAIA benchmark target
            }
        }
        
        # Performance by level
        level_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'avg_time': 0})
        for result in evaluation_results:
            level = result.get('level', 1)
            level_stats[level]['total'] += 1
            if result.get('is_correct', False):
                level_stats[level]['correct'] += 1
            
            # Track execution time
            exec_time = result.get('execution_time', 0)
            current_avg = level_stats[level]['avg_time']
            count = level_stats[level]['total']
            level_stats[level]['avg_time'] = ((current_avg * (count - 1)) + exec_time) / count
        
        analysis['level_performance'] = {}
        for level, stats in level_stats.items():
            analysis['level_performance'][f'level_{level}'] = {
                'total_questions': stats['total'],
                'correct_answers': stats['correct'],
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'avg_execution_time': stats['avg_time']
            }
        
        # File attachment performance
        file_stats = {'with_files': {'total': 0, 'correct': 0}, 'without_files': {'total': 0, 'correct': 0}}
        for result in evaluation_results:
            category = 'with_files' if result.get('has_file', False) else 'without_files'
            file_stats[category]['total'] += 1
            if result.get('is_correct', False):
                file_stats[category]['correct'] += 1
        
        analysis['file_attachment_performance'] = {}
        for category, stats in file_stats.items():
            analysis['file_attachment_performance'][category] = {
                'total_questions': stats['total'],
                'correct_answers': stats['correct'],
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            }
        
        # Strategy performance (routing effectiveness)
        strategy_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'avg_time': 0})
        for result in evaluation_results:
            strategy = result.get('strategy_used', 'unknown')
            strategy_stats[strategy]['total'] += 1
            if result.get('is_correct', False):
                strategy_stats[strategy]['correct'] += 1
            
            # Track execution time by strategy
            exec_time = result.get('execution_time', 0)
            current_avg = strategy_stats[strategy]['avg_time']
            count = strategy_stats[strategy]['total']
            strategy_stats[strategy]['avg_time'] = ((current_avg * (count - 1)) + exec_time) / count
        
        analysis['strategy_performance'] = {}
        for strategy, stats in strategy_stats.items():
            analysis['strategy_performance'][strategy] = {
                'total_questions': stats['total'],
                'correct_answers': stats['correct'],
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'avg_execution_time': stats['avg_time']
            }
        
        # Error analysis
        error_questions = [r for r in evaluation_results if not r.get('execution_successful', True)]
        
        analysis['error_analysis'] = {
            'execution_errors': len(error_questions),
            'execution_success_rate': 1 - (len(error_questions) / total_questions),
            'sample_errors': [r.get('error') for r in error_questions[:3] if r.get('error')]
        }
        
        # Routing analysis (specific to our system)
        routing_stats = {
            'one_shot_questions': len([r for r in evaluation_results if r.get('strategy_used') == 'one_shot_llm']),
            'manager_questions': len([r for r in evaluation_results if r.get('strategy_used') == 'manager_coordination']),
            'routing_accuracy': 0.0
        }
        
        # Calculate routing accuracy (simple questions should use one_shot, complex should use manager)
        correct_routing = 0
        total_routing = 0
        for result in evaluation_results:
            level = result.get('level', 1)
            strategy = result.get('strategy_used', '')
            
            # Level 1 questions should ideally use one_shot
            if level == 1:
                total_routing += 1
                if strategy == 'one_shot_llm':
                    correct_routing += 1
            # Level 3 questions should ideally use manager
            elif level == 3:
                total_routing += 1
                if strategy == 'manager_coordination':
                    correct_routing += 1
        
        routing_stats['routing_accuracy'] = correct_routing / total_routing if total_routing > 0 else 0
        analysis['routing_analysis'] = routing_stats
        
        return analysis
    
    def _print_evaluation_summary(self, analysis: Dict):
        """Print comprehensive evaluation summary"""
        
        overall = analysis.get('overall_performance', {})
        
        print(f"\n🎯 EVALUATION SUMMARY")
        print("=" * 40)
        print(f"📊 Total Questions: {overall.get('total_questions', 0)}")
        print(f"✅ Correct Answers: {overall.get('correct_answers', 0)}")
        print(f"❌ Incorrect Answers: {overall.get('incorrect_answers', 0)}")
        print(f"🎯 Overall Accuracy: {overall.get('overall_accuracy', 0):.1%}")
        print(f"🏆 GAIA Target (45%): {'✅ MET' if overall.get('gaia_target_met', False) else '❌ NOT MET'}")
        
        # Level performance
        level_perf = analysis.get('level_performance', {})
        if level_perf:
            print(f"\n📈 Performance by Level:")
            for level_key, stats in level_perf.items():
                level_num = level_key.replace('level_', '')
                accuracy = stats.get('accuracy', 0)
                total = stats.get('total_questions', 0)
                correct = stats.get('correct_answers', 0)
                avg_time = stats.get('avg_execution_time', 0)
                print(f"├── Level {level_num}: {accuracy:.1%} ({correct}/{total}) - {avg_time:.1f}s avg")
        
        # Strategy performance (routing effectiveness)
        strategy_perf = analysis.get('strategy_performance', {})
        if strategy_perf:
            print(f"\n🎯 Strategy Performance:")
            for strategy, stats in strategy_perf.items():
                accuracy = stats.get('accuracy', 0)
                total = stats.get('total_questions', 0)
                avg_time = stats.get('avg_execution_time', 0)
                print(f"├── {strategy.replace('_', ' ').title()}: {accuracy:.1%} ({total} questions) - {avg_time:.1f}s avg")
        
        # Routing analysis
        routing_analysis = analysis.get('routing_analysis', {})
        if routing_analysis:
            print(f"\n🔀 Routing Analysis:")
            one_shot = routing_analysis.get('one_shot_questions', 0)
            manager = routing_analysis.get('manager_questions', 0)
            routing_acc = routing_analysis.get('routing_accuracy', 0)
            print(f"├── One-shot LLM: {one_shot} questions")
            print(f"├── Manager Coordination: {manager} questions")
            print(f"└── Routing Accuracy: {routing_acc:.1%}")
        
        # File performance
        file_perf = analysis.get('file_attachment_performance', {})
        if file_perf:
            print(f"\n📎 File Attachment Performance:")
            for category, stats in file_perf.items():
                accuracy = stats.get('accuracy', 0)
                total = stats.get('total_questions', 0)
                print(f"├── {category.replace('_', ' ').title()}: {accuracy:.1%} ({total} questions)")
        
        # Error summary
        error_analysis = analysis.get('error_analysis', {})
        if error_analysis:
            exec_errors = error_analysis.get('execution_errors', 0)
            success_rate = error_analysis.get('execution_success_rate', 0)
            
            print(f"\n🔧 Error Analysis:")
            print(f"├── Execution Success Rate: {success_rate:.1%}")
            print(f"└── Execution Errors: {exec_errors}")

# ============================================================================
# CONVENIENCE FUNCTIONS FOR TWO-STEP TESTING
# ============================================================================

def run_gaia_test(
    agent_config_name: str = "groq",
    max_questions: int = 20,
    target_levels: List[int] = None,
    include_files: bool = True,
    include_images: bool = True,
    custom_config: Dict = None
) -> Dict:
    """Complete two-step GAIA test: execute then evaluate"""
    
    print(f"🚀 Two-Step GAIA Test")
    print(f"🤖 Agent: {agent_config_name}")
    print(f"📊 Max Questions: {max_questions}")
    if target_levels:
        print(f"🎯 Target Levels: {target_levels}")
    print("=" * 50)
    
    # Configure testing
    config = GAIATestConfig(
        max_questions=max_questions,
        target_levels=target_levels,
        include_file_questions=include_files,
        include_image_questions=include_images,
        agent_config_name=agent_config_name,
        custom_agent_config=custom_config
    )
    
    # Step 1: Execute questions (blind)
    print(f"\n📋 STEP 1: Question Execution")
    print("-" * 30)
    
    executor = GAIAQuestionExecutor(config)
    execution_file = executor.execute_questions_batch()
    
    if not execution_file:
        print("❌ Execution failed")
        return {}
    
    # Step 2: Evaluate answers (with ground truth)
    print(f"\n🎯 STEP 2: Answer Evaluation")
    print("-" * 30)
    
    evaluator = GAIAAnswerEvaluator(config)
    evaluation_results = evaluator.evaluate_execution_results(execution_file)
    
    return evaluation_results

def run_quick_gaia_test(agent_config_name: str = "groq") -> Dict:
    """Quick test for development and validation"""
    
    return run_gaia_test(
        agent_config_name=agent_config_name,
        max_questions=5,
        target_levels=[1],  # Level 1 only for quick validation
        include_files=False,  # Skip files for speed
        include_images=False
    )

def run_smart_routing_test(agent_config_name: str = "performance") -> Dict:
    """Test smart routing behavior specifically"""
    
    print(f"🔀 Smart Routing Test")
    print(f"🤖 Agent: {agent_config_name}")
    print("=" * 50)
    
    # Test with mix of simple and complex questions
    result = run_gaia_test(
        agent_config_name=agent_config_name,
        max_questions=20,
        target_levels=[1, 2, 3],  # All levels to test routing
        include_files=True,
        include_images=False
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

def compare_agent_configs(
    configs: List[str] = None,
    questions_per_config: int = 15
) -> pd.DataFrame:
    """Compare multiple agent configurations"""
    
    if configs is None:
        configs = ["groq", "google", "performance", "accuracy"]
    
    print(f"🔄 Comparing {len(configs)} configurations")
    print(f"📊 {questions_per_config} questions per config")
    print("=" * 50)
    
    comparison_results = []
    
    for config in configs:
        print(f"\n🧪 Testing {config}...")
        
        result = run_gaia_test(
            agent_config_name=config,
            max_questions=questions_per_config,
            target_levels=[1, 2],  # Focus on achievable levels
            include_files=True,
            include_images=True
        )
        
        if result and 'evaluation_metadata' in result:
            metadata = result['evaluation_metadata']
            analysis = result.get('analysis', {})
            overall = analysis.get('overall_performance', {})
            strategy_perf = analysis.get('strategy_performance', {})
            
            # Calculate average execution time
            avg_time = 0
            if strategy_perf:
                total_time = sum(stats.get('avg_execution_time', 0) * stats.get('total_questions', 0) 
                               for stats in strategy_perf.values())
                total_questions = sum(stats.get('total_questions', 0) for stats in strategy_perf.values())
                avg_time = total_time / total_questions if total_questions > 0 else 0
            
            comparison_results.append({
                'config': config,
                'total_questions': metadata.get('total_questions', 0),
                'correct_answers': metadata.get('correct_answers', 0),
                'accuracy': metadata.get('overall_accuracy', 0),
                'avg_execution_time': avg_time,
                'gaia_target_met': overall.get('gaia_target_met', False)
            })
        else:
            comparison_results.append({
                'config': config,
                'total_questions': 0,
                'correct_answers': 0,
                'accuracy': 0.0,
                'avg_execution_time': 0.0,
                'gaia_target_met': False,
                'error': 'Test failed'
            })
    
    df = pd.DataFrame(comparison_results)
    
    print(f"\n📊 CONFIGURATION COMPARISON")
    print("=" * 60)
    print(df.round(3).to_string(index=False))
    
    # Find best configuration
    if not df.empty:
        best_accuracy = df.loc[df['accuracy'].idxmax()]
        fastest_config = df.loc[df['avg_execution_time'].idxmin()]
        
        print(f"\n🏆 RECOMMENDATIONS:")
        print(f"├── Best Accuracy: {best_accuracy['config']} ({best_accuracy['accuracy']:.1%})")
        print(f"└── Fastest: {fastest_config['config']} ({fastest_config['avg_execution_time']:.1f}s avg)")
    
    return df

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
        agent_config_name=agent_config_name,
        max_questions=15,
        target_levels=[1, 2],
        include_files=False,
        include_images=False
    )
    
    if text_result:
        results['text_only'] = text_result
        metadata = text_result.get('evaluation_metadata', {})
        print(f"✅ Text-only: {metadata.get('overall_accuracy', 0):.1%} accuracy")
    
    # Test 2: File-based questions
    print(f"\n📎 Test 2: File-Based Questions")
    print("-" * 30)
    
    file_result = run_gaia_test(
        agent_config_name=agent_config_name,
        max_questions=15,
        target_levels=[1, 2],
        include_files=True,
        include_images=True
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
    
    if not evaluation_results or 'detailed_results' not in evaluation_results:
        print("❌ No evaluation results to analyze")
        return {}
    
    detailed_results = evaluation_results['detailed_results']
    incorrect_results = [r for r in detailed_results if not r.get('is_correct', False)]
    
    print(f"\n🔍 FAILURE PATTERN ANALYSIS")
    print("=" * 40)
    print(f"📊 Total Questions: {len(detailed_results)}")
    print(f"❌ Incorrect Answers: {len(incorrect_results)}")
    
    if len(incorrect_results) == 0:
        print("🎉 No failures to analyze - perfect performance!")
        return {'perfect_performance': True}
    
    # Analyze failure patterns
    failure_patterns = {
        'by_level': defaultdict(int),
        'by_strategy': defaultdict(int),
        'by_file_type': defaultdict(int),
        'execution_failures': 0,
        'low_complexity_failures': 0,
        'high_complexity_failures': 0
    }
    
    for result in incorrect_results:
        # Level pattern
        level = result.get('level', 'unknown')
        failure_patterns['by_level'][level] += 1
        
        # Strategy pattern
        strategy = result.get('strategy_used', 'unknown')
        failure_patterns['by_strategy'][strategy] += 1
        
        # File type pattern
        if result.get('has_file', False):
            file_name = result.get('file_name', '')
            if file_name:
                ext = Path(file_name).suffix.lower()
                failure_patterns['by_file_type'][ext or 'no_extension'] += 1
        else:
            failure_patterns['by_file_type']['no_file'] += 1
        
        # Execution issues
        if not result.get('execution_successful', True):
            failure_patterns['execution_failures'] += 1
        
        # Complexity detection issues
        complexity = result.get('complexity_detected', '')
        if complexity == 'simple':
            failure_patterns['low_complexity_failures'] += 1
        elif complexity == 'complex':
            failure_patterns['high_complexity_failures'] += 1
    
    # Print analysis
    print(f"\n📈 Failure Breakdown:")
    print(f"├── By Level:")
    for level, count in failure_patterns['by_level'].items():
        percentage = count / len(incorrect_results) * 100
        print(f"│   ├── Level {level}: {count} ({percentage:.1f}%)")
    
    print(f"├── By Strategy:")
    for strategy, count in failure_patterns['by_strategy'].items():
        percentage = count / len(incorrect_results) * 100
        print(f"│   ├── {strategy}: {count} ({percentage:.1f}%)")
    
    print(f"├── By File Type:")
    for file_type, count in failure_patterns['by_file_type'].items():
        percentage = count / len(incorrect_results) * 100
        print(f"│   ├── {file_type}: {count} ({percentage:.1f}%)")
    
    print(f"└── Execution Issues:")
    print(f"    ├── Execution Failures: {failure_patterns['execution_failures']}")
    print(f"    ├── Simple Question Failures: {failure_patterns['low_complexity_failures']}")
    print(f"    └── Complex Question Failures: {failure_patterns['high_complexity_failures']}")
    
    # Generate improvement recommendations
    recommendations = []
    
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
    
    # Complexity-based recommendations
    if failure_patterns['low_complexity_failures'] > failure_patterns['high_complexity_failures']:
        recommendations.append("Simple questions failing - improve basic reasoning and routing")
    
    # File-based recommendations
    file_failures = failure_patterns['by_file_type']
    if file_failures.get('.png', 0) > 0 or file_failures.get('.jpg', 0) > 0:
        recommendations.append("Image processing needs improvement - enhance visual analysis")
    if file_failures.get('.xlsx', 0) > 0 or file_failures.get('.csv', 0) > 0:
        recommendations.append("Spreadsheet analysis needs improvement - enhance data processing")
    if file_failures.get('.pdf', 0) > 0:
        recommendations.append("PDF processing needs improvement - enhance document analysis")
    
    # Execution-based recommendations
    if failure_patterns['execution_failures'] > len(incorrect_results) * 0.2:
        recommendations.append("High execution failure rate - improve error handling and stability")
    
    print(f"\n💡 Improvement Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Show sample failures for manual inspection
    print(f"\n🔍 Sample Failure Cases:")
    sample_failures = incorrect_results[:3]
    for i, failure in enumerate(sample_failures, 1):
        question = failure.get('question', '')[:60] + "..."
        agent_answer = failure.get('agent_answer', '')
        expected = failure.get('expected_answer', '')
        strategy = failure.get('strategy_used', '')
        
        print(f"  {i}. Question: {question}")
        print(f"     Agent: '{agent_answer}'")
        print(f"     Expected: '{expected}'")
        print(f"     Level: {failure.get('level')}, Strategy: {strategy}")
        if failure.get('error'):
            print(f"     Error: {failure['error']}")
        print()
    
    return {
        'failure_patterns': failure_patterns,
        'recommendations': recommendations,
        'sample_failures': sample_failures[:5]
    }

def generate_test_report(evaluation_results: Dict, 
                        agent_config_name: str,
                        save_to_file: bool = True) -> str:
    """Generate comprehensive test report"""
    
    if not evaluation_results:
        return "No evaluation results available for report generation"
    
    metadata = evaluation_results.get('evaluation_metadata', {})
    analysis = evaluation_results.get('analysis', {})
    overall = analysis.get('overall_performance', {})
    
    # Generate report
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""
🎯 GAIA AGENT TEST REPORT
{'='*60}

📅 Generated: {timestamp}
🤖 Agent Configuration: {agent_config_name}
🔬 Testing Framework: Two-Step (Blind Execution + Ground Truth Evaluation)

📊 EXECUTIVE SUMMARY
{'-'*30}
✨ Overall Performance: {overall.get('overall_accuracy', 0):.1%}
📝 Questions Tested: {overall.get('total_questions', 0)}
✅ Correct Answers: {overall.get('correct_answers', 0)}
❌ Incorrect Answers: {overall.get('incorrect_answers', 0)}
🏆 GAIA Target (45%): {'✅ ACHIEVED' if overall.get('gaia_target_met', False) else '❌ NOT ACHIEVED'}

📈 PERFORMANCE BY GAIA LEVEL
{'-'*30}"""

    level_performance = analysis.get('level_performance', {})
    for level_key in sorted(level_performance.keys()):
        level_num = level_key.replace('level_', '')
        level_data = level_performance[level_key]
        accuracy = level_data.get('accuracy', 0)
        total = level_data.get('total_questions', 0)
        correct = level_data.get('correct_answers', 0)
        avg_time = level_data.get('avg_execution_time', 0)
        
        report += f"\n🔸 Level {level_num}: {accuracy:.1%} accuracy ({correct}/{total}) - {avg_time:.1f}s avg"

    # Strategy performance
    strategy_perf = analysis.get('strategy_performance', {})
    if strategy_perf:
        report += f"\n\n🎯 STRATEGY PERFORMANCE\n{'-'*30}"
        
        for strategy, stats in strategy_perf.items():
            accuracy = stats.get('accuracy', 0)
            total = stats.get('total_questions', 0)
            avg_time = stats.get('avg_execution_time', 0)
            report += f"\n🔸 {strategy.replace('_', ' ').title()}: {accuracy:.1%} ({total} questions) - {avg_time:.1f}s"

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

    # File attachment analysis
    file_perf = analysis.get('file_attachment_performance', {})
    if file_perf:
        report += f"\n\n📎 FILE ATTACHMENT PERFORMANCE\n{'-'*30}"
        
        for category, stats in file_perf.items():
            accuracy = stats.get('accuracy', 0)
            total = stats.get('total_questions', 0)
            report += f"\n🔸 {category.replace('_', ' ').title()}: {accuracy:.1%} ({total} questions)"

    # Error analysis
    error_analysis = analysis.get('error_analysis', {})
    if error_analysis:
        report += f"\n\n🔧 SYSTEM RELIABILITY\n{'-'*30}"
        success_rate = error_analysis.get('execution_success_rate', 0)
        exec_errors = error_analysis.get('execution_errors', 0)
        
        report += f"\n🔸 Execution Success Rate: {success_rate:.1%}"
        report += f"\n🔸 Execution Errors: {exec_errors}"

    # Recommendations
    report += f"\n\n💡 KEY RECOMMENDATIONS\n{'-'*30}"
    
    if overall.get('overall_accuracy', 0) >= 0.45:
        report += "\n✅ System meets GAIA performance targets"
        report += "\n🚀 Ready for production deployment"
    elif overall.get('overall_accuracy', 0) >= 0.35:
        report += "\n⚠️ Performance approaching targets - optimization recommended"
    else:
        report += "\n❌ Performance below targets - significant improvements needed"

    report += f"\n\n{'='*60}"
    report += f"\nGenerated by GAIA Testing Framework v2.0"
    
    # Save to file if requested
    if save_to_file:
        report_file = Path("logs") / create_timestamped_filename(f"gaia_test_report_{agent_config_name}", "txt").split('/')[-1]
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Test report saved: {report_file}")
    
    return report

# ============================================================================
# MAIN EXECUTION AND EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("🎯 GAIA Testing Framework v2.0")
    print("=" * 50)
    
    # Available test functions
    test_options = {
        "quick_test": "Quick 5-question validation",
        "standard_test": "Standard 20-question evaluation", 
        "routing_test": "Test smart routing behavior",
        "comparison": "Compare multiple agent configurations",
        "file_test": "Test file vs text performance",
        "custom": "Custom test parameters"
    }
    
    print("\n📋 Available Tests:")
    for key, description in test_options.items():
        print(f"  ├── {key}: {description}")
    
    print(f"\n💡 Usage Examples:")
    print(f"  ├── run_quick_gaia_test('groq')")
    print(f"  ├── run_gaia_test('groq', max_questions=30)")
    print(f"  ├── run_smart_routing_test('performance')")
    print(f"  ├── compare_agent_configs(['groq', 'google', 'performance'])")
    print(f"  ├── test_file_vs_text_performance('groq')")
    print(f"  └── analyze_failure_patterns(result)")
    
    # Run demonstration
    print(f"\n🚀 Running quick demonstration...")
    try:
        demo_result = run_quick_gaia_test("groq")
        
        if demo_result and 'evaluation_metadata' in demo_result:
            metadata = demo_result['evaluation_metadata']
            accuracy = metadata.get('overall_accuracy', 0)
            total = metadata.get('total_questions', 0)
            
            print(f"✅ Demo completed!")
            print(f"   ├── Questions: {total}")
            print(f"   ├── Accuracy: {accuracy:.1%}")
            print(f"   └── GAIA Target: {'✅ Met' if accuracy >= 0.45 else '❌ Not Met'}")
            
            # Generate demo report
            generate_test_report(demo_result, "groq")
            
        else:
            print(f"❌ Demo failed - check your setup")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print(f"💡 Ensure metadata.jsonl is available and agent system is properly configured")
    
    print(f"\n🎓 GAIA Testing Framework Ready!")
    print(f"This framework provides rigorous testing for your GAIA agent system.")