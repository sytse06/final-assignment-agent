# gaia_blind_testing.py
# Clean two-step testing framework: 1) Execute questions 2) Evaluate results

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

# Import your production system
from gaia_agent_system import (
    create_gaia_agent, 
    create_production_gaia_agent,
    GAIAConfig,
    ModelConfigs
)

# ============================================================================
# TWO-STEP TESTING CONFIGURATION
# ============================================================================

@dataclass
class TwoStepTestConfig:
    """Configuration for clean two-step testing"""
    # Execution settings
    max_questions: int = 50
    timeout_per_question: int = 180
    save_intermediate: bool = True
    
    # Output settings
    results_dir: str = "gaia_two_step_results"
    execution_file: str = "question_executions.jsonl"
    evaluation_file: str = "evaluation_results.json"
    
    # Safety settings
    enable_retries: bool = True
    max_retries: int = 2
    
    # Content filtering
    include_file_questions: bool = True
    include_image_questions: bool = True
    target_levels: List[int] = None  # None = all levels

# ============================================================================
# STEP 1: QUESTION EXECUTION (BLIND)
# ============================================================================

class GAIAQuestionExecutor:
    """Step 1: Execute questions without knowing the ground truth"""
    
    def __init__(self, config: TwoStepTestConfig):
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
            is_image = file_name.endswith(('.png', '.jpg', '.jpeg'))
            if is_image and not self.config.include_image_questions:
                continue
            
            # Create blind question (remove ground truth)
            blind_question = {
                'task_id': question_data.get('task_id'),
                'question': question_data.get('Question'),
                'level': question_data.get('Level'),
                'file_name': question_data.get('file_name', ''),
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
    
    def execute_questions_batch(self, 
                               agent_config: str = "qwen2.5_coder",
                               questions: List[Dict] = None) -> str:
        """Step 1: Execute all questions and save results"""
        
        if questions is None:
            questions = self.load_questions_blind()
        
        if not questions:
            print("❌ No questions to execute")
            return None
        
        print(f"\n🚀 Starting Question Execution")
        print(f"📝 Agent Config: {agent_config}")
        print(f"📊 Questions: {len(questions)}")
        print("=" * 60)
        
        # Create agent
        try:
            agent = create_production_gaia_agent(
                model_config=agent_config,
                enable_logging=True,
                performance_tracking=True,
                max_retries=self.config.max_retries
            )
            print(f"✅ Agent created: {agent_config}")
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            return None
        
        # Execution log file
        execution_file = self.results_dir / f"executions_{self.session_id}.jsonl"
        
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
                        print(f"✅ Answer: {answer}")
                        print(f"⏱️  Time: {duration:.2f}s")
                    else:
                        error = execution_result.get('error', 'Unknown error')
                        print(f"❌ Error: {error}")
                    
                    # Add to session log
                    self.execution_log.append(execution_result)
        
        except Exception as e:
            print(f"❌ Execution failed: {e}")
            return None
        
        finally:
            # Clean up agent
            try:
                agent.close()
            except:
                pass
        
        print(f"\n🎉 Execution Complete!")
        print(f"📁 Results saved: {execution_file}")
        print(f"📊 Total executed: {len(self.execution_log)}")
        
        return str(execution_file)
    
    def _execute_single_question(self, agent, question_data: Dict, attempt: int = 1) -> Dict:
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
            'is_image_question': question_data['is_image_question'],
            'attempt': attempt,
            'timestamp': datetime.now().isoformat(),
            'agent_config': "production",  # Could be passed in
        }
        
        try:
            # Execute with agent
            result = agent.run_single_question(
                question=question_data['question'],
                task_id=task_id,
                ground_truth=None,  # Deliberately not provided
                level=question_data['level']
            )
            
            execution_time = time.time() - start_time
            
            # Extract agent response
            agent_answer = result.get('final_answer', '')
            raw_answer = result.get('raw_answer', '')
            strategy_used = result.get('selected_strategy', '')
            agent_used = result.get('selected_agent', '')
            confidence = result.get('confidence_score', 0.0)
            errors = result.get('errors', [])
            fallback_used = result.get('fallback_used', False)
            
            execution_record.update({
                'success': True,
                'agent_answer': agent_answer,
                'raw_answer': raw_answer,
                'strategy_used': strategy_used,
                'agent_used': agent_used,
                'confidence_score': confidence,
                'execution_time': execution_time,
                'errors': errors,
                'fallback_used': fallback_used,
                'execution_steps': result.get('execution_steps', [])
            })
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            execution_record.update({
                'success': False,
                'error': error_msg,
                'execution_time': execution_time,
                'agent_answer': '',
                'errors': [error_msg]
            })
            
            # Retry logic
            if attempt < self.config.max_retries and self.config.enable_retries:
                print(f"    🔄 Retry attempt {attempt + 1}")
                return self._execute_single_question(agent, question_data, attempt + 1)
        
        return execution_record
    
    def _generate_sample_questions(self) -> List[Dict]:
        """Generate sample questions if metadata not available"""
        sample_questions = [
            {
                'task_id': f'sample_{i}',
                'question': f'Sample question {i}: Calculate {10 + i} * {20 + i}',
                'level': 1 if i < 3 else 2,
                'file_name': '',
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
    
    def __init__(self, config: TwoStepTestConfig):
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
                        'annotator_metadata': item.get('Annotator Metadata', {})
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
                is_correct = self._evaluate_answer_match(agent_answer, expected_answer)
                correct_count += is_correct
                
                print(f"🎯 Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                
                evaluation_result = self._create_evaluation_record(
                    execution, gt_data, is_correct
                )
            
            evaluation_results.append(evaluation_result)
        
        # Generate comprehensive analysis
        analysis = self._generate_evaluation_analysis(evaluation_results, correct_count)
        
        # Save evaluation results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_file = self.results_dir / f"evaluation_{timestamp}.json"
        
        final_results = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'execution_file': execution_file,
                'total_questions': len(evaluation_results),
                'correct_answers': correct_count,
                'overall_accuracy': correct_count / len(evaluation_results) if evaluation_results else 0
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
            
            # Execution data
            'agent_answer': execution.get('agent_answer', ''),
            'raw_answer': execution.get('raw_answer', ''),
            'strategy_used': execution.get('strategy_used', ''),
            'agent_used': execution.get('agent_used', ''),
            'confidence_score': execution.get('confidence_score', 0.0),
            'execution_time': execution.get('execution_time', 0.0),
            'execution_successful': execution.get('success', False),
            'fallback_used': execution.get('fallback_used', False),
            'errors': execution.get('errors', []),
            
            # Ground truth data
            'expected_answer': ground_truth.get('final_answer', '') if ground_truth else '',
            'is_correct': is_correct,
            'note': note,
            
            # Evaluation metadata
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def _evaluate_answer_match(self, predicted: str, expected: str) -> bool:
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
        
        # Numeric comparison
        try:
            pred_num = float(pred_clean.replace(',', ''))
            exp_num = float(exp_clean.replace(',', ''))
            return abs(pred_num - exp_num) < 1e-6
        except ValueError:
            pass
        
        # List comparison (comma-separated)
        if ',' in pred_clean and ',' in exp_clean:
            pred_list = [item.strip() for item in pred_clean.split(',')]
            exp_list = [item.strip() for item in exp_clean.split(',')]
            return pred_list == exp_list
        
        # Partial match for very close answers
        if len(exp_clean) > 5:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, pred_clean, exp_clean).ratio()
            return similarity > 0.9
        
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
                'gaia_target_met': overall_accuracy >= 0.45
            }
        }
        
        # Performance by level
        level_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for result in evaluation_results:
            level = result.get('level', 1)
            level_stats[level]['total'] += 1
            if result.get('is_correct', False):
                level_stats[level]['correct'] += 1
        
        analysis['level_performance'] = {}
        for level, stats in level_stats.items():
            analysis['level_performance'][f'level_{level}'] = {
                'total_questions': stats['total'],
                'correct_answers': stats['correct'],
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0
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
        
        # Strategy performance
        strategy_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for result in evaluation_results:
            strategy = result.get('strategy_used', 'unknown')
            strategy_stats[strategy]['total'] += 1
            if result.get('is_correct', False):
                strategy_stats[strategy]['correct'] += 1
        
        analysis['strategy_performance'] = {}
        for strategy, stats in strategy_stats.items():
            analysis['strategy_performance'][strategy] = {
                'total_questions': stats['total'],
                'correct_answers': stats['correct'],
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            }
        
        # Error analysis
        error_questions = [r for r in evaluation_results if not r.get('execution_successful', True)]
        fallback_questions = [r for r in evaluation_results if r.get('fallback_used', False)]
        
        analysis['error_analysis'] = {
            'execution_errors': len(error_questions),
            'fallback_usage': len(fallback_questions),
            'execution_success_rate': 1 - (len(error_questions) / total_questions),
            'sample_errors': [r.get('errors', []) for r in error_questions[:3]]
        }
        
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
                print(f"├── Level {level_num}: {accuracy:.1%} ({correct}/{total})")
        
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
            fallback_usage = error_analysis.get('fallback_usage', 0)
            success_rate = error_analysis.get('execution_success_rate', 0)
            
            print(f"\n🔧 Error Analysis:")
            print(f"├── Execution Success Rate: {success_rate:.1%}")
            print(f"├── Execution Errors: {exec_errors}")
            print(f"└── Fallback Usage: {fallback_usage}")

# ============================================================================
# CONVENIENCE FUNCTIONS FOR TWO-STEP TESTING
# ============================================================================

def run_two_step_gaia_test(
    agent_config: str = "qwen2.5_coder",
    max_questions: int = 20,
    target_levels: List[int] = None,
    include_files: bool = True,
    include_images: bool = True
) -> Dict:
    """Complete two-step GAIA test: execute then evaluate"""
    
    print(f"🚀 Two-Step GAIA Test")
    print(f"🤖 Agent: {agent_config}")
    print(f"📊 Max Questions: {max_questions}")
    print("=" * 50)
    
    # Configure testing
    config = TwoStepTestConfig(
        max_questions=max_questions,
        target_levels=target_levels,
        include_file_questions=include_files,
        include_image_questions=include_images
    )
    
    # Step 1: Execute questions (blind)
    print(f"\n📋 STEP 1: Question Execution")
    print("-" * 30)
    
    executor = GAIAQuestionExecutor(config)
    execution_file = executor.execute_questions_batch(agent_config)
    
    if not execution_file:
        print("❌ Execution failed")
        return {}
    
    # Step 2: Evaluate answers (with ground truth)
    print(f"\n🎯 STEP 2: Answer Evaluation")
    print("-" * 30)
    
    evaluator = GAIAAnswerEvaluator(config)
    evaluation_results = evaluator.evaluate_execution_results(execution_file)
    
    return evaluation_results

def run_quick_two_step_test(agent_config: str = "qwen2.5_coder") -> Dict:
    """Quick two-step test for development"""
    
    return run_two_step_gaia_test(
        agent_config=agent_config,
        max_questions=5,
        target_levels=[1],  # Level 1 only
        include_files=True,
        include_images=False  # Skip images for quick test
    )

def compare_agent_configs_two_step(
    configs: List[str] = None,
    questions_per_config: int = 10
) -> pd.DataFrame:
    """Compare multiple agent configurations using two-step testing"""
    
    if configs is None:
        configs = ["qwen2.5_coder", "qwen_qwq_groq", "gemini_flash_04"]
    
    print(f"🔄 Comparing {len(configs)} configurations")
    print(f"📊 {questions_per_config} questions per config")
    
    comparison_results = []
    
    for config in configs:
        print(f"\n🧪 Testing {config}...")
        
        result = run_two_step_gaia_test(
            agent_config=config,
            max_questions=questions_per_config,
            target_levels=[1, 2],  # Focus on achievable levels
            include_files=True,
            include_images=True
        )
        
        if result and 'evaluation_metadata' in result:
            metadata = result['evaluation_metadata']
            analysis = result.get('analysis', {})
            overall = analysis.get('overall_performance', {})
            
            comparison_results.append({
                'config': config,
                'total_questions': metadata.get('total_questions', 0),
                'correct_answers': metadata.get('correct_answers', 0),
                'accuracy': metadata.get('overall_accuracy', 0),
                'gaia_target_met': overall.get('gaia_target_met', False)
            })
        else:
            comparison_results.append({
                'config': config,
                'total_questions': 0,
                'correct_answers': 0,
                'accuracy': 0.0,
                'gaia_target_met': False,
                'error': 'Test failed'
            })
    
    df = pd.DataFrame(comparison_results)
    
    print(f"\n📊 CONFIGURATION COMPARISON")
    print("=" * 50)
    print(df.round(3).to_string(index=False))
    
    return df

def test_image_questions_specifically(agent_config: str = "qwen2.5_coder") -> Dict:
    """Test specifically on image-based questions"""
    
    print(f"🖼️ Testing Image-Based GAIA Questions")
    print(f"🤖 Agent: {agent_config}")
    print("=" * 50)
    
    # Load all questions and filter for images
    try:
        with open("metadata.jsonl", 'r', encoding='utf-8') as f:
            all_questions = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("❌ metadata.jsonl not found")
        return {}
    
    # Filter for image questions
    image_questions = []
    for q in all_questions:
        file_name = q.get('file_name', '').lower()
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_questions.append(q)
    
    print(f"📊 Found {len(image_questions)} image-based questions")
    
    if len(image_questions) == 0:
        print("⚠️ No image questions found in dataset")
        return {
            'total_image_questions': 0,
            'message': 'No image questions available for testing'
        }
    
    # Show image question details
    print(f"\n📋 Image Questions Overview:")
    for i, q in enumerate(image_questions):
        level = q.get('Level', 'Unknown')
        file_name = q.get('file_name', '')
        question_preview = q.get('Question', '')[:80] + "..."
        print(f"├── {i+1}. Level {level} | {file_name}")
        print(f"│   Question: {question_preview}")
    
    # Run two-step test on image questions only
    config = TwoStepTestConfig(
        max_questions=len(image_questions),
        target_levels=None,  # All levels
        include_file_questions=True,
        include_image_questions=True
    )
    
    # Execute questions
    executor = GAIAQuestionExecutor(config)
    
    # Convert to blind questions format
    blind_questions = []
    for q in image_questions:
        file_name = q.get('file_name', '')
        blind_q = {
            'task_id': q.get('task_id'),
            'question': q.get('Question'),
            'level': q.get('Level'),
            'file_name': file_name,
            'has_file': bool(file_name.strip()),
            'is_image_question': True
        }
        blind_questions.append(blind_q)
    
    execution_file = executor.execute_questions_batch(agent_config, blind_questions)
    
    if not execution_file:
        print("❌ Image question execution failed")
        return {}
    
    # Evaluate results
    evaluator = GAIAAnswerEvaluator(config)
    evaluation_results = evaluator.evaluate_execution_results(execution_file)
    
    # Add image-specific analysis
    if evaluation_results and 'analysis' in evaluation_results:
        analysis = evaluation_results['analysis']
        overall = analysis.get('overall_performance', {})
        
        print(f"\n🖼️ IMAGE QUESTION ANALYSIS")
        print("=" * 40)
        print(f"📊 Total Image Questions: {overall.get('total_questions', 0)}")
        print(f"✅ Correct: {overall.get('correct_answers', 0)}")
        print(f"🎯 Image Accuracy: {overall.get('overall_accuracy', 0):.1%}")
        print(f"🏆 Image Performance: {'✅ Good' if overall.get('overall_accuracy', 0) > 0.3 else '⚠️ Needs Improvement'}")
    
    return evaluation_results

def test_file_attachment_performance(agent_config: str = "qwen2.5_coder") -> Dict:
    """Test performance on questions with vs without file attachments"""
    
    print(f"📎 Testing File Attachment Performance")
    print(f"🤖 Agent: {agent_config}")
    print("=" * 50)
    
    # Run separate tests for file vs non-file questions
    results = {}
    
    # Test 1: Questions WITH file attachments
    print(f"\n📋 Test 1: Questions WITH File Attachments")
    print("-" * 40)
    
    with_files_result = run_two_step_gaia_test(
        agent_config=agent_config,
        max_questions=15,
        target_levels=[1, 2],
        include_files=True,
        include_images=True
    )
    
    if with_files_result:
        results['with_files'] = with_files_result
        metadata = with_files_result.get('evaluation_metadata', {})
        print(f"✅ With Files: {metadata.get('overall_accuracy', 0):.1%} accuracy")
    
    # Test 2: Questions WITHOUT file attachments  
    print(f"\n📋 Test 2: Questions WITHOUT File Attachments")
    print("-" * 40)
    
    without_files_result = run_two_step_gaia_test(
        agent_config=agent_config,
        max_questions=15,
        target_levels=[1, 2],
        include_files=False,
        include_images=False
    )
    
    if without_files_result:
        results['without_files'] = without_files_result
        metadata = without_files_result.get('evaluation_metadata', {})
        print(f"✅ Without Files: {metadata.get('overall_accuracy', 0):.1%} accuracy")
    
    # Compare results
    if 'with_files' in results and 'without_files' in results:
        with_acc = results['with_files']['evaluation_metadata']['overall_accuracy']
        without_acc = results['without_files']['evaluation_metadata']['overall_accuracy']
        
        print(f"\n📊 FILE ATTACHMENT COMPARISON")
        print("=" * 40)
        print(f"📎 With Files: {with_acc:.1%}")
        print(f"📝 Without Files: {without_acc:.1%}")
        print(f"📈 Difference: {(without_acc - with_acc)*100:+.1f} percentage points")
        
        if with_acc < without_acc:
            print(f"⚠️ File processing may need improvement")
        else:
            print(f"✅ File processing performance is good")
    
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
        'fallback_failures': 0,
        'low_confidence_failures': 0
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
                failure_patterns['by_file_type'][ext] += 1
        else:
            failure_patterns['by_file_type']['no_file'] += 1
        
        # Execution issues
        if not result.get('execution_successful', True):
            failure_patterns['execution_failures'] += 1
        
        if result.get('fallback_used', False):
            failure_patterns['fallback_failures'] += 1
        
        if result.get('confidence_score', 1.0) < 0.5:
            failure_patterns['low_confidence_failures'] += 1
    
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
    print(f"    ├── Fallback Usage: {failure_patterns['fallback_failures']}")
    print(f"    └── Low Confidence: {failure_patterns['low_confidence_failures']}")
    
    # Generate improvement recommendations
    recommendations = []
    
    # Level-based recommendations
    level_failures = failure_patterns['by_level']
    if level_failures.get(1, 0) > level_failures.get(2, 0):
        recommendations.append("Focus on improving Level 1 performance - basic capabilities need strengthening")
    if level_failures.get(3, 0) > 0:
        recommendations.append("Level 3 questions are challenging - consider advanced reasoning techniques")
    
    # Strategy-based recommendations  
    strategy_failures = failure_patterns['by_strategy']
    if strategy_failures.get('direct_llm', 0) > strategy_failures.get('smolag_agent', 0):
        recommendations.append("Direct LLM failing more than SmolagAgent - improve prompt engineering")
    elif strategy_failures.get('smolag_agent', 0) > strategy_failures.get('direct_llm', 0):
        recommendations.append("SmolagAgent failing more - check tool integration and agent reliability")
    
    # File-based recommendations
    file_failures = failure_patterns['by_file_type']
    if file_failures.get('.png', 0) > 0 or file_failures.get('.jpg', 0) > 0:
        recommendations.append("Image processing needs improvement - enhance visual analysis capabilities")
    if file_failures.get('.xlsx', 0) > 0:
        recommendations.append("Spreadsheet analysis needs improvement - enhance data processing tools")
    
    # Execution-based recommendations
    if failure_patterns['execution_failures'] > len(incorrect_results) * 0.2:
        recommendations.append("High execution failure rate - improve error handling and stability")
    if failure_patterns['fallback_failures'] > len(incorrect_results) * 0.3:
        recommendations.append("High fallback usage - investigate SmolagAgent reliability issues")
    
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
        
        print(f"  {i}. Question: {question}")
        print(f"     Agent: '{agent_answer}'")
        print(f"     Expected: '{expected}'")
        print(f"     Level: {failure.get('level')}, Strategy: {failure.get('strategy_used')}")
        if failure.get('errors'):
            print(f"     Errors: {failure['errors'][:1]}")  # Show first error
        print()
    
    return {
        'failure_patterns': failure_patterns,
        'recommendations': recommendations,
        'sample_failures': sample_failures[:5]  # Store more samples
    }

def generate_final_report(evaluation_results: Dict, 
                         agent_config: str,
                         save_to_file: bool = True) -> str:
    """Generate comprehensive final report for stakeholders"""
    
    if not evaluation_results:
        return "No evaluation results available for report generation"
    
    metadata = evaluation_results.get('evaluation_metadata', {})
    analysis = evaluation_results.get('analysis', {})
    overall = analysis.get('overall_performance', {})
    
    # Generate executive summary
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""
🎯 GAIA AGENT EVALUATION REPORT
{'='*60}

📅 Generated: {timestamp}
🤖 Agent Configuration: {agent_config}
🔬 Testing Methodology: Two-Step (Blind Execution + Ground Truth Evaluation)

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
        
        report += f"\n🔸 Level {level_num}: {accuracy:.1%} accuracy ({correct}/{total} questions)"
        
        # Level-specific insights
        if level_num == '1' and accuracy < 0.6:
            report += " ⚠️ Below expected Level 1 performance"
        elif level_num == '2' and accuracy > 0.4:
            report += " ✅ Strong Level 2 performance"
        elif level_num == '3' and accuracy > 0.2:
            report += " 🌟 Excellent Level 3 performance"

    # File attachment analysis
    file_perf = analysis.get('file_attachment_performance', {})
    if file_perf:
        report += f"\n\n📎 FILE ATTACHMENT PERFORMANCE\n{'-'*30}"
        
        with_files = file_perf.get('with_files', {})
        without_files = file_perf.get('without_files', {})
        
        if with_files and without_files:
            with_acc = with_files.get('accuracy', 0)
            without_acc = without_files.get('accuracy', 0)
            
            report += f"\n🔸 With Files: {with_acc:.1%} ({with_files.get('total_questions', 0)} questions)"
            report += f"\n🔸 Without Files: {without_acc:.1%} ({without_files.get('total_questions', 0)} questions)"
            
            if with_acc < without_acc - 0.1:
                report += "\n⚠️ File processing capabilities need improvement"
            else:
                report += "\n✅ Good file processing performance"

    # Strategy analysis
    strategy_perf = analysis.get('strategy_performance', {})
    if strategy_perf:
        report += f"\n\n🎯 STRATEGY PERFORMANCE\n{'-'*30}"
        
        for strategy, stats in strategy_perf.items():
            if stats.get('total_questions', 0) > 0:
                accuracy = stats.get('accuracy', 0)
                total = stats.get('total_questions', 0)
                report += f"\n🔸 {strategy.replace('_', ' ').title()}: {accuracy:.1%} ({total} questions)"

    # Error analysis
    error_analysis = analysis.get('error_analysis', {})
    if error_analysis:
        report += f"\n\n🔧 SYSTEM RELIABILITY\n{'-'*30}"
        success_rate = error_analysis.get('execution_success_rate', 0)
        exec_errors = error_analysis.get('execution_errors', 0)
        fallback_usage = error_analysis.get('fallback_usage', 0)
        
        report += f"\n🔸 Execution Success Rate: {success_rate:.1%}"
        report += f"\n🔸 Execution Errors: {exec_errors}"
        report += f"\n🔸 Fallback Usage: {fallback_usage}"
        
        if success_rate > 0.95:
            report += "\n✅ Excellent system reliability"
        elif success_rate > 0.85:
            report += "\n⚠️ Good reliability, some improvements possible"
        else:
            report += "\n❌ Reliability issues need attention"

    # Recommendations
    report += f"\n\n💡 KEY RECOMMENDATIONS\n{'-'*30}"
    
    if overall.get('overall_accuracy', 0) >= 0.45:
        report += "\n✅ System meets GAIA performance targets"
        report += "\n🚀 Ready for production deployment with monitoring"
    elif overall.get('overall_accuracy', 0) >= 0.35:
        report += "\n⚠️ Performance approaching targets - focused improvements needed"
        report += "\n🔧 Recommend optimization before production deployment"
    else:
        report += "\n❌ Performance below targets - significant improvements needed"
        report += "\n🔬 Recommend additional development cycle"
    
    # Specific improvement areas
    if level_performance.get('level_1', {}).get('accuracy', 0) < 0.6:
        report += "\n🎯 Priority: Improve Level 1 question handling"
    
    if file_perf:
        with_files_acc = file_perf.get('with_files', {}).get('accuracy', 0)
        if with_files_acc < 0.4:
            report += "\n📎 Priority: Enhance file processing capabilities"
    
    if error_analysis.get('execution_success_rate', 1) < 0.9:
        report += "\n🔧 Priority: Improve system stability and error handling"

    # Next steps
    report += f"\n\n🚀 NEXT STEPS\n{'-'*30}"
    
    if overall.get('gaia_target_met', False):
        report += "\n1. Proceed with final validation testing"
        report += "\n2. Prepare production deployment"
        report += "\n3. Implement monitoring and feedback systems"
    else:
        report += "\n1. Address identified performance gaps"
        report += "\n2. Conduct focused improvement cycle"
        report += "\n3. Re-test with expanded question set"

    report += f"\n\n{'='*60}"
    report += f"\nReport generated by GAIA Two-Step Testing Framework"
    report += f"\nFor detailed results, see: {evaluation_results.get('detailed_results_file', 'N/A')}"
    
    # Save to file if requested
    if save_to_file:
        timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = Path(f"gaia_final_report_{agent_config}_{timestamp_file}.txt")
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Final report saved: {report_file}")
    
    return report

# ============================================================================
# MAIN EXECUTION FOR TWO-STEP TESTING
# ============================================================================

if __name__ == "__main__":
    print("🎯 GAIA Two-Step Testing Framework")
    print("=" * 50)
    
    # Available test functions
    test_options = {
        "quick_test": "Quick 5-question validation",
        "standard_test": "Standard 20-question evaluation", 
        "comparison": "Compare multiple agent configurations",
        "image_test": "Test image processing capabilities",
        "file_test": "Test file attachment performance",
        "custom": "Custom test parameters"
    }
    
    print("\n📋 Available Tests:")
    for key, description in test_options.items():
        print(f"  ├── {key}: {description}")
    
    print(f"\n💡 Usage Examples:")
    print(f"  ├── run_quick_two_step_test('qwen2.5_coder')")
    print(f"  ├── run_two_step_gaia_test('qwen2.5_coder', max_questions=30)")
    print(f"  ├── compare_agent_configs_two_step(['qwen2.5_coder', 'qwen_qwq_groq'])")
    print(f"  ├── test_image_questions_specifically('qwen2.5_coder')")
    print(f"  └── test_file_attachment_performance('qwen2.5_coder')")
    
    # Run demonstration
    print(f"\n🚀 Running quick demonstration...")
    try:
        demo_result = run_quick_two_step_test("qwen2.5_coder")
        
        if demo_result and 'evaluation_metadata' in demo_result:
            metadata = demo_result['evaluation_metadata']
            accuracy = metadata.get('overall_accuracy', 0)
            total = metadata.get('total_questions', 0)
            
            print(f"✅ Demo completed!")
            print(f"   ├── Questions: {total}")
            print(f"   ├── Accuracy: {accuracy:.1%}")
            print(f"   └── GAIA Target: {'✅ Met' if accuracy >= 0.45 else '❌ Not Met'}")
        else:
            print(f"❌ Demo failed - check your setup")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print(f"💡 Ensure gaia_agent_system.py and metadata.jsonl are available")
    
    print(f"\n🎓 Two-Step Testing Framework Ready!")
    print(f"This framework provides clean separation between execution and evaluation for rigorous testing.")