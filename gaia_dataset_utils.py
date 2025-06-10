import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

class GAIADatasetManager:
    """Utility class for working with GAIA dataset files and metadata"""
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize GAIA dataset manager
        
        Args:
            dataset_path: Path to GAIA dataset. If None, tries to find in common locations
        """
        self.dataset_path = self._find_dataset_path(dataset_path)
        self.metadata = None
        self.file_questions = {}
        
        if self.dataset_path:
            self._load_metadata()
    
    def _find_dataset_path(self, provided_path: str = None) -> Optional[str]:
        """Find GAIA dataset path in common locations"""
        if provided_path and os.path.exists(provided_path):
            return provided_path
        
        # Common locations where GAIA dataset might be (prioritize local test directory)
        common_paths = [
            "./tests/gaia_data",              # Your local test directory
            "../tests/gaia_data",
            "./gaia_data",
            "./test_data/gaia",
            "./data/gaia",
            os.path.expanduser("~/.cache/huggingface/datasets/gaia"),
            os.path.expanduser("~/Documents/gaia_dataset")
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                metadata_file = os.path.join(path, "metadata.json")
                if os.path.exists(metadata_file):
                    print(f"✅ Found GAIA dataset at: {path}")
                    return path
        
        print("⚠️  GAIA dataset not found. Please specify path manually.")
        print("💡 Expected structure: tests/gaia_data/metadata.json + data files")
        return None
    
    def _load_metadata(self):
        """Load metadata.json file"""
        metadata_file = os.path.join(self.dataset_path, "metadata.json")
        
        if not os.path.exists(metadata_file):
            print(f"❌ metadata.json not found at {metadata_file}")
            return
        
        try:
            with open(metadata_file, 'r') as f:
                raw_metadata = json.load(f)
            
            # Handle GAIA dataset structure: {'validation': [...], 'test': [...], 'stats': {...}}
            if isinstance(raw_metadata, dict) and 'validation' in raw_metadata:
                print(f"📊 GAIA Dataset Structure Detected:")
                print(f"  Validation set: {len(raw_metadata.get('validation', []))} questions")
                print(f"  Test set: {len(raw_metadata.get('test', []))} questions")
                
                # Use validation set for testing (test set typically doesn't have answers)
                self.metadata = raw_metadata['validation']
                
                # Also store test set if needed
                self.test_metadata = raw_metadata.get('test', [])
                
            elif isinstance(raw_metadata, list):
                self.metadata = raw_metadata
                
            elif isinstance(raw_metadata, dict):
                # Handle other dict structures
                if 'questions' in raw_metadata:
                    self.metadata = raw_metadata['questions']
                elif 'data' in raw_metadata:
                    self.metadata = raw_metadata['data']
                else:
                    # Assume the dict values are the questions
                    self.metadata = list(raw_metadata.values())
            else:
                print(f"❌ Unexpected metadata structure: {type(raw_metadata)}")
                return
            
            # Index questions by task_id for quick lookup
            self.file_questions = {}
            for item in self.metadata:
                if isinstance(item, dict):
                    # Check for file references
                    if ('file_name' in item and item['file_name']) or ('file_path' in item and item['file_path']):
                        self.file_questions[item['task_id']] = item
            
            print(f"✅ Loaded {len(self.metadata)} GAIA questions")
            print(f"📁 Found {len(self.file_questions)} questions with files")
            
            # Show distribution by level
            level_dist = {}
            for item in self.metadata:
                level = item.get('Level', 'Unknown')
                level_dist[level] = level_dist.get(level, 0) + 1
            
            print(f"📊 Level distribution:", end=" ")
            for level in sorted(level_dist.keys()):
                print(f"L{level}: {level_dist[level]}", end="  ")
            print()
            
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            print(f"💡 Check that {metadata_file} contains valid JSON")

    def get_dataset_info(self) -> dict:
        """Get comprehensive dataset information"""
        if not self.metadata:
            return {"error": "No metadata loaded"}
        
        info = {
            "total_questions": len(self.metadata),
            "questions_with_files": len(self.file_questions),
            "level_distribution": {},
            "file_type_distribution": {},
            "has_test_set": hasattr(self, 'test_metadata') and self.test_metadata
        }
        
        # Level distribution
        for item in self.metadata:
            level = item.get('Level', 'Unknown')
            info["level_distribution"][level] = info["level_distribution"].get(level, 0) + 1
        
        # File type distribution
        for question in self.file_questions.values():
            file_name = question.get('file_name', '')
            if file_name:
                ext = file_name.split('.')[-1].lower()
                info["file_type_distribution"][ext] = info["file_type_distribution"].get(ext, 0) + 1
        
        return info
    
    def get_questions_with_files(self, level: int = None, file_type: str = None) -> List[Dict]:
        """
        Get questions that have attached files
        
        Args:
            level: Filter by difficulty level (1, 2, or 3)
            file_type: Filter by file extension (e.g., 'xlsx', 'pdf', 'csv')
        
        Returns:
            List of question dictionaries with file attachments
        """
        if not self.file_questions:
            return []
        
        results = []
        for task_id, question in self.file_questions.items():
            # Filter by level if specified
            if level and str(question.get('Level', '')) != str(level):
                continue
            
            # Filter by file type if specified
            if file_type:
                file_name = question.get('file_name', '')
                if not file_name.lower().endswith(f'.{file_type.lower()}'):
                    continue
            
            results.append(question)
        
        return results
    
    def get_file_types_distribution(self) -> Dict[str, int]:
        """Get distribution of file types in the dataset"""
        file_types = {}
        
        for question in self.file_questions.values():
            file_name = question.get('file_name', '')
            if file_name:
                ext = file_name.split('.')[-1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
        
        return file_types
    
    def setup_test_file(self, task_id: str, target_dir: str = "./test_files") -> Optional[str]:
        """
        Locate a GAIA file using metadata.json file references
        
        Args:
            task_id: The task ID of the question
            target_dir: Directory to organize files (if needed)
        
        Returns:
            Path to the file, or None if not found
        """
        if task_id not in self.file_questions:
            print(f"❌ Task {task_id} not found in file questions")
            return None
        
        question = self.file_questions[task_id]
        file_name = question.get('file_name')
        
        if not file_name:
            print(f"❌ No file name specified for task {task_id}")
            return None
        
        # Strategy 1: Look for exact filename in dataset directory (as specified in metadata)
        exact_file_path = os.path.join(self.dataset_path, file_name)
        if os.path.exists(exact_file_path):
            print(f"✅ Found file by exact name: {exact_file_path}")
            return exact_file_path
        
        # Strategy 2: Look for file using task_id as filename (common organization pattern)
        file_extension = file_name.split('.')[-1] if '.' in file_name else ''
        task_id_filename = f"{task_id}.{file_extension}" if file_extension else task_id
        task_id_file_path = os.path.join(self.dataset_path, task_id_filename)
        
        if os.path.exists(task_id_file_path):
            print(f"✅ Found file by task ID: {task_id_file_path}")
            return task_id_file_path
        
        # Strategy 3: Use original file_path from metadata if it exists (HuggingFace cache)
        original_path = question.get('file_path')
        if original_path and os.path.exists(original_path):
            print(f"✅ Found file in original cache location: {original_path}")
            return original_path
        
        # Strategy 4: Search for any file containing the task_id in the dataset directory
        if os.path.exists(self.dataset_path):
            for file in os.listdir(self.dataset_path):
                if task_id in file:
                    found_path = os.path.join(self.dataset_path, file)
                    if os.path.isfile(found_path):
                        print(f"✅ Found file by task ID pattern: {found_path}")
                        return found_path
        
        # Strategy 5: Search for any file with matching extension containing part of task_id
        if os.path.exists(self.dataset_path) and file_extension:
            for file in os.listdir(self.dataset_path):
                if (file.lower().endswith(f'.{file_extension.lower()}') and 
                    any(part in file for part in task_id.split('-')[:2])):  # Match first 2 parts of UUID
                    found_path = os.path.join(self.dataset_path, file)
                    if os.path.isfile(found_path):
                        print(f"✅ Found file by extension and partial match: {found_path}")
                        return found_path
        
        # Debug information
        print(f"❌ File not found for task {task_id}")
        print(f"   Expected filename: {file_name}")
        print(f"   Expected task ID filename: {task_id_filename}")
        print(f"   Searched in directory: {self.dataset_path}")
        
        if original_path:
            print(f"   Original metadata path: {original_path}")
        
        # List available files for debugging
        if os.path.exists(self.dataset_path):
            available_files = [f for f in os.listdir(self.dataset_path) 
                             if os.path.isfile(os.path.join(self.dataset_path, f))][:10]
            print(f"   Available files (first 10): {available_files}")
        
        return None
    
    def create_small_test_batch(self) -> List[Dict]:
        """
        Create small test batch: 5 questions with diverse file types
        
        Returns:
            List of 5 test cases with different file types
        """
        print("\n📦 Creating Small Test Batch (5 questions)")
        print("=" * 50)
        
        # Target file types for diversity (prioritize common and testable formats)
        target_file_types = ['xlsx', 'csv', 'pdf', 'png', 'txt', 'json', 'py', 'xml']
        
        selected_questions = []
        used_file_types = set()
        
        # Get all file questions
        all_file_questions = list(self.file_questions.values())
        
        # First pass: one question per file type
        for question in all_file_questions:
            if len(selected_questions) >= 5:
                break
                
            file_name = question.get('file_name', '')
            if not file_name:
                continue
                
            file_type = file_name.split('.')[-1].lower()
            
            if file_type in target_file_types and file_type not in used_file_types:
                # Check if file actually exists
                file_path = self.setup_test_file(question['task_id'])
                if file_path:
                    selected_questions.append(question)
                    used_file_types.add(file_type)
                    print(f"✅ Selected {file_type.upper()}: {question['task_id']}")
        
        # Second pass: fill remaining slots with any working files
        for question in all_file_questions:
            if len(selected_questions) >= 5:
                break
                
            if question not in selected_questions:
                file_path = self.setup_test_file(question['task_id'])
                if file_path:
                    selected_questions.append(question)
                    file_type = question.get('file_name', '').split('.')[-1].lower()
                    print(f"✅ Added {file_type.upper()}: {question['task_id']}")
        
        # Create test cases
        test_cases = []
        for question in selected_questions:
            file_path = self.setup_test_file(question['task_id'])
            if file_path:
                test_cases.append({
                    'task_id': question['task_id'],
                    'question': question['Question'],
                    'level': question.get('Level'),
                    'file_path': file_path,
                    'file_name': question.get('file_name'),
                    'file_type': question.get('file_name', '').split('.')[-1].lower(),
                    'expected_answer': question.get('Final answer'),
                    'annotator_steps': question.get('Annotator Metadata', {}).get('Steps'),
                    'expected_tools': question.get('Annotator Metadata', {}).get('Tools'),
                    'batch_type': 'small'
                })
        
        print(f"\n📊 Small Batch Summary:")
        print(f"Selected {len(test_cases)} questions")
        file_types_used = [tc['file_type'] for tc in test_cases]
        for ft in set(file_types_used):
            count = file_types_used.count(ft)
            print(f"  {ft.upper()}: {count} question{'s' if count != 1 else ''}")
        
        return test_cases
    
    def create_large_test_batch(self) -> List[Dict]:
        """
        Create large test batch: 25 questions with comprehensive coverage
        
        Returns:
            List of 25 test cases covering all difficulty levels and file types
        """
        print("\n📦 Creating Large Test Batch (25 questions)")
        print("=" * 50)
        
        selected_questions = []
        
        # Strategy: distribute across levels and file types
        target_distribution = {
            '1': 8,   # 8 Level 1 questions (easier)
            '2': 12,  # 12 Level 2 questions (moderate) 
            '3': 5    # 5 Level 3 questions (hardest)
        }
        
        level_counts = {'1': 0, '2': 0, '3': 0}
        file_type_counts = {}
        
        # Get all file questions sorted by level
        all_file_questions = list(self.file_questions.values())
        
        # Process by level priority
        for target_level in ['1', '2', '3']:
            level_questions = [q for q in all_file_questions if str(q.get('Level', '')) == target_level]
            target_count = target_distribution[target_level]
            
            print(f"\n🎯 Level {target_level}: targeting {target_count} questions")
            
            for question in level_questions:
                if level_counts[target_level] >= target_count:
                    break
                    
                if len(selected_questions) >= 25:
                    break
                
                # Check if file exists
                file_path = self.setup_test_file(question['task_id'])
                if file_path:
                    selected_questions.append(question)
                    level_counts[target_level] += 1
                    
                    # Track file type distribution
                    file_type = question.get('file_name', '').split('.')[-1].lower()
                    file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
                    
                    print(f"  ✅ Added {file_type.upper()}: {question['task_id']}")
        
        # Fill remaining slots if we're under 25
        remaining_questions = [q for q in all_file_questions if q not in selected_questions]
        for question in remaining_questions:
            if len(selected_questions) >= 25:
                break
                
            file_path = self.setup_test_file(question['task_id'])
            if file_path:
                selected_questions.append(question)
                level = str(question.get('Level', ''))
                level_counts[level] = level_counts.get(level, 0) + 1
                
                file_type = question.get('file_name', '').split('.')[-1].lower()
                file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
        
        # Create test cases
        test_cases = []
        for question in selected_questions:
            file_path = self.setup_test_file(question['task_id'])
            if file_path:
                test_cases.append({
                    'task_id': question['task_id'],
                    'question': question['Question'],
                    'level': question.get('Level'),
                    'file_path': file_path,
                    'file_name': question.get('file_name'),
                    'file_type': question.get('file_name', '').split('.')[-1].lower(),
                    'expected_answer': question.get('Final answer'),
                    'annotator_steps': question.get('Annotator Metadata', {}).get('Steps'),
                    'expected_tools': question.get('Annotator Metadata', {}).get('Tools'),
                    'batch_type': 'large'
                })
        
        print(f"\n📊 Large Batch Summary:")
        print(f"Selected {len(test_cases)} questions")
        print(f"Level distribution:")
        for level in ['1', '2', '3']:
            count = level_counts.get(level, 0)
            print(f"  Level {level}: {count} questions")
        
        print(f"File type distribution:")
        for file_type, count in sorted(file_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {file_type.upper()}: {count} questions")
        
        return test_cases
    
    def analyze_question_complexity(self, task_id: str) -> Dict:
        """Analyze a specific question's complexity and requirements"""
        if task_id not in self.file_questions:
            return {"error": "Task not found"}
        
        question = self.file_questions[task_id]
        
        analysis = {
            "task_id": task_id,
            "level": question.get('Level'),
            "file_type": question.get('file_name', '').split('.')[-1].lower(),
            "question_length": len(question.get('Question', '')),
            "expected_steps": question.get('Annotator Metadata', {}).get('Number of steps'),
            "expected_time": question.get('Annotator Metadata', {}).get('How long did this take?'),
            "tools_needed": question.get('Annotator Metadata', {}).get('Tools'),
            "requires_file_processing": 'file' in question.get('Question', '').lower(),
            "requires_calculation": any(word in question.get('Question', '').lower() 
                                      for word in ['calculate', 'compute', 'sum', 'average', 'total']),
            "requires_search": any(word in question.get('Question', '').lower() 
                                 for word in ['find', 'search', 'locate', 'identify'])
        }
        
        return analysis

def _compare_answers(agent_answer: str, expected_answer: str) -> bool:
    """
    Enhanced answer comparison with multiple strategies
    
    Args:
        agent_answer: Answer from the agent
        expected_answer: Expected correct answer
    
    Returns:
        True if answers match using any comparison strategy
    """
    if not agent_answer or not expected_answer:
        return False
    
    # Normalize both answers
    agent_norm = agent_answer.lower().strip()
    expected_norm = expected_answer.lower().strip()
    
    # Strategy 1: Exact match
    if agent_norm == expected_norm:
        return True
    
    # Strategy 2: Expected answer contained in agent answer
    if expected_norm in agent_norm:
        return True
    
    # Strategy 3: Agent answer contained in expected (for longer expected answers)
    if len(agent_norm) > 3 and agent_norm in expected_norm:
        return True
    
    # Strategy 4: Remove common prefixes/suffixes and compare
    import re
    
    # Remove common GAIA formatting
    agent_clean = re.sub(r'^(final answer:?|answer:?|result:?)\s*', '', agent_norm)
    expected_clean = re.sub(r'^(final answer:?|answer:?|result:?)\s*', '', expected_norm)
    
    if agent_clean == expected_clean:
        return True
    
    # Strategy 5: For numeric answers, extract and compare numbers
    agent_numbers = re.findall(r'\d+\.?\d*', agent_answer)
    expected_numbers = re.findall(r'\d+\.?\d*', expected_answer)
    
    if agent_numbers and expected_numbers and agent_numbers[0] == expected_numbers[0]:
        return True
    
    return False

def _generate_batch_summary(results: List[Dict], batch_name: str, total_time: float):
    """Generate comprehensive summary of batch test results"""
    
    print(f"\n📊 {batch_name} Test Summary")
    print("=" * 60)
    
    total_tests = len(results)
    successful_tests = [r for r in results if r.get('success', False)]
    correct_answers = [r for r in results if r.get('answer_match', False)]
    error_tests = [r for r in results if 'error' in r]
    
    # Basic metrics
    print(f"📈 Overall Performance:")
    print(f"  Total tests: {total_tests}")
    print(f"  Successful executions: {len(successful_tests)} ({len(successful_tests)/total_tests:.1%})")
    print(f"  Correct answers: {len(correct_answers)} ({len(correct_answers)/total_tests:.1%})")
    print(f"  Failed tests: {len(error_tests)} ({len(error_tests)/total_tests:.1%})")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average time per question: {total_time/total_tests:.2f}s")
    
    if successful_tests:
        avg_execution_time = sum(r.get('execution_time', 0) for r in successful_tests) / len(successful_tests)
        avg_steps = sum(r.get('steps_count', 0) for r in successful_tests) / len(successful_tests)
        print(f"  Average execution time: {avg_execution_time:.2f}s")
        print(f"  Average steps per question: {avg_steps:.1f}")
    
    # Performance by level
    level_stats = {}
    for result in results:
        level = str(result.get('level', 'Unknown'))
        if level not in level_stats:
            level_stats[level] = {'total': 0, 'correct': 0, 'successful': 0}
        
        level_stats[level]['total'] += 1
        if result.get('success', False):
            level_stats[level]['successful'] += 1
        if result.get('answer_match', False):
            level_stats[level]['correct'] += 1
    
    print(f"\n📊 Performance by Level:")
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
        accuracy_rate = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  Level {level}: {stats['correct']}/{stats['total']} correct ({accuracy_rate:.1%}), {stats['successful']} successful ({success_rate:.1%})")
    
    # Performance by file type
    file_type_stats = {}
    for result in successful_tests:
        file_type = result.get('file_type', 'unknown')
        if file_type not in file_type_stats:
            file_type_stats[file_type] = {'total': 0, 'correct': 0}
        
        file_type_stats[file_type]['total'] += 1
        if result.get('answer_match', False):
            file_type_stats[file_type]['correct'] += 1
    
    if file_type_stats:
        print(f"\n📁 Performance by File Type:")
        for file_type in sorted(file_type_stats.keys()):
            stats = file_type_stats[file_type]
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {file_type.upper()}: {stats['correct']}/{stats['total']} correct ({accuracy:.1%})")
    
    # Strategy distribution
    strategy_stats = {}
    for result in successful_tests:
        strategy = result.get('strategy_used', 'unknown')
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {'total': 0, 'correct': 0}
        
        strategy_stats[strategy]['total'] += 1
        if result.get('answer_match', False):
            strategy_stats[strategy]['correct'] += 1
    
    if strategy_stats:
        print(f"\n🔄 Performance by Strategy:")
        for strategy in sorted(strategy_stats.keys()):
            stats = strategy_stats[strategy]
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {strategy}: {stats['correct']}/{stats['total']} correct ({accuracy:.1%})")
    
    # Show errors if any
    if error_tests:
        print(f"\n🐛 Error Analysis ({len(error_tests)} failures):")
        error_types = {}
        for error_test in error_tests[:5]:  # Show first 5 errors
            error_msg = error_test.get('error', 'Unknown error')
            error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
            error_types[error_type] = error_types.get(error_type, 0) + 1
            print(f"  Test {error_test.get('test_number', '?')}: {error_msg[:80]}...")
        
        if len(error_tests) > 5:
            print(f"  ... and {len(error_tests) - 5} more errors")
    
    print(f"\n✅ {batch_name} test completed!")

def _run_batch_test(agent, test_cases: List[Dict], batch_name: str):
    """
    Internal function to run a batch of tests
    
    Args:
        agent: The GAIA agent to test
        test_cases: List of test case dictionaries
        batch_name: Name of the batch for reporting
    """
    from agent_interface import run_single_question_enhanced
    import time
    
    results = []
    start_time = time.time()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/{len(test_cases)}: Level {test_case['level']} - {test_case['file_type'].upper()}")
        print(f"📁 File: {test_case['file_name']}")
        print(f"❓ Question: {test_case['question'][:80]}...")
        
        try:
            # Modify question to include file path reference
            question_with_file = f"{test_case['question']}\n\n[FILE]: {test_case['file_path']}"
            
            test_start = time.time()
            result = run_single_question_enhanced(agent, question_with_file, test_case['task_id'])
            test_time = time.time() - test_start
            
            agent_answer = result.get('final_answer', '').strip()
            expected_answer = test_case['expected_answer'].strip()
            
            # Enhanced answer comparison
            answer_match = _compare_answers(agent_answer, expected_answer)
            
            print(f"🤖 Agent: {agent_answer[:60]}{'...' if len(agent_answer) > 60 else ''}")
            print(f"✅ Expected: {expected_answer[:60]}{'...' if len(expected_answer) > 60 else ''}")
            print(f"🎯 Match: {'✅' if answer_match else '❌'}")
            print(f"⏱️ Time: {test_time:.2f}s")
            print(f"🔄 Strategy: {result.get('selected_strategy', 'unknown')}")
            
            results.append({
                **test_case,
                'agent_answer': agent_answer,
                'answer_match': answer_match,
                'execution_time': test_time,
                'strategy_used': result.get('selected_strategy', 'unknown'),
                'selected_agent': result.get('selected_agent', 'unknown'),
                'steps_count': len(result.get('steps', [])),
                'success': not ('error' in result),
                'test_number': i
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                **test_case,
                'error': str(e),
                'success': False,
                'test_number': i
            })
    
    total_time = time.time() - start_time
    
    # Generate comprehensive summary
    _generate_batch_summary(results, batch_name, total_time)
    
    return results

def run_small_batch_test(agent, dataset_manager: GAIADatasetManager = None):
    """
    Run small batch test: 5 questions with diverse file types
    
    Args:
        agent: The GAIA agent to test
        dataset_manager: GAIADatasetManager instance (optional)
    """
    if dataset_manager is None:
        dataset_manager = GAIADatasetManager()
    
    if not dataset_manager.metadata:
        print("❌ Cannot run test - no dataset loaded")
        return []
    
    print(f"\n🧪 GAIA Small Batch Test (5 questions)")
    print("=" * 60)
    
    test_cases = dataset_manager.create_small_test_batch()
    
    if not test_cases:
        print("❌ No test cases available")
        return []
    
    return _run_batch_test(agent, test_cases, "Small Batch")

def run_large_batch_test(agent, dataset_manager: GAIADatasetManager = None):
    """
    Run large batch test: 25 questions with comprehensive coverage
    
    Args:
        agent: The GAIA agent to test
        dataset_manager: GAIADatasetManager instance (optional)
    """
    if dataset_manager is None:
        dataset_manager = GAIADatasetManager()
    
    if not dataset_manager.metadata:
        print("❌ Cannot run test - no dataset loaded")
        return []
    
    print(f"\n🧪 GAIA Large Batch Test (25 questions)")
    print("=" * 60)
    
    test_cases = dataset_manager.create_large_test_batch()
    
    if not test_cases:
        print("❌ No test cases available")
        return []
    
    return _run_batch_test(agent, test_cases, "Large Batch")

def quick_dataset_check(dataset_path: str = "./tests/gaia_data"):
    """Quick check of your local GAIA dataset with metadata validation"""
    manager = GAIADatasetManager(dataset_path)
    
    if not manager.metadata:
        print("❌ Dataset not found or invalid")
        return False
    
    print(f"\n📊 Local GAIA Dataset Check")
    print("=" * 60)
    print(f"🔍 Checking metadata file: {os.path.join(manager.dataset_path, 'metadata.json')}")
    print(f"📁 Scanning data directory: {manager.dataset_path}")
    print(f"✅ Successfully loaded GAIA dataset from: {manager.dataset_path}")
    
    # Handle both list and dict metadata structures
    if isinstance(manager.metadata, list):
        print(f"📋 Total questions in metadata.json: {len(manager.metadata)}")
        print(f"📎 Questions with file attachments: {len(manager.file_questions)}")
        sample_question = manager.metadata[0] if manager.metadata else None
    elif isinstance(manager.metadata, dict):
        print(f"📋 Total questions in metadata.json: {len(manager.metadata)}")
        print(f"📎 Questions with file attachments: {len(manager.file_questions)}")
        # Get first item from dict
        sample_question = next(iter(manager.metadata.values())) if manager.metadata else None
    else:
        print(f"❌ Unexpected metadata structure: {type(manager.metadata)}")
        return False
    
    # Validate metadata structure
    if sample_question:
        print(f"\n📝 Metadata Structure Analysis (from metadata.json):")
        print(f"  Required GAIA fields present in metadata.json:")
        required_fields = ['task_id', 'Question', 'Level', 'Final answer']
        for field in required_fields:
            present = field in sample_question
            print(f"    ✅ {field}: Found" if present else f"    ❌ {field}: Missing")
        
        # Show additional fields found
        additional_fields = [k for k in sample_question.keys() if k not in required_fields]
        if additional_fields:
            print(f"  Additional fields in metadata.json: {', '.join(additional_fields)}")
        
        # Show sample question structure
        print(f"\n  📄 Sample question from metadata.json:")
        print(f"    Task ID: {sample_question.get('task_id', 'Not found')}")
        print(f"    Level: {sample_question.get('Level', 'Not found')}")
        print(f"    Question preview: {sample_question.get('Question', 'Not found')[:80]}...")
        
        if 'file_name' in sample_question:
            print(f"    Associated file: {sample_question.get('file_name', 'None')}")
    else:
        print(f"\n❌ No sample question available from metadata.json")
    
    # Check file references in metadata vs actual files in directory
    file_ref_stats = {
        'has_file_name': 0,
        'has_file_path': 0,
        'file_found_in_directory': 0,
        'file_missing_from_directory': 0
    }
    
    print(f"\n📁 File Reference Analysis:")
    print(f"   Comparing metadata.json file references with actual files in {manager.dataset_path}")
    
    # Check first 10 file questions for detailed analysis
    sample_file_questions = list(manager.file_questions.items())[:10]
    
    if not sample_file_questions:
        print(f"  ⚠️  No file references found in metadata.json")
        print(f"  💡 This could mean:")
        print(f"     - Your dataset contains only text-based questions (no files needed)")
        print(f"     - File references use different field names")
        
        # Check if ANY questions have file-related fields
        all_questions = manager.metadata if isinstance(manager.metadata, list) else list(manager.metadata.values())
        
        file_related_fields = ['file_name', 'file_path', 'Attachments', 'Files']
        questions_with_files = []
        
        print(f"  🔍 Searching metadata.json for file-related fields...")
        for question in all_questions[:5]:  # Check first 5
            has_file_field = any(field in question for field in file_related_fields)
            if has_file_field:
                questions_with_files.append(question)
                print(f"    ✅ Found file fields in task: {question.get('task_id', 'unknown')}")
                for field in file_related_fields:
                    if field in question:
                        print(f"      {field}: {question[field]}")
        
        if not questions_with_files:
            print(f"  ❓ No file-related fields found in metadata.json sample")
            print(f"  📋 Available fields in metadata.json: {list(sample_question.keys()) if sample_question else 'None'}")
    else:
        print(f"  📋 Found {len(sample_file_questions)} questions with file references in metadata.json")
        print(f"  🔍 Checking if referenced files exist in directory {manager.dataset_path}...")
        
        for task_id, question in sample_file_questions:
            file_name = question.get('file_name', '')
            file_path = question.get('file_path', '')
            
            if file_name:
                file_ref_stats['has_file_name'] += 1
            if file_path:
                file_ref_stats['has_file_path'] += 1
            
            # Try to locate the actual file in the directory
            actual_file_path = manager.setup_test_file(task_id)
            if actual_file_path:
                file_ref_stats['file_found_in_directory'] += 1
                print(f"    ✅ {file_name} -> Found in directory")
            else:
                file_ref_stats['file_missing_from_directory'] += 1
                print(f"    ❌ {file_name} -> Not found in directory")
        
        print(f"\n  📊 File Reference Summary:")
        print(f"    Questions with 'file_name' in metadata.json: {file_ref_stats['has_file_name']}")
        print(f"    Questions with 'file_path' in metadata.json: {file_ref_stats['has_file_path']}")
        print(f"    Files actually found in {manager.dataset_path}: {file_ref_stats['file_found_in_directory']}")
        print(f"    Files missing from {manager.dataset_path}: {file_ref_stats['file_missing_from_directory']}")
    
    # Show actual files present in the directory
    print(f"\n📂 Directory Contents Analysis:")
    print(f"   Scanning actual files in directory: {manager.dataset_path}")
    
    if os.path.exists(manager.dataset_path):
        actual_files = [f for f in os.listdir(manager.dataset_path) 
                       if os.path.isfile(os.path.join(manager.dataset_path, f))
                       and not f.startswith('.')]
        
        print(f"  📊 Files found in {manager.dataset_path}:")
        print(f"    Total files: {len(actual_files)}")
        
        # Show file types in directory
        file_types_in_dir = {}
        for file in actual_files:
            ext = file.split('.')[-1].lower() if '.' in file else 'no_ext'
            file_types_in_dir[ext] = file_types_in_dir.get(ext, 0) + 1
        
        print(f"    File types found in directory:")
        for ext, count in sorted(file_types_in_dir.items(), key=lambda x: x[1], reverse=True):
            print(f"      .{ext}: {count} files")
        
        # Show sample filenames from directory
        print(f"    Sample files in {manager.dataset_path}:")
        for file in actual_files[:5]:
            print(f"      {file}")
        if len(actual_files) > 5:
            print(f"      ... and {len(actual_files) - 5} more files")
        
        # Special check for metadata.json
        if 'metadata.json' in actual_files:
            print(f"    ✅ metadata.json confirmed present in directory")
        else:
            print(f"    ⚠️  metadata.json not listed (but was loaded successfully)")
    else:
        print(f"  ❌ Directory {manager.dataset_path} not accessible")
    
    # Show file types expected from metadata
    file_types_expected = manager.get_file_types_distribution()
    if file_types_expected:
        print(f"\n📋 File Types Expected (from metadata.json references):")
        for file_type, count in sorted(file_types_expected.items(), key=lambda x: x[1], reverse=True)[:8]:
            print(f"    .{file_type}: {count} questions expect this file type")
    else:
        print(f"\n📋 No specific file types referenced in metadata.json")
    
    # Detailed file organization analysis
    print(f"\n🔍 File Organization Analysis:")
    print(f"   Comparing metadata.json expectations vs {manager.dataset_path} contents")
    
    # Sample a few questions and show what files should exist vs what's found
    if sample_file_questions:
        sample_questions = list(manager.file_questions.values())[:3]
    else:
        # If no file questions, show sample of all questions
        all_questions = manager.metadata if isinstance(manager.metadata, list) else list(manager.metadata.values())
        sample_questions = all_questions[:3]
        print(f"  ⚠️  Using sample questions (no file questions found in metadata.json)")
    
    for i, question in enumerate(sample_questions, 1):
        task_id = question['task_id']
        expected_file = question.get('file_name', 'no file specified in metadata.json')
        metadata_path = question.get('file_path', 'not specified in metadata.json')
        
        print(f"\n  Example {i} (from metadata.json):")
        print(f"    Task ID: {task_id}")
        print(f"    Expected filename (per metadata.json): {expected_file}")
        print(f"    Metadata file_path field: {metadata_path}")
        
        # Try to find the file if there's a file reference
        if expected_file and expected_file != 'no file specified in metadata.json':
            found_path = manager.setup_test_file(task_id)
            if found_path:
                print(f"    ✅ File found in directory: {found_path}")
            else:
                print(f"    ❌ File not found in directory: {manager.dataset_path}")
                # Suggest possible file organization
                if expected_file:
                    possible_names = [
                        expected_file,
                        f"{task_id}.{expected_file.split('.')[-1]}",
                        f"{task_id[:8]}.{expected_file.split('.')[-1]}"  # First part of UUID
                    ]
                    print(f"    💡 Searched for these names: {possible_names}")
        else:
            print(f"    📝 Text-only question: {question.get('Question', '')[:60]}...")
    
    # Calculate success rate and provide summary
    if sample_file_questions:
        success_rate = file_ref_stats['file_found_in_directory'] / len(sample_file_questions) if sample_file_questions else 0
        print(f"\n📈 Dataset Readiness Assessment:")
        print(f"   File availability: {success_rate:.1%} of metadata.json file references found in {manager.dataset_path}")
        
        if success_rate < 0.5:
            print(f"\n💡 Suggestions for improving file organization:")
            print(f"   1. Copy missing files to directory: {manager.dataset_path}")
            print(f"   2. Use exact filenames specified in metadata.json 'file_name' field")
            print(f"   3. Alternative: rename files to use task_id as filename")
            print(f"   4. Verify metadata.json 'file_path' references point to correct locations")
        
        print(f"\n✅ Summary: metadata.json loaded successfully, {len(manager.file_questions)} file references found")
        return success_rate > 0.5
    else:
        print(f"\n📈 Dataset Readiness Assessment:")
        print(f"   Text-only dataset: metadata.json loaded successfully with {len(manager.metadata)} questions")
        print(f"   No file dependencies found - ready for text-based testing!")
        return True  # Text-only datasets are valid
        
def run_quick_file_test(agent_config: str = "groq"):
    """
    Quick test with one file from your local dataset
    
    Args:
        agent_config: Agent configuration to use
    """
    from agent_interface import create_gaia_agent
    
    print(f"\n🎯 Quick GAIA File Test")
    print("=" * 30)
    
    # Create agent
    agent = create_gaia_agent(agent_config)
    
    # Create dataset manager
    manager = GAIADatasetManager()
    
    if not manager.metadata:
        print("❌ No dataset found")
        return None
    
    # Get one test case
    test_cases = manager.create_small_test_batch()
    
    if not test_cases:
        print("❌ No test cases available")
        return None
    
    # Run just the first test
    test_case = test_cases[0]
    
    print(f"📁 File: {test_case['file_name']} ({test_case['file_type'].upper()})")
    print(f"📊 Level: {test_case['level']}")
    print(f"❓ Question: {test_case['question'][:100]}...")
    
    from agent_interface import run_single_question_enhanced
    
    question_with_file = f"{test_case['question']}\n\n[FILE]: {test_case['file_path']}"
    result = run_single_question_enhanced(agent, question_with_file, test_case['task_id'])
    
    agent_answer = result.get('final_answer', '')
    expected_answer = test_case['expected_answer']
    match = _compare_answers(agent_answer, expected_answer)
    
    print(f"\n🤖 Agent Answer: {agent_answer}")
    print(f"✅ Expected: {expected_answer}")
    print(f"🎯 Match: {'✅' if match else '❌'}")
    print(f"⏱️ Time: {result.get('execution_time', 0):.2f}s")
    print(f"🔄 Strategy: {result.get('selected_strategy', 'unknown')}")
    
    return result

def run_gaia_tests(agent_config: str = "groq", batch_size: str = "small"):
    """
    Main function to run GAIA tests with your local dataset
    
    Args:
        agent_config: Agent configuration ('groq', 'performance', 'accuracy', etc.)
        batch_size: 'small' (5 questions) or 'large' (25 questions)
    """
    from agent_interface import create_gaia_agent
    
    print(f"\n🚀 Running GAIA Tests")
    print(f"Config: {agent_config}")
    print(f"Batch: {batch_size}")
    print("=" * 50)
    
    # Create agent
    try:
        agent = create_gaia_agent(agent_config)
        print(f"✅ Agent created: {agent.config.model_provider}/{agent.config.model_name}")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return None
    
    # Create dataset manager (will automatically find ./tests/gaia_data)
    manager = GAIADatasetManager()
    
    if not manager.metadata:
        print("❌ GAIA dataset not found in ./tests/gaia_data")
        return None
    
    # Run appropriate batch
    if batch_size.lower() == "small":
        results = run_small_batch_test(agent, manager)
    elif batch_size.lower() == "large":
        results = run_large_batch_test(agent, manager)
    else:
        print(f"❌ Unknown batch size: {batch_size}. Use 'small' or 'large'")
        return None
    
    return results

def test_with_gaia_files(agent, dataset_manager: GAIADatasetManager, num_tests: int = 3):
    """
    Legacy function - use run_small_batch_test or run_large_batch_test instead
    """
    print("⚠️  test_with_gaia_files is deprecated. Use run_small_batch_test() or run_large_batch_test() instead")
    
    if num_tests <= 5:
        return run_small_batch_test(agent, dataset_manager)
    else:
        return run_large_batch_test(agent, dataset_manager)

# Example usage functions
def show_gaia_dataset_info(dataset_path: str = None):
    """Show information about the GAIA dataset"""
    manager = GAIADatasetManager(dataset_path)
    
    if not manager.metadata:
        return
    
    print("\n📊 GAIA Dataset Information:")
    print("=" * 40)
    
    # Level distribution
    levels = {}
    for item in manager.metadata:
        level = item.get('Level', 'Unknown')
        levels[level] = levels.get(level, 0) + 1
    
    print(f"Questions by level:")
    for level, count in sorted(levels.items()):
        print(f"  Level {level}: {count} questions")
    
    # File type distribution
    file_types = manager.get_file_types_distribution()
    print(f"\nFile types in dataset:")
    for file_type, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  .{file_type}: {count} files")
    
    # Sample questions with files
    sample_questions = manager.get_questions_with_files()[:3]
    print(f"\nSample questions with files:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. Level {q.get('Level')}: {q.get('file_name')} - {q['Question'][:80]}...")

def quick_gaia_file_test(agent, dataset_path: str = None):
    """Quick test with one GAIA file"""
    manager = GAIADatasetManager(dataset_path)
    
    # Get one Excel file question for testing
    excel_questions = manager.get_questions_with_files(file_type='xlsx')
    
    if not excel_questions:
        print("❌ No Excel questions found in dataset")
        return
    
    # Use the example you provided
    test_question = None
    for q in excel_questions:
        if q['task_id'] == '32102e3e-d12a-4209-9163-7b3a104efe5d':
            test_question = q
            break
    
    if not test_question:
        test_question = excel_questions[0]  # Use first available
    
    print(f"\n🎯 Quick GAIA File Test")
    print(f"Task ID: {test_question['task_id']}")
    print(f"File: {test_question.get('file_name')}")
    print(f"Question: {test_question['Question']}")
    
    # Setup file
    file_path = manager.setup_test_file(test_question['task_id'])
    
    if file_path:
        from agent_interface import run_single_question_enhanced
        
        question_with_file = f"{test_question['Question']}\n\nFile location: {file_path}"
        result = run_single_question_enhanced(agent, question_with_file)
        
        print(f"\n🤖 Agent Answer: {result.get('final_answer', '')}")
        print(f"✅ Expected: {test_question.get('Final answer', '')}")
        print(f"⏱️ Time: {result.get('execution_time', 0):.2f}s")
        
        return result
    
    return None

def compare_configs_on_gaia(configs: List[str] = None, batch_size: str = "small"):
    """
    Compare different agent configurations on GAIA dataset
    
    Args:
        configs: List of config names to compare
        batch_size: 'small' or 'large' batch
    """
    if configs is None:
        configs = ["groq", "performance", "accuracy"]
    
    print(f"\n🔬 GAIA Configuration Comparison")
    print(f"Configs: {', '.join(configs)}")
    print(f"Batch: {batch_size}")
    print("=" * 60)
    
    comparison_results = {}
    
    for config in configs:
        print(f"\n🧪 Testing {config} configuration...")
        
        try:
            results = run_gaia_tests(config, batch_size)
            
            if results:
                successful = len([r for r in results if r.get('success', False)])
                correct = len([r for r in results if r.get('answer_match', False)])
                avg_time = sum(r.get('execution_time', 0) for r in results) / len(results)
                
                comparison_results[config] = {
                    'total_tests': len(results),
                    'successful_tests': successful,
                    'correct_answers': correct,
                    'success_rate': successful / len(results),
                    'accuracy_rate': correct / len(results),
                    'avg_time': avg_time,
                    'results': results
                }
                
                print(f"✅ {config}: {correct}/{len(results)} correct ({correct/len(results):.1%})")
                
        except Exception as e:
            print(f"❌ {config} failed: {e}")
            comparison_results[config] = {'error': str(e)}
    
    # Generate comparison summary
    print(f"\n📊 Configuration Comparison Summary")
    print("=" * 50)
    
    if comparison_results:
        # Sort by accuracy
        sorted_configs = sorted(
            [(k, v) for k, v in comparison_results.items() if 'error' not in v],
            key=lambda x: x[1].get('accuracy_rate', 0),
            reverse=True
        )
        
        print(f"{'Config':<12} {'Accuracy':<10} {'Success':<10} {'Avg Time':<10}")
        print("-" * 50)
        
        for config, stats in sorted_configs:
            accuracy = stats.get('accuracy_rate', 0)
            success = stats.get('success_rate', 0)
            avg_time = stats.get('avg_time', 0)
            
            print(f"{config:<12} {accuracy:<10.1%} {success:<10.1%} {avg_time:<10.2f}s")
        
        # Show best performing config
        if sorted_configs:
            best_config, best_stats = sorted_configs[0]
            print(f"\n🏆 Best performing: {best_config}")
            print(f"   Accuracy: {best_stats['accuracy_rate']:.1%}")
            print(f"   Success rate: {best_stats['success_rate']:.1%}")
            print(f"   Average time: {best_stats['avg_time']:.2f}s")
    
    return comparison_results

if __name__ == "__main__":
    # Example usage with your local dataset
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "check":
            quick_dataset_check()
        elif command == "quick":
            config = sys.argv[2] if len(sys.argv) > 2 else "groq"
            run_quick_file_test(config)
        elif command == "small":
            config = sys.argv[2] if len(sys.argv) > 2 else "groq"
            run_gaia_tests(config, "small")
        elif command == "large":
            config = sys.argv[2] if len(sys.argv) > 2 else "groq"
            run_gaia_tests(config, "large")
        elif command == "compare":
            configs = sys.argv[2:] if len(sys.argv) > 2 else ["groq", "performance"]
            compare_configs_on_gaia(configs, "small")
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  check - Check dataset availability")
            print("  quick [config] - Quick single file test")
            print("  small [config] - Small batch test (5 questions)")
            print("  large [config] - Large batch test (25 questions)")
            print("  compare [configs...] - Compare configurations")
    else:
        print("🚀 GAIA Dataset Testing Utilities")
        print("=" * 40)
        print("Your dataset should be in: ./tests/gaia_data/")
        print("")
        print("Available functions:")
        print("- quick_dataset_check() - Verify your dataset")
        print("- run_quick_file_test() - Test one file")
        print("- run_gaia_tests(config, 'small') - 5 question test")
        print("- run_gaia_tests(config, 'large') - 25 question test")
        print("- compare_configs_on_gaia() - Compare configurations")
        print("")
        print("Command line usage:")
        print("  python gaia_dataset_utils.py check")
        print("  python gaia_dataset_utils.py small groq")
        print("  python gaia_dataset_utils.py compare groq performance accuracy")
