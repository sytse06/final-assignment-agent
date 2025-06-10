# gaia_dataset_utils.py - PURE DATA LAYER
# Responsibilities: Dataset management, validation, file operations, test batch creation
# NO agent execution, evaluation, or performance analysis

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime
import re
from collections import defaultdict

class GAIADatasetManager:
    """Pure dataset management - no agent interaction"""
    
    def __init__(self, dataset_path: str = None):
        """Initialize dataset manager and load metadata"""
        self.dataset_path = self._find_dataset_path(dataset_path)
        self.metadata = None
        self.file_questions = {}
        self.test_metadata = None  # For test set if available
        
        if self.dataset_path:
            self._load_metadata()
    
    def _find_dataset_path(self, provided_path: str = None) -> Optional[str]:
        """Find GAIA dataset path in common locations"""
        if provided_path and os.path.exists(provided_path):
            return provided_path
        
        # Common locations where GAIA dataset might be
        common_paths = [
            "./tests/gaia_data",
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
        return None
    
    def _load_metadata(self):
        """Load and parse metadata.json with structure validation"""
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
                
                # Use validation set (has answers for evaluation)
                self.metadata = raw_metadata['validation']
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
    
    # ============================================================================
    # CORE DATA OPERATIONS
    # ============================================================================
    
    def get_question_by_id(self, task_id: str) -> Optional[Dict]:
        """Retrieve specific question with all metadata"""
        if not self.metadata:
            return None
        
        for question in self.metadata:
            if question.get('task_id') == task_id:
                return question.copy()  # Return copy to prevent modification
        
        return None
    
    def get_ground_truth(self, task_id: str) -> Optional[Dict]:
        """Get expected answer and metadata for evaluation"""
        question = self.get_question_by_id(task_id)
        
        if not question:
            return None
        
        return {
            'final_answer': question.get('Final answer', ''),
            'level': question.get('Level'),
            'question': question.get('Question', ''),
            'annotator_metadata': question.get('Annotator Metadata', {}),
            'file_name': question.get('file_name', ''),
            'file_path': question.get('file_path', ''),
            'task_id': task_id
        }
    
    def find_file_for_question(self, task_id: str) -> Optional[str]:
        """Locate actual file for a question (file system operations)"""
        if task_id not in self.file_questions:
            return None
        
        question = self.file_questions[task_id]
        file_name = question.get('file_name')
        
        if not file_name:
            return None
        
        # Strategy 1: Look for exact filename in dataset directory
        exact_file_path = os.path.join(self.dataset_path, file_name)
        if os.path.exists(exact_file_path):
            return exact_file_path
        
        # Strategy 2: Look for file using task_id as filename
        file_extension = file_name.split('.')[-1] if '.' in file_name else ''
        task_id_filename = f"{task_id}.{file_extension}" if file_extension else task_id
        task_id_file_path = os.path.join(self.dataset_path, task_id_filename)
        
        if os.path.exists(task_id_file_path):
            return task_id_file_path
        
        # Strategy 3: Use original file_path from metadata
        original_path = question.get('file_path')
        if original_path and os.path.exists(original_path):
            return original_path
        
        # Strategy 4: Search for any file containing the task_id
        if os.path.exists(self.dataset_path):
            for file in os.listdir(self.dataset_path):
                if task_id in file:
                    found_path = os.path.join(self.dataset_path, file)
                    if os.path.isfile(found_path):
                        return found_path
        
        return None
    
    def validate_answer_format(self, answer: str, task_id: str) -> Dict:
        """Validate answer format against GAIA requirements"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'formatted_answer': answer
        }
        
        if not answer or not answer.strip():
            validation_result['is_valid'] = False
            validation_result['issues'].append("Empty answer")
            return validation_result
        
        # GAIA formatting rules
        answer_clean = answer.strip()
        
        # Check for common formatting issues
        if answer_clean.lower().startswith('final answer:'):
            # Extract the actual answer part
            answer_clean = re.sub(r'^final answer:\s*', '', answer_clean, flags=re.IGNORECASE)
            validation_result['formatted_answer'] = answer_clean
        
        # Check for excessive punctuation or formatting
        if len(answer_clean) != len(answer_clean.strip('.,!?')):
            validation_result['issues'].append("Contains trailing punctuation")
        
        # Check for articles that should be removed
        articles = ['the ', 'a ', 'an ']
        if any(answer_clean.lower().startswith(article) for article in articles):
            validation_result['issues'].append("Contains articles (the, a, an)")
        
        return validation_result
    
    # ============================================================================
    # DATASET ANALYSIS (NO AGENT EXECUTION)
    # ============================================================================
    
    def analyze_dataset_distribution(self) -> Dict:
        """Analyze levels, file types, complexity indicators"""
        if not self.metadata:
            return {}
        
        analysis = {
            'total_questions': len(self.metadata),
            'questions_with_files': len(self.file_questions),
            'level_distribution': {},
            'file_type_distribution': {},
            'complexity_indicators': {},
            'question_length_stats': {},
            'annotator_metadata_coverage': {}
        }
        
        # Level distribution
        for question in self.metadata:
            level = question.get('Level', 'Unknown')
            analysis['level_distribution'][level] = analysis['level_distribution'].get(level, 0) + 1
        
        # File type distribution
        for question in self.file_questions.values():
            file_name = question.get('file_name', '')
            if file_name:
                ext = file_name.split('.')[-1].lower()
                analysis['file_type_distribution'][ext] = analysis['file_type_distribution'].get(ext, 0) + 1
        
        # Question length analysis
        question_lengths = [len(q.get('Question', '')) for q in self.metadata]
        if question_lengths:
            analysis['question_length_stats'] = {
                'min': min(question_lengths),
                'max': max(question_lengths),
                'avg': sum(question_lengths) / len(question_lengths),
                'median': sorted(question_lengths)[len(question_lengths)//2]
            }
        
        # Complexity indicators
        for question in self.metadata:
            question_text = question.get('Question', '').lower()
            
            # Count complexity keywords
            calc_keywords = ['calculate', 'compute', 'sum', 'average', 'total', 'count']
            analysis_keywords = ['analyze', 'compare', 'evaluate', 'determine', 'identify']
            
            has_calculation = any(keyword in question_text for keyword in calc_keywords)
            has_analysis = any(keyword in question_text for keyword in analysis_keywords)
            
            complexity_key = 'calculation' if has_calculation else ('analysis' if has_analysis else 'other')
            analysis['complexity_indicators'][complexity_key] = analysis['complexity_indicators'].get(complexity_key, 0) + 1
        
        return analysis
    
    def get_questions_by_criteria(self, 
                                level: int = None,
                                file_type: str = None,
                                has_files: bool = None,
                                min_length: int = None,
                                max_length: int = None) -> List[Dict]:
        """Filter questions by various criteria"""
        if not self.metadata:
            return []
        
        filtered_questions = []
        
        for question in self.metadata:
            # Level filtering
            if level is not None and question.get('Level') != level:
                continue
            
            # File filtering
            question_has_file = question.get('task_id') in self.file_questions
            if has_files is not None and question_has_file != has_files:
                continue
            
            # File type filtering
            if file_type is not None:
                if not question_has_file:
                    continue
                question_file = self.file_questions[question['task_id']]
                question_file_name = question_file.get('file_name', '')
                if not question_file_name.lower().endswith(f'.{file_type.lower()}'):
                    continue
            
            # Length filtering
            question_length = len(question.get('Question', ''))
            if min_length is not None and question_length < min_length:
                continue
            if max_length is not None and question_length > max_length:
                continue
            
            filtered_questions.append(question.copy())
        
        return filtered_questions
    
    def get_file_types_distribution(self) -> Dict[str, int]:
        """Get distribution of file types in the dataset"""
        file_types = {}
        
        for question in self.file_questions.values():
            file_name = question.get('file_name', '')
            if file_name:
                ext = file_name.split('.')[-1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
        
        return file_types
    
    # ============================================================================
    # TEST BATCH CREATION (DATA ONLY)
    # ============================================================================
    
    def create_test_batch(self, 
                         size: int,
                         strategy: str = "balanced",
                         **kwargs) -> List[Dict]:
        """Create test batches using various strategies"""
        
        if not self.metadata:
            print("❌ No metadata loaded")
            return []
        
        strategies = {
            "balanced": self._create_balanced_batch,
            "level_focused": self._create_level_focused_batch,
            "file_type_diverse": self._create_file_diverse_batch,
            "complexity_gradient": self._create_complexity_batch,
            "small_sample": self._create_small_sample_batch,
            "large_comprehensive": self._create_large_comprehensive_batch
        }
        
        if strategy not in strategies:
            print(f"❌ Unknown strategy: {strategy}")
            return []
        
        batch = strategies[strategy](size, **kwargs)
        
        print(f"📦 Created {strategy} batch: {len(batch)} questions")
        self._print_batch_summary(batch)
        
        return batch
    
    def _create_balanced_batch(self, size: int, **kwargs) -> List[Dict]:
        """Create balanced batch across levels and file types"""
        
        # Target distribution
        level_targets = {1: 0.4, 2: 0.4, 3: 0.2}  # 40% L1, 40% L2, 20% L3
        file_target = 0.6  # 60% with files, 40% without
        
        selected = []
        
        # Select by level
        for level, proportion in level_targets.items():
            level_size = int(size * proportion)
            level_questions = self.get_questions_by_criteria(level=level)
            
            if level_questions:
                # Split between file and non-file questions
                file_questions = [q for q in level_questions if q['task_id'] in self.file_questions]
                text_questions = [q for q in level_questions if q['task_id'] not in self.file_questions]
                
                file_count = min(int(level_size * file_target), len(file_questions))
                text_count = min(level_size - file_count, len(text_questions))
                
                selected.extend(file_questions[:file_count])
                selected.extend(text_questions[:text_count])
        
        # Fill remaining slots
        remaining = size - len(selected)
        if remaining > 0:
            used_ids = {q['task_id'] for q in selected}
            available = [q for q in self.metadata if q['task_id'] not in used_ids]
            selected.extend(available[:remaining])
        
        return selected[:size]
    
    def _create_level_focused_batch(self, size: int, target_level: int = 1, **kwargs) -> List[Dict]:
        """Create batch focused on specific level"""
        
        level_questions = self.get_questions_by_criteria(level=target_level)
        
        if len(level_questions) >= size:
            return level_questions[:size]
        else:
            # Fill with other levels if not enough
            selected = level_questions.copy()
            remaining = size - len(selected)
            
            used_ids = {q['task_id'] for q in selected}
            other_questions = [q for q in self.metadata if q['task_id'] not in used_ids]
            selected.extend(other_questions[:remaining])
            
            return selected
    
    def _create_file_diverse_batch(self, size: int, **kwargs) -> List[Dict]:
        """Create batch with diverse file types"""
        
        file_types = self.get_file_types_distribution()
        selected = []
        
        # Try to get at least one question per file type
        for file_type in file_types.keys():
            type_questions = self.get_questions_by_criteria(file_type=file_type)
            if type_questions and len(selected) < size:
                selected.append(type_questions[0])
        
        # Fill remaining slots
        remaining = size - len(selected)
        if remaining > 0:
            used_ids = {q['task_id'] for q in selected}
            file_questions = [q for q in self.metadata if q['task_id'] in self.file_questions and q['task_id'] not in used_ids]
            selected.extend(file_questions[:remaining])
        
        return selected[:size]
    
    def _create_complexity_batch(self, size: int, **kwargs) -> List[Dict]:
        """Create batch with complexity gradient (easy to hard)"""
        
        # Sort by level and question length as complexity proxy
        sorted_questions = sorted(self.metadata, key=lambda q: (
            q.get('Level', 1),
            len(q.get('Question', '')),
            1 if q['task_id'] in self.file_questions else 0
        ))
        
        # Take questions distributed across complexity spectrum
        step = max(1, len(sorted_questions) // size)
        selected = []
        
        for i in range(0, min(len(sorted_questions), size * step), step):
            selected.append(sorted_questions[i])
        
        return selected[:size]
    
    def _create_small_sample_batch(self, size: int = 5, **kwargs) -> List[Dict]:
        """Create small diverse sample for quick testing"""
        
        # Prioritize different file types and levels
        target_file_types = ['xlsx', 'csv', 'pdf', 'png', 'txt']
        selected = []
        used_types = set()
        
        # One question per file type if possible
        for file_type in target_file_types:
            if len(selected) >= size:
                break
                
            type_questions = self.get_questions_by_criteria(file_type=file_type, level=1)  # Start with Level 1
            if not type_questions:
                type_questions = self.get_questions_by_criteria(file_type=file_type)
            
            if type_questions and file_type not in used_types:
                # Verify file exists
                file_path = self.find_file_for_question(type_questions[0]['task_id'])
                if file_path:
                    selected.append(type_questions[0])
                    used_types.add(file_type)
        
        # Fill remaining with text questions
        remaining = size - len(selected)
        if remaining > 0:
            used_ids = {q['task_id'] for q in selected}
            text_questions = [q for q in self.metadata 
                            if q['task_id'] not in self.file_questions 
                            and q['task_id'] not in used_ids
                            and q.get('Level', 1) <= 2]  # Easier questions for quick test
            selected.extend(text_questions[:remaining])
        
        return selected[:size]
    
    def _create_large_comprehensive_batch(self, size: int = 25, **kwargs) -> List[Dict]:
        """Create large comprehensive batch for thorough evaluation"""
        
        # Target distribution for comprehensive testing
        level_distribution = {1: 8, 2: 12, 3: 5}  # Remaining slots flexible
        
        selected = []
        
        # Select by level with file/text mix
        for level, target_count in level_distribution.items():
            level_questions = self.get_questions_by_criteria(level=level)
            
            if level_questions:
                # 70% with files, 30% text for comprehensive testing
                file_questions = [q for q in level_questions if q['task_id'] in self.file_questions]
                text_questions = [q for q in level_questions if q['task_id'] not in self.file_questions]
                
                file_count = min(int(target_count * 0.7), len(file_questions))
                text_count = min(target_count - file_count, len(text_questions))
                
                # Verify files exist for file questions
                verified_file_questions = []
                for q in file_questions:
                    if self.find_file_for_question(q['task_id']):
                        verified_file_questions.append(q)
                
                selected.extend(verified_file_questions[:file_count])
                selected.extend(text_questions[:text_count])
        
        # Fill remaining slots with diverse questions
        remaining = size - len(selected)
        if remaining > 0:
            used_ids = {q['task_id'] for q in selected}
            available = [q for q in self.metadata if q['task_id'] not in used_ids]
            selected.extend(available[:remaining])
        
        return selected[:size]
    
    def _print_batch_summary(self, batch: List[Dict]):
        """Print summary of created batch"""
        
        if not batch:
            return
        
        # Level distribution
        level_counts = defaultdict(int)
        file_counts = {'with_files': 0, 'without_files': 0}
        file_types = defaultdict(int)
        
        for question in batch:
            level = question.get('Level', 'Unknown')
            level_counts[level] += 1
            
            if question['task_id'] in self.file_questions:
                file_counts['with_files'] += 1
                file_name = self.file_questions[question['task_id']].get('file_name', '')
                if file_name:
                    ext = file_name.split('.')[-1].lower()
                    file_types[ext] += 1
            else:
                file_counts['without_files'] += 1
        
        print(f"📊 Batch Summary:")
        print(f"├── By Level:")
        for level in sorted(level_counts.keys()):
            print(f"│   ├── Level {level}: {level_counts[level]} questions")
        
        print(f"├── File Attachments:")
        print(f"│   ├── With files: {file_counts['with_files']}")
        print(f"│   └── Without files: {file_counts['without_files']}")
        
        if file_types:
            print(f"└── File Types:")
            for file_type, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    ├── .{file_type}: {count} questions")
    
    # ============================================================================
    # DATASET UTILITIES
    # ============================================================================
    
    def export_test_batch(self, questions: List[Dict], output_path: str, format: str = "jsonl"):
        """Export test batch for external use"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "jsonl":
            with open(output_path, 'w') as f:
                for question in questions:
                    json.dump(question, f)
                    f.write('\n')
        
        elif format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(questions, f, indent=2)
        
        elif format.lower() == "csv":
            df = pd.DataFrame([
                {
                    'task_id': q['task_id'],
                    'question': q.get('Question', ''),
                    'level': q.get('Level'),
                    'has_file': q['task_id'] in self.file_questions,
                    'file_name': self.file_questions.get(q['task_id'], {}).get('file_name', ''),
                    'expected_answer': q.get('Final answer', '')
                }
                for q in questions
            ])
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📄 Exported {len(questions)} questions to {output_path}")
    
    def generate_dataset_report(self) -> str:
        """Generate comprehensive dataset analysis report"""
        
        if not self.metadata:
            return "No dataset loaded"
        
        analysis = self.analyze_dataset_distribution()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""
📊 GAIA Dataset Analysis Report
{'='*50}

📅 Generated: {timestamp}
📁 Dataset Path: {self.dataset_path}
📋 Total Questions: {analysis['total_questions']}
📎 Questions with Files: {analysis['questions_with_files']}

📈 Level Distribution:
{'-'*30}"""
        
        for level, count in sorted(analysis['level_distribution'].items()):
            percentage = count / analysis['total_questions'] * 100
            report += f"\n  Level {level}: {count} questions ({percentage:.1f}%)"
        
        if analysis['file_type_distribution']:
            report += f"\n\n📁 File Type Distribution:\n{'-'*30}"
            
            for file_type, count in sorted(analysis['file_type_distribution'].items(), 
                                         key=lambda x: x[1], reverse=True):
                percentage = count / analysis['questions_with_files'] * 100
                report += f"\n  .{file_type}: {count} files ({percentage:.1f}%)"
        
        if analysis['complexity_indicators']:
            report += f"\n\n🧠 Complexity Indicators:\n{'-'*30}"
            
            for complexity, count in analysis['complexity_indicators'].items():
                percentage = count / analysis['total_questions'] * 100
                report += f"\n  {complexity.title()}: {count} questions ({percentage:.1f}%)"
        
        if analysis['question_length_stats']:
            stats = analysis['question_length_stats']
            report += f"\n\n📏 Question Length Statistics:\n{'-'*30}"
            report += f"\n  Average: {stats['avg']:.0f} characters"
            report += f"\n  Range: {stats['min']} - {stats['max']} characters"
            report += f"\n  Median: {stats['median']} characters"
        
        # Dataset readiness assessment
        file_availability = 0
        if self.file_questions:
            available_files = sum(1 for task_id in self.file_questions.keys() 
                                if self.find_file_for_question(task_id))
            file_availability = available_files / len(self.file_questions)
        
        report += f"\n\n✅ Dataset Readiness:\n{'-'*30}"
        report += f"\n  File Availability: {file_availability:.1%}"
        report += f"\n  Metadata Integrity: {'✅ Good' if analysis['total_questions'] > 0 else '❌ Issues'}"
        
        if file_availability < 0.8:
            report += f"\n  ⚠️ Some referenced files are missing"
        
        report += f"\n\n{'='*50}"
        report += f"\nDataset ready for testing: {'✅ Yes' if file_availability > 0.7 else '⚠️ With limitations'}"
        
        return report


# ============================================================================
# DATA-ONLY UTILITY FUNCTIONS
# ============================================================================

def quick_dataset_check(dataset_path: str = "./tests/gaia_data") -> bool:
    """Validate dataset without running any agents - PURE DATA VALIDATION"""
    
    manager = GAIADatasetManager(dataset_path)
    
    if not manager.metadata:
        print("❌ Dataset not found or invalid")
        return False
    
    print(f"\n📊 Local GAIA Dataset Check")
    print("=" * 60)
    print(f"🔍 Checking metadata file: {os.path.join(manager.dataset_path, 'metadata.json')}")
    print(f"📁 Scanning data directory: {manager.dataset_path}")
    print(f"✅ Successfully loaded GAIA dataset from: {manager.dataset_path}")
    
    # Basic validation
    print(f"📋 Total questions in metadata.json: {len(manager.metadata)}")
    print(f"📎 Questions with file attachments: {len(manager.file_questions)}")
    
    # Validate file availability
    if manager.file_questions:
        available_files = 0
        sample_questions = list(manager.file_questions.items())[:5]
        
        print(f"\n📁 File Availability Check (sample of {len(sample_questions)}):")
        for task_id, question in sample_questions:
            file_path = manager.find_file_for_question(task_id)
            file_name = question.get('file_name', 'unknown')
            
            if file_path:
                available_files += 1
                print(f"  ✅ {file_name} -> Found")
            else:
                print(f"  ❌ {file_name} -> Missing")
        
        availability_rate = available_files / len(sample_questions)
        print(f"\n📈 File availability rate: {availability_rate:.1%}")
        
        return availability_rate > 0.5
    else:
        print(f"\n📝 Text-only dataset (no file dependencies)")
        return True

def compare_dataset_versions(path1: str, path2: str) -> Dict:
    """Compare two dataset versions"""
    
    manager1 = GAIADatasetManager(path1)
    manager2 = GAIADatasetManager(path2)
    
    if not manager1.metadata or not manager2.metadata:
        return {"error": "Could not load one or both datasets"}
    
    analysis1 = manager1.analyze_dataset_distribution()
    analysis2 = manager2.analyze_dataset_distribution()
    
    comparison = {
        'dataset1': {'path': path1, 'analysis': analysis1},
        'dataset2': {'path': path2, 'analysis': analysis2},
        'differences': {
            'total_questions': analysis2['total_questions'] - analysis1['total_questions'],
            'questions_with_files': analysis2['questions_with_files'] - analysis1['questions_with_files']
        }
    }
    
    return comparison

def generate_test_batches(dataset_path: str, 
                         batch_configs: List[Dict]) -> Dict[str, List[Dict]]:
    """Generate multiple test batches for different testing scenarios"""
    
    manager = GAIADatasetManager(dataset_path)
    
    if not manager.metadata:
        return {}
    
    batches = {}
    
    for config in batch_configs:
        batch_name = config.get('name', 'unnamed_batch')
        size = config.get('size', 10)
        strategy = config.get('strategy', 'balanced')
        
        batch = manager.create_test_batch(size, strategy, **config)
        batches[batch_name] = batch
        
        print(f"📦 Created batch '{batch_name}': {len(batch)} questions")
    
    return batches

def validate_gaia_format(answer: str) -> Dict:
    """Validate answer format against GAIA requirements - PURE VALIDATION"""
    
    validation = {
        'is_valid': True,
        'issues': [],
        'suggestions': []
    }
    
    if not answer or not answer.strip():
        validation['is_valid'] = False
        validation['issues'].append("Empty answer")
        return validation
    
    answer_clean = answer.strip()
    
    # GAIA format checks
    if answer_clean.lower().startswith('final answer:'):
        validation['suggestions'].append("Remove 'Final Answer:' prefix for submission")
        answer_clean = re.sub(r'^final answer:\s*', '', answer_clean, flags=re.IGNORECASE)
    
    # Check for formatting issues
    if answer_clean.endswith(('.', ',', '!', '?')):
        validation['issues'].append("Contains trailing punctuation")
    
    if answer_clean.lower().startswith(('the ', 'a ', 'an ')):
        validation['issues'].append("Contains article (the, a, an)")
    
    # Check for units in numbers (GAIA prefers no units unless specified)
    if re.search(r'\d+\s*[%$€£¥]', answer_clean):
        validation['suggestions'].append("Consider removing units unless specified in question")
    
    if validation['issues']:
        validation['is_valid'] = False
    
    return validation


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("📊 GAIA Dataset Utils - Pure Data Layer")
    print("=" * 50)
    
    # Example usage
    print("\n🔍 Dataset Validation:")
    dataset_ready = quick_dataset_check()
    
    if dataset_ready:
        print("\n📦 Creating Sample Test Batches:")
        
        manager = GAIADatasetManager()
        
        if manager.metadata:
            # Create different types of batches
            small_batch = manager.create_test_batch(5, "small_sample")
            balanced_batch = manager.create_test_batch(15, "balanced")
            level1_batch = manager.create_test_batch(10, "level_focused", target_level=1)
            
            # Generate dataset report
            print("\n📄 Generating Dataset Report:")
            report = manager.generate_dataset_report()
            print(report)
            
            # Example batch export
            if small_batch:
                print("\n💾 Exporting Sample Batch:")
                manager.export_test_batch(small_batch, "sample_batch.jsonl", "jsonl")
        
        else:
            print("❌ Could not load dataset for demonstration")
    
    print("\n✅ Dataset utilities ready for use with testing framework!")