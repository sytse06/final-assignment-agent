# gaia_dataset_utils.py - MINIMAL FIX VERSION
# Preserves all existing features, just adds ground truth separation

import json
import os
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict
import random

class GAIADatasetManager:
    """
    Manages GAIA dataset with blind testing compliance.
    
    MINIMAL FIX: Added ground truth separation to prevent contamination
    while preserving all existing dataset discovery and file handling features.
    """
    
    def __init__(self, dataset_path: str = None):
        """Initialize dataset manager and load metadata"""
        self.dataset_path = self._find_dataset_path(dataset_path)
        self.metadata = []
        self.file_questions = {}
        self.ground_truth = {}  # NEW: Separate ground truth storage
        self.test_metadata = None  # For test set if available
        
        if self.dataset_path:
            self._load_metadata()
    
    def _find_dataset_path(self, provided_path: str = None) -> Optional[str]:
        """Find GAIA dataset path in common locations - UPDATED for working metadata.json"""
        if provided_path and os.path.exists(provided_path):
            return provided_path
        
        # UPDATED: Check for working metadata.json FIRST
        common_paths = [
            "./metadata.json",  # Your working metadata file
            "../metadata.json",  # If running from subdirectory
            "./gaia_data",
            "./test_data/gaia", 
            "./data/gaia",
            os.path.expanduser("~/.cache/huggingface/datasets/gaia-benchmark___gaia/2023_all/0.0.1/gaia_data"),
            os.path.expanduser("~/.cache/huggingface/datasets/gaia-benchmark___gaia"),
            "./tests/gaia_data",
            "../tests/gaia_data",
            os.path.expanduser("~/Documents/gaia_dataset")
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                # Check for your working metadata.json format
                if path.endswith('.json'):
                    print(f"✅ Found working metadata file: {path}")
                    return path
                else:
                    # Check for directory with metadata files
                    metadata_files = ["metadata.json", "dataset_info.json", "gaia_validation.jsonl"]
                    for metadata_file in metadata_files:
                        full_path = os.path.join(path, metadata_file)
                        if os.path.exists(full_path):
                            print(f"✅ Found GAIA dataset at: {path}")
                            return path
        
        print("⚠️  GAIA dataset not found. Please specify path manually.")
        return None

    def _load_metadata(self):
        """Load and parse metadata - UPDATED for working metadata.json format"""
        
        # Handle direct JSON file (your working format)
        if self.dataset_path.endswith('.json'):
            metadata_file = self.dataset_path
        else:
            # Handle directory with metadata files
            metadata_files = ["metadata.json", "dataset_info.json", "gaia_validation.jsonl"]
            metadata_file = None
            for filename in metadata_files:
                potential_file = os.path.join(self.dataset_path, filename)
                if os.path.exists(potential_file):
                    metadata_file = potential_file
                    break
            
            if not metadata_file:
                print(f"❌ No metadata file found in {self.dataset_path}")
                return
        
        print(f"📁 Loading metadata from: {metadata_file}")
        
        try:
            with open(metadata_file, 'r') as f:
                raw_metadata = json.load(f)
            
            # Handle YOUR working format: {"validation_examples": [...], "test_examples": [...]}
            if isinstance(raw_metadata, dict) and 'validation_examples' in raw_metadata:
                print(f"📊 Working Metadata Format Detected:")
                self.metadata = raw_metadata['validation_examples']
                self.test_metadata = raw_metadata.get('test_examples', [])
                print(f"  Validation examples: {len(self.metadata)} questions")
                print(f"  Test examples: {len(self.test_metadata)} questions")
                
            # Handle HuggingFace cache format: {'validation': [...], 'test': [...]}
            elif isinstance(raw_metadata, dict) and 'validation' in raw_metadata:
                print(f"📊 HuggingFace Cache Format Detected:")
                self.metadata = raw_metadata['validation']
                self.test_metadata = raw_metadata.get('test', [])
                print(f"  Validation set: {len(self.metadata)} questions")
                print(f"  Test set: {len(self.test_metadata)} questions")
                
            elif isinstance(raw_metadata, list):
                print(f"📊 Simple List Format Detected:")
                self.metadata = raw_metadata
                print(f"  Total questions: {len(self.metadata)}")
                
            elif isinstance(raw_metadata, dict):
                # Handle other dict structures
                if 'questions' in raw_metadata:
                    self.metadata = raw_metadata['questions']
                elif 'data' in raw_metadata:
                    self.metadata = raw_metadata['data']
                else:
                    print(f"⚠️  Unexpected dict structure, trying to extract values...")
                    # Try to find the largest list in the dict
                    largest_list = []
                    for key, value in raw_metadata.items():
                        if isinstance(value, list) and len(value) > len(largest_list):
                            largest_list = value
                            print(f"  Using '{key}' with {len(value)} items")
                    self.metadata = largest_list
            else:
                print(f"❌ Unexpected metadata structure: {type(raw_metadata)}")
                return
            
            # Process the metadata
            self.file_questions = {}
            self.ground_truth = {}
            
            for item in self.metadata:
                if isinstance(item, dict):
                    task_id = item.get('task_id')
                    
                    # Check for file references
                    if ('file_name' in item and item['file_name']) or ('file_path' in item and item['file_path']):
                        self.file_questions[task_id] = item
                    
                    # Extract ground truth separately (handles both 'Final answer' and 'final_answer')
                    answer_field = item.get('Final answer') or item.get('final_answer')
                    if task_id and answer_field:
                        self.ground_truth[task_id] = {
                            'final_answer': answer_field,
                            'level': item.get('Level'),
                            'annotator_metadata': item.get('Annotator Metadata', {})
                        }
            
            print(f"✅ Loaded {len(self.metadata)} GAIA questions")
            print(f"📁 Found {len(self.file_questions)} questions with files")
            print(f"🎯 Found {len(self.ground_truth)} questions with ground truth")
            
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
            import traceback
            traceback.print_exc()
    
    def _ensure_file_path_preserved(self, questions: List[Dict]) -> List[Dict]:
        """
        This method takes a list of questions and ensures each one has the file_path 
        from the dataset.
        """
        enhanced_questions = []
        
        for question in questions:
            # Start with the question as-is
            enhanced_question = question.copy()
            
            # If it has a file but no file_path, get it from original metadata
            task_id = question.get('task_id')
            has_file = question.get('file_name') or (task_id in self.file_questions)
            
            if has_file and task_id:
                # Check if file_path is already present and valid
                current_file_path = enhanced_question.get('file_path', '')
                
                if not current_file_path:
                    # Get file_path from original metadata
                    if task_id in self.file_questions:
                        original_file_path = self.file_questions[task_id].get('file_path', '')
                        if original_file_path:
                            enhanced_question['file_path'] = original_file_path
                            print(f"   ✅ Added file_path for {task_id}")
                        else:
                            print(f"   ⚠️  No file_path found in metadata for {task_id}")
                    else:
                        # Fallback: search in full metadata
                        for metadata_item in self.metadata:
                            if metadata_item.get('task_id') == task_id:
                                original_file_path = metadata_item.get('file_path', '')
                                if original_file_path:
                                    enhanced_question['file_path'] = original_file_path
                                    print(f"   ✅ Added file_path for {task_id} (from full search)")
                                break
                        else:
                            print(f"   ❌ Could not find file_path for {task_id}")
                else:
                    print(f"   ✅ file_path already present for {task_id}")
            
            enhanced_questions.append(enhanced_question)
        
        return enhanced_questions
    
    def create_test_batch(self, num_questions: int = 20, strategy: str = "balanced", 
                         has_files: Optional[bool] = None, level: Optional[int] = None) -> List[Dict]:
        """
        Create test batch with BLIND TESTING compliance.
        
        CRITICAL FIX: Now removes "Final answer" field to prevent ground truth contamination.
        All other functionality preserved exactly as before.
        """
        if not self.metadata:
            print("❌ No metadata loaded")
            return []
        
        print(f"📦 Created {strategy} batch: {num_questions} questions")
        
        # Apply filters (existing logic preserved)
        filtered_questions = self.metadata.copy()
        
        if has_files is not None:
            if has_files:
                filtered_questions = [q for q in filtered_questions if q.get('file_name')]
            else:
                filtered_questions = [q for q in filtered_questions if not q.get('file_name')]
        
        if level is not None:
            filtered_questions = [q for q in filtered_questions if q.get('Level') == level]
        
        if not filtered_questions:
            print("❌ No questions match filters")
            return []
        
        # Selection strategies (existing logic preserved)
        if strategy == "balanced":
            selected = self._balanced_selection(filtered_questions, num_questions)
        elif strategy == "file_type_diverse":
            selected = self._file_type_diverse_selection(filtered_questions, num_questions)
        elif strategy == "small_sample":
            selected = self._small_sample_selection(filtered_questions, min(num_questions, 5))
        elif strategy == "large_comprehensive":
            selected = self._large_comprehensive_selection(filtered_questions, num_questions)
        else:
            # Random selection fallback
            selected = random.sample(filtered_questions, min(num_questions, len(filtered_questions)))
        
        # NEW: Create blind versions (remove contamination)
        blind_questions = self._create_blind_questions(selected)
        
        # Preserve file path functionality
        blind_questions = self._ensure_file_path_preserved(blind_questions)
        
        self._print_batch_summary(blind_questions)
        return blind_questions
    
    def _create_blind_questions(self, selected_questions: List[Dict]) -> List[Dict]:
        """
        NEW METHOD: Remove ground truth contamination for blind testing.
        
        This is the key fix - removes "Final answer" and other contamination fields
        while preserving all other question data.
        """
        blind_questions = []
        
        for question in selected_questions:
            # Create clean copy
            blind_question = {}
            
            # Copy all fields EXCEPT contamination fields
            contamination_fields = {
                'Final answer', 'final_answer', 'answer', 'solution',
                'ground_truth', 'expected_answer', 'correct_answer'
            }
            
            for field, value in question.items():
                if field not in contamination_fields:
                    blind_question[field] = value
                else:
                    # Log removal for transparency
                    print(f"🔒 Removing {field} from {question.get('task_id', 'unknown')} (blind testing)")
            
            blind_questions.append(blind_question)
        
        return blind_questions
    
    def _balanced_selection(self, questions: List[Dict], num_questions: int) -> List[Dict]:
        """Existing balanced selection logic (unchanged)"""
        level_questions = defaultdict(list)
        
        for q in questions:
            level = q.get('Level', 1)
            level_questions[level].append(q)
        
        # Target distribution: 40% L1, 40% L2, 20% L3
        target_l1 = int(num_questions * 0.4)
        target_l2 = int(num_questions * 0.4)
        target_l3 = num_questions - target_l1 - target_l2
        
        selected = []
        
        for level, target in [(1, target_l1), (2, target_l2), (3, target_l3)]:
            available = level_questions.get(level, [])
            if available:
                count = min(target, len(available))
                selected.extend(random.sample(available, count))
        
        # Fill remaining slots if needed
        remaining = num_questions - len(selected)
        if remaining > 0:
            all_remaining = [q for q in questions if q not in selected]
            if all_remaining:
                selected.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))
        
        return selected[:num_questions]
    
    def _file_type_diverse_selection(self, questions: List[Dict], num_questions: int) -> List[Dict]:
        """Existing file type diverse selection logic (unchanged)"""
        file_type_questions = defaultdict(list)
        text_only_questions = []
        
        for q in questions:
            file_name = q.get('file_name', '')
            if file_name:
                ext = Path(file_name).suffix.lower()
                file_type_questions[ext or 'no_extension'].append(q)
            else:
                text_only_questions.append(q)
        
        selected = []
        
        # Select one from each file type
        for ext, ext_questions in file_type_questions.items():
            if len(selected) >= num_questions:
                break
            selected.append(random.choice(ext_questions))
        
        # Add text-only questions
        remaining = num_questions - len(selected)
        if remaining > 0 and text_only_questions:
            text_sample = random.sample(text_only_questions, min(remaining // 2, len(text_only_questions)))
            selected.extend(text_sample)
        
        # Fill remaining slots
        remaining = num_questions - len(selected)
        if remaining > 0:
            all_remaining = [q for q in questions if q not in selected]
            if all_remaining:
                selected.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))
        
        return selected[:num_questions]
    
    def _small_sample_selection(self, questions: List[Dict], num_questions: int) -> List[Dict]:
        """Existing small sample selection logic (unchanged)"""
        verified_file_questions = []
        text_questions = []
        
        for q in questions:
            file_path = q.get('file_path', '')
            if file_path and Path(file_path).exists():
                verified_file_questions.append(q)
            elif not q.get('file_name'):
                text_questions.append(q)
        
        selected = []
        
        if verified_file_questions:
            file_count = min(num_questions // 2, len(verified_file_questions))
            selected.extend(random.sample(verified_file_questions, file_count))
        
        remaining = num_questions - len(selected)
        if remaining > 0 and text_questions:
            text_count = min(remaining, len(text_questions))
            selected.extend(random.sample(text_questions, text_count))
        
        return selected[:num_questions]
    
    def _large_comprehensive_selection(self, questions: List[Dict], num_questions: int) -> List[Dict]:
        """Existing large comprehensive selection logic (unchanged)"""
        return self._balanced_selection(questions, num_questions)
    
    def _print_batch_summary(self, questions: List[Dict]):
        """Print batch creation summary (enhanced with contamination check)"""
        file_questions = [q for q in questions if q.get('file_name')]
        file_questions_with_path = [q for q in file_questions if q.get('file_path')]
        
        level_counts = defaultdict(int)
        contamination_check = 0
        
        for q in questions:
            level = q.get('Level', 'Unknown')
            level_counts[level] += 1
            
            # Check for contamination
            if 'Final answer' in q or 'final_answer' in q:
                contamination_check += 1
        
        print(f"📁 File questions: {len(file_questions)}")
        print(f"📁 File questions with path: {len(file_questions_with_path)}")
        
        print(f"📊 Batch Summary:")
        print(f"├── By Level:")
        for level, count in sorted(level_counts.items()):
            print(f"│   ├── Level {level}: {count} questions")
        
        print(f"├── File Attachments:")
        print(f"│   ├── With files: {len(file_questions)}")
        print(f"│   └── Without files: {len(questions) - len(file_questions)}")
        
        # NEW: Contamination verification
        if contamination_check == 0:
            print(f"✅ Blind testing verified: No ground truth contamination")
        else:
            print(f"🚨 WARNING: {contamination_check} questions still contain contamination!")
    
    def get_ground_truth(self, task_id: str) -> Optional[Dict]:
        """
        NEW METHOD: Get ground truth for evaluation phase.
        
        This provides clean access to answers during evaluation without
        compromising blind testing during execution.
        """
        return self.ground_truth.get(task_id)
    
    def get_question_by_id(self, task_id: str) -> Optional[Dict]:
        """Get complete question data by task ID (existing functionality)"""
        for question in self.metadata:
            if question.get('task_id') == task_id:
                return question
        return None
    
    def test_file_path_preservation(self, num_test_questions: int = 3) -> bool:
        """Test that file paths are properly preserved (existing functionality)"""
        print(f"🧪 Testing file path preservation with {num_test_questions} questions...")
        
        test_batch = self.create_test_batch(num_test_questions, "file_type_diverse")
        file_questions = [q for q in test_batch if q.get('file_name')]
        
        if not file_questions:
            print("✅ No file questions in test batch (expected for text-only datasets)")
            return True
        
        success = True
        for q in file_questions:
            task_id = q.get('task_id')
            file_name = q.get('file_name')
            file_path = q.get('file_path')
            
            if not file_path:
                print(f"❌ Missing file_path for {task_id} ({file_name})")
                success = False
            elif not Path(file_path).exists():
                print(f"⚠️  File not found: {file_path}")
            else:
                print(f"✅ File accessible: {task_id}")
        
        return success
    
    def analyze_dataset_distribution(self) -> Dict:
        """Analyze dataset for comprehensive statistics (existing functionality)"""
        if not self.metadata:
            return {"error": "No metadata loaded"}
        
        analysis = {
            "total_questions": len(self.metadata),
            "questions_with_files": len(self.file_questions),
            "questions_with_answers": len(self.ground_truth),
            "level_distribution": defaultdict(int),
            "file_type_distribution": defaultdict(int)
        }
        
        for question in self.metadata:
            level = question.get('Level', 'Unknown')
            analysis["level_distribution"][level] += 1
            
            file_name = question.get('file_name', '')
            if file_name:
                ext = Path(file_name).suffix.lower()
                analysis["file_type_distribution"][ext or 'no_extension'] += 1
        
        analysis["level_distribution"] = dict(analysis["level_distribution"])
        analysis["file_type_distribution"] = dict(analysis["file_type_distribution"])
        
        return analysis


def quick_dataset_check(dataset_path: str = "./tests/gaia_data") -> bool:
    """Quick validation of dataset without agent execution"""
    print(f"🧪 Quick GAIA Dataset Check: {dataset_path}")
    
    try:
        manager = GAIADatasetManager(dataset_path)
        
        if not manager.metadata:
            print("❌ No questions loaded")
            return False
        
        # Test batch creation with contamination check
        test_batch = manager.create_test_batch(3, "small_sample")
        
        if not test_batch:
            print("❌ Failed to create test batch")
            return False
        
        # Verify blind testing compliance
        contamination_found = False
        for question in test_batch:
            if 'Final answer' in question or 'final_answer' in question:
                print(f"❌ Ground truth contamination detected in {question.get('task_id')}")
                contamination_found = True
        
        if contamination_found:
            print("❌ Dataset check failed - contamination detected")
            return False
        
        print("✅ Dataset check passed - ready for blind testing")
        return True
        
    except Exception as e:
        print(f"❌ Dataset check failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 GAIA Dataset Manager - MINIMAL FIX VERSION")
    print("=" * 50)
    print("✅ Preserves existing dataset discovery features")
    print("✅ Preserves existing file path handling")
    print("🔒 Adds ground truth separation for blind testing")
    print("🔒 Removes 'Final answer' field from test batches")
    print("")
    
    success = quick_dataset_check(os.path.expanduser("~/.cache/huggingface/datasets/gaia-benchmark___gaia/2023_all/0.0.1/gaia_data"))
    
    if success:
        print("\n🎯 Dataset utilities ready for agent testing!")
    else:
        print("\n❌ Check dataset setup")