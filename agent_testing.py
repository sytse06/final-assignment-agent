# agent_testing.py - COMPLETE TESTING FRAMEWORK (All Classes Restored)
# Maintains backward compatibility while adding improvements

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
    get_groq_config, 
    get_openrouter_config, 
    get_google_config, 
    get_ollama_config, 
    get_performance_config, 
    get_accuracy_config
)
from agent_logging import create_timestamped_filename

# Try to import dataset management (graceful degradation if not available)
try:
    from gaia_dataset_utils import GAIADatasetManager, quick_dataset_check
    DATASET_UTILS_AVAILABLE = True
    print("✅ GAIA dataset utilities available for complete testing")
except ImportError:
    print("⚠️  GAIA dataset utilities not available - limited testing mode")
    DATASET_UTILS_AVAILABLE = False

# ============================================================================
# TESTING CONFIGURATION
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

# ============================================================================
# CONFIGURATION HELPERS (Enhanced)
# ============================================================================

# agent_testing.py - COMPLETE FIX for GAIAConfig handling

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
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
    get_groq_config, 
    get_openrouter_config, 
    get_google_config, 
    get_ollama_config, 
    get_performance_config, 
    get_accuracy_config
)
from agent_logging import create_timestamped_filename

# Try to import dataset management (graceful degradation if not available)
try:
    from gaia_dataset_utils import GAIADatasetManager, quick_dataset_check
    DATASET_UTILS_AVAILABLE = True
    print("✅ GAIA dataset utilities available for complete testing")
except ImportError:
    print("⚠️  GAIA dataset utilities not available - limited testing mode")
    DATASET_UTILS_AVAILABLE = False

# ============================================================================
# Configuration Management Functions
# ============================================================================

def get_agent_config_by_name(config_name: str) -> GAIAConfig:
    """Get agent configuration by name - COMPLETELY FIXED"""
    
    config_functions = {
        "groq": get_groq_config,
        "qwen": get_groq_config,
        "qwen3_32b": get_groq_config,
        "google": get_google_config,
        "gemini": get_google_config,
        "openrouter": get_openrouter_config,
        "or": get_openrouter_config,
        "ollama": get_ollama_config,
        "local": get_ollama_config,
        "performance": get_performance_config,
        "accuracy": get_accuracy_config
    }
    
    config_name_lower = config_name.lower()
    
    if config_name_lower in config_functions:
        # These functions return GAIAConfig objects directly - NO CONVERSION NEEDED!
        return config_functions[config_name_lower]()
    else:
        # Fallback to groq config
        print(f"⚠️  Unknown config name '{config_name}', using default groq config")
        return get_groq_config()

def debug_agent_config_creation(config_name: str):
    """Debug helper to understand config creation issues"""
    print(f"🔧 DEBUG: Testing config creation for '{config_name}'")
    print("=" * 50)
    
    try:
        # Test the interface function directly
        if config_name.lower() == "ollama":
            config = get_ollama_config()
        elif config_name.lower() == "groq":
            config = get_groq_config()
        else:
            config = get_groq_config()  # Fallback
        
        print(f"✅ Config type: {type(config)}")
        print(f"✅ Is GAIAConfig: {isinstance(config, GAIAConfig)}")
        
        # FIXED: Access attributes directly instead of using .get()
        if isinstance(config, GAIAConfig):
            print(f"✅ Model provider: {config.model_provider}")
            print(f"✅ Model name: {config.model_name}")
            print(f"✅ Smart routing: {config.enable_smart_routing}")
        
        # Test agent creation
        agent = create_gaia_agent(config)
        print(f"✅ Agent created successfully: {type(agent)}")
        
        # Test executor creation with string name (not config object)
        executor = GAIATestExecutor(config_name)  # Pass string, not config object
        print(f"✅ Executor created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# PHASE 1: BLIND EXECUTION (No Ground Truth Access)
# ============================================================================

class GAIAQuestionExecutor:
    """
    Execute GAIA questions WITHOUT access to ground truth to ensures 
    blind testing.
    """
    
    def __init__(self, agent_config: Union[str, GAIAConfig], test_config=None):
        """
        Initialize test executor - FIXED to handle GAIAConfig objects properly
        
        Args:
            agent_config: Can be config name (str) or GAIAConfig object
            test_config: Test execution parameters
        """
        
        # Proper handling of GAIAConfig objects
        if isinstance(agent_config, str):
            self.gaia_config = get_agent_config_by_name(agent_config)
            self.agent_config_name = agent_config
        elif isinstance(agent_config, GAIAConfig):
            self.gaia_config = agent_config
            # FIXED: Access attributes directly, not with .get()
            self.agent_config_name = f"{agent_config.model_provider}_{agent_config.model_name}"
        else:
            raise ValueError(f"Expected str or GAIAConfig, got {type(agent_config)}")
        
        # Create agent using the GAIAConfig object
        self.agent = create_gaia_agent(self.gaia_config)
        
        # Setup test configuration
        self.test_config = test_config or self._default_test_config()
        
        # Generate execution timestamp
        self.execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"✅ GAIATestExecutor initialized:")
        print(f"   Config: {self.agent_config_name}")
        # FIXED: Access attributes directly
        print(f"   Model: {self.gaia_config.model_provider}/{self.gaia_config.model_name}")
        print(f"   Smart Routing: {self.gaia_config.enable_smart_routing}")

    def _default_test_config(self):
        """Create default test configuration"""
        return {
            "batch_size": 20,
            "timeout_per_question": 120,
            "enable_detailed_logging": True,
            "save_execution_results": True,
            "results_directory": "./test_results"
        }
    
    def execute_questions_batch(self, blind_questions: List[Dict], batch_name: str = None) -> str:
        """
        Execute batch of questions WITHOUT ground truth access.
        
        Args:
            blind_questions: List of questions WITHOUT 'Final answer' field
            batch_name: Name for this batch execution
            
        Returns:
            Path to execution results file (for later evaluation)
        """
        if batch_name is None:
            batch_name = f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🚀 BLIND EXECUTION: {len(blind_questions)} questions")
        print(f"📝 Batch: {batch_name}")
        
        # Verify questions are truly blind (no ground truth)
        if self.test_config.enable_ground_truth_isolation:
            self._verify_blind_questions(blind_questions)
        
        execution_results = []
        start_time = time.time()
        
        for i, question_data in enumerate(blind_questions, 1):
            print(f"\n🎯 Question {i}/{len(blind_questions)}")
            
            result = self._execute_single_question_blind(question_data, i)
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
        """
        RESTORED: Verify questions don't contain ground truth
        
        Critical for GAIA benchmark compliance
        """
        contaminated_questions = []
        
        for i, q in enumerate(questions):
            # Check for ground truth contamination
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
            for contamination in contaminated_questions[:5]:  # Show first 5
                print(f"   ❌ {contamination}")
            
            if len(contaminated_questions) > 5:
                print(f"   ... and {len(contaminated_questions) - 5} more contaminations")
            
            if self.test_config.enable_ground_truth_isolation:
                raise ValueError(f"Blind testing compromised! {len(contaminated_questions)} questions contain ground truth.")
            else:
                print("   ⚠️ Continuing despite contamination (isolation disabled)")
        else:
            print("✅ Verified: Questions are properly blind (no ground truth contamination)")
    
    def _execute_single_question_blind(self, question_data: Dict, question_num: int) -> Dict:
        """Execute single question without ground truth access"""
        task_id = question_data.get('task_id', f"blind_{question_num}")
        question = question_data.get('Question', question_data.get('question', ''))
        level = question_data.get('Level', 'Unknown')
        
        start_time = time.time()
        
        try:
            print(f"   🔒 Processing (BLIND): {question[:50]}...")
            
            # CRITICAL: Execute question - agent has NO access to ground truth
            result = self.agent.process_question(question, task_id=task_id)
            
            execution_time = time.time() - start_time
            
            # Determine strategy used
            strategy_used = self._determine_strategy_used(result)
            
            # RESTORED: Complete blind execution record
            return {
                "question_number": question_num,
                "task_id": task_id,
                "question": question,
                "level": level,
                "final_answer": result.get("final_answer", ""),
                "raw_answer": result.get("raw_answer", ""),
                "steps": result.get("steps", []),
                "execution_successful": result.get("execution_successful", False),
                "execution_time": execution_time,
                "strategy_used": strategy_used,
                "complexity": result.get("complexity"),
                "similar_examples_count": len(result.get("similar_examples", [])),
                "context_bridge_used": result.get("context_bridge_used", False),
                "model_provider": self.agent_config.model_provider,
                "model_name": self.agent_config.model_name,
                "execution_timestamp": datetime.now().isoformat(),
                "blind_execution_verified": True  # RESTORED: Blind testing marker
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ Execution failed: {str(e)}")
            
            # RESTORED: Complete error record for blind testing
            return {
                "question_number": question_num,
                "task_id": task_id,
                "question": question,
                "level": level,
                "final_answer": "ERROR",
                "raw_answer": f"Execution error: {str(e)}",
                "steps": [],
                "execution_successful": False,
                "execution_time": execution_time,
                "strategy_used": "error",
                "complexity": "unknown",
                "similar_examples_count": 0,
                "context_bridge_used": False,
                "model_provider": self.agent_config.model_provider,
                "model_name": self.agent_config.model_name,
                "execution_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "blind_execution_verified": True  # RESTORED: Even errors are blind
            }
    
    def _determine_strategy_used(self, result: Dict) -> str:
        """Determine which strategy was used based on result analysis"""
        steps = result.get("steps", [])
        step_text = " ".join(steps).lower()
        
        if "one-shot" in step_text or "direct llm" in step_text:
            return "one_shot_llm"
        elif "manager coordination" in step_text or "manager execution" in step_text:
            return "manager_coordination"
        elif "error" in step_text:
            return "error"
        else:
            return "unknown"
    
    def _save_execution_results(self, results: List[Dict], batch_name: str, total_time: float) -> str:
        """Save execution results WITHOUT ground truth"""
        
        # Create timestamped filename using agent_logging utility
        base_filename = f"{batch_name}_execution"
        execution_file = os.path.join(
            self.test_config.results_dir,
            create_timestamped_filename(base_filename, "json")
        )
        
        execution_data = {
            "batch_name": batch_name,
            "execution_timestamp": datetime.now().isoformat(),
            "agent_config": {
                "provider": self.agent_config.model_provider,
                "model": self.agent_config.model_name,
                "temperature": self.agent_config.temperature,
                "smart_routing": self.agent_config.enable_smart_routing,
                "context_bridge": self.agent_config.enable_context_bridge
            },
            "execution_summary": {
                "total_questions": len(results),
                "successful_executions": sum(1 for r in results if r["execution_successful"]),
                "total_execution_time": total_time,
                "avg_execution_time": total_time / len(results) if results else 0
            },
            "results": results,
            "blind_testing_verified": True  # Confirms no ground truth contamination
        }
        
        try:
            with open(execution_file, 'w', encoding='utf-8') as f:
                json.dump(execution_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Execution results saved: {execution_file}")
            return execution_file
            
        except Exception as e:
            print(f"❌ Failed to save execution results: {e}")
            # Fallback to simple filename
            fallback_file = os.path.join(self.test_config.results_dir, f"{batch_name}_execution.json")
            with open(fallback_file, 'w', encoding='utf-8') as f:
                json.dump(execution_data, f, indent=2, ensure_ascii=False)
            return fallback_file

# ============================================================================
# ENHANCED AGENT EXECUTION (With Improved Error Detection) - RESTORED
# ============================================================================

class GAIATestExecutor:
    """
    Enhanced test executor with comprehensive error detection and real-time monitoring.
    This is the main executor class for production testing. Gaiaconfig objects are
    properly handled.
    """
    
    def __init__(self, agent_config: Union[str, Dict, GAIAConfig], test_config: GAIATestConfig = None):
        """Initialize tests with agent and test configurations"""
        
        self.test_config = test_config or GAIATestConfig()
        self.results_dir = Path(self.test_config.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Handle different config types
        if isinstance(agent_config, str):
            # Get GAIAConfig object from string name
            self.gaia_config = get_agent_config_by_name(agent_config)
            self.agent_config_name = agent_config
        elif isinstance(agent_config, GAIAConfig):
            # Already a GAIAConfig object
            self.gaia_config = agent_config
            # FIXED: Access attributes directly, not with .get()
            self.agent_config_name = f"{agent_config.model_provider}_{agent_config.model_name}"
        else:
            raise ValueError(f"Expected str or GAIAConfig, got {type(agent_config)}")
        
        try:
            self.agent = create_gaia_agent(self.gaia_config)
            print(f"✅ Agent created successfully with {self.gaia_config.model_provider}/{self.gaia_config.model_name}")
        except Exception as e:
            print(f"❌ Agent creation failed: {e}")
            raise
            
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
        """Execute agent on provided questions with comprehensive tracking"""
        
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
        
        task_id = question_data.get("task_id", str(uuid.uuid4()))
        question = question_data.get("Question", question_data.get("question", ""))
        level = question_data.get("Level", question_data.get("level", "unknown"))
        
        start_time = time.time()
        
        print(f"🔄 Executing: {task_id} (Level {level})")
        print(f"   Question: {question[:60]}...")
        
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
            
            # ENHANCED: Detect execution failures in agent response
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
        """ENHANCED: Detect execution failures in agent responses"""
        
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
# PHASE 2: EVALUATION (Ground Truth Comparison)
# ============================================================================

class GAIAAnswerEvaluator:
    """
    Phase 2: Evaluate execution results AGAINST ground truth.
    This phase has access to expected answers for comparison.
    """
    
    def __init__(self, dataset_manager: 'GAIADatasetManager', test_config: GAIATestConfig = None):
        if not dataset_manager:
            raise ValueError("Dataset manager required for evaluation phase")
        
        self.dataset_manager = dataset_manager
        self.test_config = test_config or GAIATestConfig()
        
        print(f"🎯 EVALUATOR initialized - HAS ground truth access")
        print(f"   Dataset: {dataset_manager.dataset_path}")
    
    def evaluate_execution_results(self, execution_file: str) -> Dict:
        """
        Evaluate execution results against ground truth.
        
        Args:
            execution_file: Path to execution results (from Phase 1)
            
        Returns:
            Comprehensive evaluation results
        """
        print(f"📊 EVALUATION PHASE: {execution_file}")
        
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
        
        results = execution_data.get("results", [])
        
        print(f"🔍 Evaluating {len(results)} agent responses against ground truth...")
        
        evaluated_results = []
        correct_answers = 0
        level_performance = defaultdict(lambda: {"total": 0, "correct": 0})
        
        for result in results:
            task_id = result.get("task_id")
            agent_answer = result.get("final_answer", "")
            level = result.get("level", "Unknown")
            
            try:
                # NOW we access ground truth (Phase 2 only)
                ground_truth = self.dataset_manager.get_ground_truth(task_id)
                
                if ground_truth:
                    # Perform GAIA-compliant answer matching
                    is_correct = self.gaia_answer_matching(agent_answer, ground_truth)
                    
                    if not is_correct:
                        # Try fuzzy matching as fallback
                        fuzzy_correct = self.fuzzy_answer_matching(agent_answer, ground_truth)
                        if fuzzy_correct:
                            is_correct = True
                            matching_method = "fuzzy"
                        else:
                            matching_method = "no_match"
                    else:
                        matching_method = "exact"
                    
                    if is_correct:
                        correct_answers += 1
                    
                    # Track level performance
                    level_performance[str(level)]["total"] += 1
                    if is_correct:
                        level_performance[str(level)]["correct"] += 1
                    
                    evaluated_result = result.copy()
                    evaluated_result.update({
                        "ground_truth": ground_truth,
                        "is_correct": is_correct,
                        "matching_method": matching_method,
                        "evaluation_timestamp": datetime.now().isoformat()
                    })
                    
                else:
                    print(f"⚠️  No ground truth found for task: {task_id}")
                    evaluated_result = result.copy()
                    evaluated_result.update({
                        "ground_truth": None,
                        "is_correct": None,
                        "matching_method": "no_ground_truth",
                        "evaluation_timestamp": datetime.now().isoformat()
                    })
                
                evaluated_results.append(evaluated_result)
                
            except Exception as e:
                print(f"⚠️  Evaluation error for {task_id}: {e}")
                traceback.print_exc()
                evaluated_result = result.copy()
                evaluated_result.update({
                    "ground_truth": None,
                    "is_correct": None,
                    "matching_method": "error",
                    "evaluation_error": str(e),
                    "evaluation_timestamp": datetime.now().isoformat()
                })
                evaluated_results.append(evaluated_result)
        
        # Calculate performance metrics
        accuracy = correct_answers / len(results) if results else 0
        
        # Calculate level-specific accuracy
        level_accuracy = {}
        for level, perf in level_performance.items():
            level_accuracy[level] = {
                "accuracy": perf["correct"] / perf["total"] if perf["total"] > 0 else 0,
                "correct": perf["correct"],
                "total": perf["total"]
            }
        
        evaluation_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "execution_file": execution_file,
            "batch_info": execution_data.get("batch_name", "Unknown"),
            "agent_config": execution_data.get("agent_config", {}),
            "overall_performance": {
                "total_questions": len(results),
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "successful_executions": execution_data.get("execution_summary", {}).get("successful_executions", 0)
            },
            "level_performance": level_accuracy,
            "strategy_analysis": self._analyze_strategy_performance(evaluated_results),
            "results": evaluated_results,
            "evaluation_verified": True
        }
        
        # Save evaluation results
        evaluation_file = self._save_evaluation_results(evaluation_results, execution_file)
        
        print(f"\n📊 EVALUATION COMPLETE")
        print(f"   Total Questions: {len(results)}")
        print(f"   Correct Answers: {correct_answers}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Evaluation saved: {evaluation_file}")
        
        return evaluation_results
    
    def _analyze_strategy_performance(self, results: List[Dict]) -> Dict:
        """Analyze performance by strategy used"""
        strategy_stats = {}
        
        for result in results:
            strategy = result.get("strategy_used", "unknown")
            is_correct = result.get("is_correct")
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"total": 0, "correct": 0}
            
            strategy_stats[strategy]["total"] += 1
            if is_correct:
                strategy_stats[strategy]["correct"] += 1
        
        # Calculate accuracy per strategy
        strategy_analysis = {}
        for strategy, stats in strategy_stats.items():
            strategy_analysis[strategy] = {
                "total_questions": stats["total"],
                "correct_answers": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            }
        
        return strategy_analysis
    
    def gaia_answer_matching(self, predicted: str, expected: str) -> bool:
        """GAIA-compliant exact matching"""
        if not predicted or not expected:
            return False
        
        # Normalize both answers
        pred_norm = self._normalize_gaia_answer(predicted)
        exp_norm = self._normalize_gaia_answer(expected)
        
        return pred_norm == exp_norm
    
    def fuzzy_answer_matching(self, predicted: str, expected: str, threshold: float = 0.8) -> bool:
        """Enhanced fuzzy string matching with multiple methods"""
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
        """Normalize answer for GAIA comparison"""
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
        """Save evaluation results with enhanced file handling"""
        base_name = os.path.basename(execution_file).replace("_execution.json", "")
        
        # Use timestamped filename
        evaluation_filename = create_timestamped_filename(f"{base_name}_evaluation", "json")
        evaluation_file = os.path.join(self.test_config.results_dir, evaluation_filename)
        
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
# ENHANCED TEST EVALUATOR (Production Ready) - RESTORED
# ============================================================================

class GAIATestEvaluator:
    """Enhanced test evaluator with production-ready features"""
    
    def __init__(self, dataset_manager: 'GAIADatasetManager'):
        """Initialize with dataset manager for ground truth access"""
        if not DATASET_UTILS_AVAILABLE:
            raise ImportError("GAIADatasetManager required for evaluation")
        
        self.dataset_manager = dataset_manager
        
        print(f"🎯 GAIA Test Evaluator Initialized")
        print(f"📊 Dataset: {len(dataset_manager.metadata)} questions available")
        print(f"🔧 Features: GAIA compliance, Enhanced matching, Performance analysis")
    
    def evaluate_execution_results(self, execution_results: List[Dict]) -> Dict:
        """Evaluate agent results against ground truth with enhanced metrics"""
        
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
                # Evaluate with enhanced metrics
                evaluation_result = self._evaluate_single_execution(execution, ground_truth)
                
                if evaluation_result['is_correct']:
                    correct_count += 1
                
                if evaluation_result['execution_successful']:
                    execution_success_count += 1
                
                # Print evaluation details
                self._print_evaluation_details(evaluation_result)
            
            evaluation_results.append(evaluation_result)
        
        # Generate comprehensive analysis
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
        """Evaluate single execution with enhanced metrics"""
        
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
        """Create comprehensive evaluation record"""
        
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
        """Enhanced fuzzy matching for near-correct answers"""
        
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
        
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Evaluation results saved: {filepath}")

# ============================================================================
# HIGH-LEVEL TESTING FUNCTIONS
# ============================================================================

def run_gaia_test(agent_config_name: str = "groq", dataset_path: str = "./tests/gaia_data", 
                  max_questions: int = 20, test_config: GAIATestConfig = None) -> Optional[Dict]:
    """
    COMPLETE BLIND TESTING WORKFLOW: Execute then Evaluate
    
    This is the main function for GAIA benchmark testing.
    """
    print(f"🎯 COMPLETE GAIA TEST: {agent_config_name}")
    print("=" * 60)
    print("📋 Two-Phase Blind Testing:")
    print("   Phase 1: Blind Execution (no ground truth)")
    print("   Phase 2: Evaluation (with ground truth)")
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
        
        # PHASE 1: Blind Execution
        print(f"\n🔒 PHASE 1: BLIND EXECUTION")
        print("-" * 30)
        
        agent_config = get_agent_config_by_name(agent_config_name)
        executor = GAIAQuestionExecutor(agent_config, test_config)
        
        batch_name = f"gaia_test_{agent_config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        execution_file = executor.execute_questions_batch(blind_questions, batch_name)
        
        # PHASE 2: Evaluation
        print(f"\n🎯 PHASE 2: EVALUATION")
        print("-" * 20)
        
        evaluator = GAIAAnswerEvaluator(dataset_manager, test_config)
        evaluation_results = evaluator.evaluate_execution_results(execution_file)
        
        print(f"\n🏆 COMPLETE GAIA TEST RESULTS")
        print("=" * 40)
        overall = evaluation_results["overall_performance"]
        print(f"📊 Overall Accuracy: {overall['accuracy']:.1%}")
        print(f"🎯 Correct Answers: {overall['correct_answers']}/{overall['total_questions']}")
        print(f"⚡ Execution Success: {overall['successful_executions']}/{overall['total_questions']}")
        
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
        return None

def run_quick_gaia_test(agent_config_name: str = "groq", **kwargs) -> Optional[Dict]:
    """Quick GAIA test with proper blind testing"""
    
    # Extract parameters with backward compatibility
    num_questions = kwargs.get('num_questions', kwargs.get('max_questions', 5))
    dataset_path = kwargs.get('dataset_path', './tests/gaia_data')
    
    print(f"🚀 Quick GAIA Test (Blind): {agent_config_name}")
    print(f"   Questions: {num_questions}")
    
    return run_gaia_test(
        agent_config_name=agent_config_name,
        dataset_path=dataset_path,
        max_questions=num_questions
    )

def run_small_batch_test(agent_config: Union[str, Dict] = "groq", 
                        dataset_path: str = "./tests/gaia_data") -> Dict:
    """Quick test with small diverse batch using enhanced executor"""
    
    if not DATASET_UTILS_AVAILABLE:
        print("❌ Dataset utilities required")
        return {}
    
    print(f"🚀 Small Batch Test: {agent_config}")
    
    # Use enhanced executor
    from gaia_dataset_utils import GAIADatasetManager
    dataset_manager = GAIADatasetManager(dataset_path)
    test_batch = dataset_manager.create_test_batch(5, "small_sample")
    
    if test_batch:
        executor = GAIATestExecutor(agent_config)
        execution_results = executor.execute_test_batch(test_batch)
        
        # Evaluate results
        evaluator = GAIATestEvaluator(dataset_manager)
        evaluation_results = evaluator.evaluate_execution_results(execution_results)
        
        return evaluation_results
    
    return {}

def run_large_batch_test(agent_config: Union[str, Dict] = "groq",
                        dataset_path: str = "./tests/gaia_data") -> Dict:
    """Comprehensive test with large batch using enhanced executor"""
    
    if not DATASET_UTILS_AVAILABLE:
        print("❌ Dataset utilities required")
        return {}
    
    print(f"🚀 Large Batch Test: {agent_config}")
    
    from gaia_dataset_utils import GAIADatasetManager
    dataset_manager = GAIADatasetManager(dataset_path)
    test_batch = dataset_manager.create_test_batch(25, "large_comprehensive")
    
    if test_batch:
        executor = GAIATestExecutor(agent_config)
        execution_results = executor.execute_test_batch(test_batch)
        
        # Evaluate results
        evaluator = GAIATestEvaluator(dataset_manager)
        evaluation_results = evaluator.evaluate_execution_results(execution_results)
        
        return evaluation_results
    
    return {}

def compare_agent_configs(config_names: List[str], num_questions: int = 10, 
                         dataset_path: str = "./tests/gaia_data") -> Dict:
    """Compare multiple agent configurations using blind testing"""
    
    print(f"🔄 AGENT COMPARISON: {len(config_names)} configs")
    print(f"   Configs: {', '.join(config_names)}")
    print(f"   Questions: {num_questions}")
    
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
                "strategy_analysis": result.get("strategy_analysis", {})
            }
            print(f"   ✅ {config_name}: {result['overall_performance']['accuracy']:.1%}")
        else:
            comparison_results[config_name] = {"error": "Test failed"}
            print(f"   ❌ {config_name}: Failed")
    
    return {
        "comparison_results": comparison_results,
        "timestamp": datetime.now().isoformat(),
        "test_questions": num_questions
    }

def run_smart_routing_test(agent_config_name: str = "performance") -> Dict:
    """Test smart routing behavior specifically"""
    
    print(f"🔀 Smart Routing Test: {agent_config_name}")
    
    result = run_gaia_test(
        agent_config_name=agent_config_name,
        max_questions=20
    )
    
    if result and 'analysis' in result:
        routing_analysis = result['analysis'].get('routing_analysis', {})
        strategy_performance = result['analysis'].get('strategy_performance', {})
        
        print(f"\n🔀 ROUTING ANALYSIS")
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
            accuracy = stats.get('overall_accuracy', 0)
            avg_time = stats.get('avg_execution_time', 0)
            total = stats.get('total_questions', 0)
            print(f"├── {strategy}: {accuracy:.1%} accuracy, {avg_time:.1f}s avg ({total} questions)")
    
    return result

def test_file_vs_text_performance(agent_config_name: str = "groq") -> Dict:
    """Compare performance on file vs text-only questions"""
    
    print(f"📎 File vs Text Performance Test: {agent_config_name}")
    
    if not DATASET_UTILS_AVAILABLE:
        print("❌ Dataset utilities required")
        return {}
    
    from gaia_dataset_utils import GAIADatasetManager
    dataset_manager = GAIADatasetManager("./tests/gaia_data")
    
    results = {}
    
    # Test 1: Text-only questions
    print(f"\n📝 Test 1: Text-Only Questions")
    text_batch = dataset_manager.create_test_batch(10, "balanced", has_files=False)
    
    if text_batch:
        executor = GAIATestExecutor(agent_config_name)
        execution_results = executor.execute_test_batch(text_batch)
        evaluator = GAIATestEvaluator(dataset_manager)
        results['text_only'] = evaluator.evaluate_execution_results(execution_results)
    
    # Test 2: File-based questions
    print(f"\n📎 Test 2: File-Based Questions")
    file_batch = dataset_manager.create_test_batch(10, "file_type_diverse")
    
    if file_batch:
        executor = GAIATestExecutor(agent_config_name)
        execution_results = executor.execute_test_batch(file_batch)
        evaluator = GAIATestEvaluator(dataset_manager)
        results['with_files'] = evaluator.evaluate_execution_results(execution_results)
    
    # Compare results
    if 'text_only' in results and 'with_files' in results:
        text_acc = results['text_only']['evaluation_metadata']['overall_accuracy']
        file_acc = results['with_files']['evaluation_metadata']['overall_accuracy']
        
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 30)
        print(f"📝 Text-only: {text_acc:.1%}")
        print(f"📎 With files: {file_acc:.1%}")
        print(f"📈 Difference: {(text_acc - file_acc)*100:+.1f} percentage points")
    
    return results

def diagnose_agent_issues(agent_config: Union[str, Dict] = "groq",
                         dataset_path: str = "./tests/gaia_data") -> Dict:
    """Diagnostic test to identify agent issues"""
    
    print(f"🔧 Agent Diagnostic Test: {agent_config}")
    
    if not DATASET_UTILS_AVAILABLE:
        print("❌ Dataset utilities required")
        return {}
    
    # Run test with maximum error detection
    test_config = GAIATestConfig(
        max_retries=1,  # Don't retry to see raw errors
        detect_execution_failures=True,
        enable_strategy_fallback=False,  # Don't mask with fallbacks
        enable_real_time_monitoring=True
    )
    
    # Create small diagnostic batch
    from gaia_dataset_utils import GAIADatasetManager
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
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_failure_patterns(evaluation_results: Dict) -> Dict:
    """Analyze failure patterns from evaluation results"""
    
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
    recommendations = _generate_improvement_recommendations(failure_patterns, incorrect_results)
    
    return {
        'failure_patterns': dict(failure_patterns),
        'recommendations': recommendations,
        'sample_failures': incorrect_results[:5]
    }

def _generate_improvement_recommendations(failure_patterns: Dict, incorrect_results: List[Dict]) -> List[str]:
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
# UTILITY FUNCTIONS
# ============================================================================

def test_config_types():
    """Test that all config functions return GAIAConfig objects"""
    
    print("🔧 TESTING CONFIG TYPES")
    print("=" * 30)
    
    config_functions = [
        ("groq", get_groq_config),
        ("google", get_google_config),
        ("openrouter", get_openrouter_config),
        ("ollama", get_ollama_config),
        ("performance", get_performance_config),
        ("accuracy", get_accuracy_config)
    ]
    
    for name, func in config_functions:
        try:
            config = func()
            is_gaia_config = isinstance(config, GAIAConfig)
            print(f"✅ {name:12} -> {type(config).__name__:>15} {'✅' if is_gaia_config else '❌'}")
            
            if is_gaia_config:
                print(f"   Provider: {config.model_provider}, Model: {config.model_name}")
            
        except Exception as e:
            print(f"❌ {name:12} -> ERROR: {e}")
    
    print("\n🎯 All config functions should return GAIAConfig objects!")

if __name__ == "__main__":
    print("🧪 AGENT TESTING FRAMEWORK - FIXED VERSION")
    print("=" * 50)
    
    # Test config types
    test_config_types()
    
    # Test executor creation
    print(f"\n🔧 Testing GAIATestExecutor creation...")
    try:
        executor = GAIATestExecutor("ollama")
        print("✅ GAIATestExecutor works correctly!")
    except Exception as e:
        print(f"❌ GAIATestExecutor failed: {e}")
        traceback.print_exc()

# ============================================================================
# MAIN EXECUTION AND EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("🧪 GAIA Testing Framework v3.0 - COMPLETE VERSION")
    print("=" * 60)
    print("✅ ALL CLASSES RESTORED:")
    print("   ├── GAIAQuestionExecutor (Blind execution)")
    print("   ├── GAIATestExecutor (Enhanced execution)")
    print("   ├── GAIAAnswerEvaluator (Blind testing evaluation)")
    print("   ├── GAIATestEvaluator (Production evaluation)")
    print("   └── All convenience functions")
    print("")
    print("🔧 Key Features:")
    features = [
        "✅ Backward compatibility maintained",
        "✅ Enhanced execution failure detection", 
        "✅ Real-time progress monitoring",
        "✅ Comprehensive error categorization",
        "✅ Strategy fallback mechanisms",
        "✅ GAIA-compliant answer matching",
        "✅ Blind testing workflow (Phase 1 + Phase 2)",
        "✅ Production-ready evaluation metrics",
        "✅ Detailed failure pattern analysis",
        "✅ Multi-configuration comparison tools"
    ]
    
    for feature in features:
        print(f"  {feature}")

    print(f"\n📋 Available Functions:")
    functions = [
        'run_gaia_test',
        'run_quick_gaia_test', 
        'run_small_batch_test',
        'run_large_batch_test',
        'compare_agent_configs',
        'run_smart_routing_test',
        'test_file_vs_text_performance',
        'diagnose_agent_issues',
        'analyze_failure_patterns'
    ]
    
    for func in functions:
        print(f"   ├── {func}")

    print(f"\n💡 Quick Start:")
    print(f"   from agent_testing import GAIATestExecutor, run_quick_gaia_test")
    print(f"   result = run_quick_gaia_test('groq')")
    print(f"   print(f\"Accuracy: {{result['evaluation_metadata']['overall_accuracy']:.1%}}\")")
    
    print(f"\n🎯 Import Test:")
    try:
        # Test import of key classes
        print("   Testing GAIATestExecutor import...")
        executor = GAIATestExecutor("groq")
        print("   ✅ GAIATestExecutor import successful")
        
        print("   Testing function availability...")
        if callable(run_quick_gaia_test):
            print("   ✅ run_quick_gaia_test available")
        
        print("   ✅ All imports working correctly!")
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        print("   💡 Check dependencies and agent_interface availability")
    