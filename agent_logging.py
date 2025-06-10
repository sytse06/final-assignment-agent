# agent_logging.py
# CSV logging functionality for GAIA agent

import csv
import datetime
import os
from pathlib import Path
from typing import Dict
from rich.console import Console
from smolagents import AgentLogger, LogLevel

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def create_timestamped_filename(base_name: str, extension: str = "csv") -> str:
    """Create timestamped filename in logs directory"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Remove extension from base_name if present
    base_name = base_name.replace(f".{extension}", "")
    
    filename = f"{base_name}_{timestamp}.{extension}"
    return str(logs_dir / filename)

def get_latest_log_file(pattern: str) -> str:
    """Get the most recent log file matching pattern"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    
    # Find all files matching pattern
    files = list(logs_dir.glob(f"{pattern}_*.csv"))
    if not files:
        return None
    
    # Return most recent file
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)

# ============================================================================
# CSV LOGGERS
# ============================================================================

class StepLogger:
    """Logs individual agent steps to CSV with timestamped filenames"""
    
    def __init__(self, log_file: str = None):
        if log_file is None:
            self.log_file = Path(create_timestamped_filename("gaia_steps"))
        else:
            # If custom filename provided, still put in logs directory with timestamp
            if not log_file.startswith("logs/"):
                self.log_file = Path(create_timestamped_filename(log_file.replace(".csv", "")))
            else:
                self.log_file = Path(log_file)
        
        self._ensure_csv_header()
        print(f"📝 Step logging to: {self.log_file}")
    
    def _ensure_csv_header(self):
        """Create CSV with header if it doesn't exist"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'task_id', 'step_number', 
                    'action', 'tool_name', 'input', 'output'
                ])
    
    def log_step(self, task_id: str, step_number: int, step_data: Dict):
        """Append single step to CSV"""
        timestamp = datetime.datetime.now().isoformat()
        
        row = [
            timestamp,
            task_id,
            step_number,
            step_data.get('action', ''),
            step_data.get('tool_name', ''),
            str(step_data.get('input', ''))[:200],  # Truncate long inputs
            str(step_data.get('output', ''))[:200]  # Truncate long outputs
        ]
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

class QuestionLogger:
    """Logs completed questions to CSV with timestamped filenames"""
    
    def __init__(self, log_file: str = None):
        if log_file is None:
            self.log_file = Path(create_timestamped_filename("gaia_questions"))
        else:
            # If custom filename provided, still put in logs directory with timestamp
            if not log_file.startswith("logs/"):
                self.log_file = Path(create_timestamped_filename(log_file.replace(".csv", "")))
            else:
                self.log_file = Path(log_file)
        
        self._ensure_header()
        print(f"📊 Question logging to: {self.log_file}")
    
    def _ensure_header(self):
        """Create CSV with header if it doesn't exist"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'task_id', 'question', 'final_answer', 
                    'total_steps', 'success', 'complexity', 'routing_path',
                    'execution_time', 'model_used', 'similar_examples_count'
                ])
    
    def log_question(self, task_id: str, question: str, final_answer: str, 
                    total_steps: int, success: bool, complexity: str = None, 
                    routing_path: str = None, execution_time: float = None,
                    model_used: str = None, similar_examples_count: int = None):
        """Log completed question result with enhanced metadata"""
        row = [
            datetime.datetime.now().isoformat(),
            task_id,
            question[:100],  # Truncate long questions
            final_answer,
            total_steps,
            success,
            complexity or "unknown",
            routing_path or "unknown",
            execution_time or 0.0,
            model_used or "unknown",
            similar_examples_count or 0
        ]
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

class EvaluationLogger:
    """Logs evaluation results matching your example format"""
    
    def __init__(self, log_file: str = None):
        if log_file is None:
            self.log_file = Path(create_timestamped_filename("evaluation"))
        else:
            if not log_file.startswith("logs/"):
                self.log_file = Path(create_timestamped_filename(log_file.replace(".csv", "")))
            else:
                self.log_file = Path(log_file)
        
        self._ensure_header()
        print(f"🧪 Evaluation logging to: {self.log_file}")
    
    def _ensure_header(self):
        """Create CSV with evaluation header matching your example"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'task_id', 'question', 'final_answer', 'ground_truth', 
                    'level', 'is_correct', 'execution_time', 'strategy_used',
                    'selected_agent', 'model_used', 'similar_examples_count', 'timestamp'
                ])
    
    def log_evaluation(self, task_id: str, question: str, final_answer: str, 
                      ground_truth: str, level: int, is_correct: bool,
                      execution_time: float, strategy_used: str = None,
                      selected_agent: str = None, model_used: str = None,
                      similar_examples_count: int = None):
        """Log evaluation result matching your example format"""
        row = [
            task_id,
            question,
            final_answer,
            ground_truth,
            level,
            is_correct,
            execution_time,
            strategy_used or "unknown",
            selected_agent or "unknown", 
            model_used or "unknown",
            similar_examples_count or 0,
            datetime.datetime.now().isoformat()
        ]
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

# ============================================================================
# LOGGER FOR TASK EXECUTION TRACKING
# ============================================================================

class EnhancedAgentLogger:
    """Enhanced logger that wraps SmolAgent logger with step tracking"""
    
    def __init__(self, smolag_logger: AgentLogger, console: Console):
        self.smolag_logger = smolag_logger
        self.console = console
        self.current_task_context = {}
    
    def log_task(self, content: str, title: str = None, subtitle: str = None, level=None):
        """Log task start with context - now accepts level parameter"""
        if title:
            self.console.print(f"\n🎯 {title}", style="bold blue")
        if subtitle:
            self.console.print(f"   {subtitle}", style="dim")
        
        self.console.print(f"📝 {content}")
        
        # Store context for potential use
        self.current_task_context = {
            "title": title,
            "subtitle": subtitle,
            "content": content,
            "level": level,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # If level is provided, also log to underlying smolag logger
        if level and hasattr(self.smolag_logger, 'log'):
            try:
                self.smolag_logger.log(level, content)
            except Exception as e:
                # Fallback if smolag logger doesn't support this format
                print(f"Debug: Smolag logger call failed: {e}")
    
    def log_complexity(self, complexity: str, reason: str = None):
        """Log complexity assessment"""
        complexity_color = "green" if complexity == "simple" else "yellow"
        self.console.print(f"🧠 Complexity: [{complexity_color}]{complexity}[/{complexity_color}]")
        if reason:
            self.console.print(f"   Reason: {reason}", style="dim")
    
    def log_routing(self, route: str, reason: str = None):
        """Log routing decision"""
        route_color = "cyan" if route == "one_shot" else "magenta"
        self.console.print(f"🔀 Route: [{route_color}]{route}[/{route_color}]")
        if reason:
            self.console.print(f"   Reason: {reason}", style="dim")
    
    def log(self, level, message: str):
        """Direct logging method that matches SmolAgent interface"""
        try:
            # Try to use the underlying smolag logger
            if hasattr(self.smolag_logger, 'log'):
                self.smolag_logger.log(level, message)
            else:
                # Fallback to console output
                level_str = str(level).upper() if hasattr(level, 'name') else str(level)
                self.console.print(f"[{level_str}] {message}")
        except Exception as e:
            # Ultimate fallback
            print(f"[LOG] {message}")
    
    def info(self, message: str):
        """Info level logging"""
        self.console.print(f"ℹ️  {message}", style="blue")
        if hasattr(self.smolag_logger, 'info'):
            try:
                self.smolag_logger.info(message)
            except:
                pass
    
    def debug(self, message: str):
        """Debug level logging"""
        self.console.print(f"🐛 {message}", style="dim")
        if hasattr(self.smolag_logger, 'debug'):
            try:
                self.smolag_logger.debug(message)
            except:
                pass
    
    def warning(self, message: str):
        """Warning level logging"""
        self.console.print(f"⚠️  {message}", style="yellow")
        if hasattr(self.smolag_logger, 'warning'):
            try:
                self.smolag_logger.warning(message)
            except:
                pass
    
    def error(self, message: str):
        """Error level logging"""
        self.console.print(f"❌ {message}", style="red")
        if hasattr(self.smolag_logger, 'error'):
            try:
                self.smolag_logger.error(message)
            except:
                pass
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped logger"""
        try:
            return getattr(self.smolag_logger, name)
        except AttributeError:
            # If the smolag logger doesn't have the attribute, create a no-op method
            def no_op(*args, **kwargs):
                pass
            return no_op

# ============================================================================
# AGENT LOGGING HELPER
# ============================================================================

class AgentLoggingSetup:
    """Sets up SmolAgents logging with timestamped files"""
    
    def __init__(self, debug_mode: bool = True, 
                 step_log_file: str = None,
                 question_log_file: str = None,
                 evaluation_log_file: str = None):
        
        # Rich console for beautiful output
        self.console = Console(record=True)
        
        # Create basic SmolAgent logger (no custom callbacks to avoid parameter issues)
        try:
            smolag_logger = AgentLogger(
                level=LogLevel.DEBUG if debug_mode else LogLevel.INFO,
                console=self.console
            )
            print("✅ SmolAgent logger created successfully")
        except Exception as e:
            print(f"⚠️  SmolAgent logger creation failed: {e}")
            # Create minimal mock logger
            smolag_logger = self._create_mock_logger()
        
        # Use basic logger without custom wrapper to avoid parameter conflicts
        self.logger = smolag_logger
        
        # Manual tracking variables for comprehensive logging
        self.steps_buffer = []
        self.current_task_id = None
        self.step_counter = 0
        self.task_start_time = None
        self.manual_steps = []  # For manual step tracking
        
        # Routing tracking
        self.current_complexity = None
        self.current_routing_path = None
        self.current_model_used = None
        self.current_similar_examples_count = 0
        
        # CSV loggers with timestamped files
        try:
            self.step_logger = StepLogger(step_log_file)
            self.question_logger = QuestionLogger(question_log_file)
            self.evaluation_logger = EvaluationLogger(evaluation_log_file)
            
            # Store CSV file paths for easy access
            self.csv_loggers = {
                'question_file': self.question_logger.log_file,
                'step_file': self.step_logger.log_file,
                'evaluation_file': self.evaluation_logger.log_file
            }
            
        except Exception as e:
            print(f"⚠️  CSV logger setup failed: {e}")
            # Create minimal fallback loggers
            self.step_logger = self._create_mock_step_logger()
            self.question_logger = self._create_mock_question_logger()
            self.evaluation_logger = self._create_mock_evaluation_logger()
            self.csv_loggers = {}
        
        self.debug_mode = debug_mode
        
        print(f"🚀 AgentLoggingSetup complete:")
        if hasattr(self.step_logger, 'log_file'):
            print(f"   Steps: {self.step_logger.log_file}")
        if hasattr(self.question_logger, 'log_file'):
            print(f"   Questions: {self.question_logger.log_file}")
        if hasattr(self.evaluation_logger, 'log_file'):
            print(f"   Evaluation: {self.evaluation_logger.log_file}")
    
    def _create_mock_logger(self):
        """Create a minimal mock logger when SmolAgent logger fails"""
        class MockLogger:
            def log(self, level, message):
                print(f"[{level}] {message}")
            
            def info(self, message):
                print(f"[INFO] {message}")
            
            def debug(self, message):
                print(f"[DEBUG] {message}")
            
            def warning(self, message):
                print(f"[WARNING] {message}")
            
            def error(self, message):
                print(f"[ERROR] {message}")
        
        return MockLogger()
    
    def _create_mock_step_logger(self):
        """Create a mock step logger that doesn't write to file"""
        class MockStepLogger:
            def log_step(self, task_id, step_number, step_data):
                pass
        
        return MockStepLogger()
    
    def _create_mock_question_logger(self):
        """Create a mock question logger that doesn't write to file"""
        class MockQuestionLogger:
            def log_question(self, **kwargs):
                pass
        
        return MockQuestionLogger()
    
    def _create_mock_evaluation_logger(self):
        """Create a mock evaluation logger that doesn't write to file"""
        class MockEvaluationLogger:
            def log_evaluation(self, **kwargs):
                pass
        
        return MockEvaluationLogger()
    
    def log_step(self, action: str, details: str = ""):
        """Manual step logging method - the key to making step logging work"""
        self.step_counter += 1
        step_time = datetime.datetime.now()
        
        step_data = {
            'step_number': self.step_counter,
            'action': action,
            'details': details,
            'timestamp': step_time
        }
        
        self.manual_steps.append(step_data)
        
        # Log to CSV if available
        if hasattr(self.step_logger, 'log_step'):
            try:
                duration = (step_time - self.task_start_time).total_seconds() if self.task_start_time else 0
                
                # Use the existing log_step method format
                self.step_logger.log_step(
                    task_id=self.current_task_id or "unknown",
                    step_number=self.step_counter,
                    step_data={
                        'action': action,
                        'tool_name': '',
                        'input': details[:100],  # Truncate long details
                        'output': ''
                    }
                )
            except Exception as e:
                if self.debug_mode:
                    print(f"Debug: Step CSV logging failed: {e}")
        
        if self.debug_mode:
            print(f"📝 Step {self.step_counter}: {action}")
    
    def capture_step_log(self, step):
        """Legacy method for compatibility - now uses manual logging"""
        # This method is kept for compatibility but doesn't use callbacks
        # Instead, we rely on manual log_step() calls
        pass
    
    def start_task(self, task_id: str, complexity: str = None, model_used: str = None):
        """Start tracking a new task with timing"""
        try:
            self.current_task_id = task_id
            self.step_counter = 0
            self.steps_buffer.clear()
            self.manual_steps.clear()
            self.current_complexity = complexity
            self.current_routing_path = None
            self.current_model_used = model_used
            self.current_similar_examples_count = 0
            self.task_start_time = datetime.datetime.now()
            
            # Log the task start
            self.log_step("task_start", f"Started task: {task_id}")
            
            if self.debug_mode:
                print(f"🚀 Starting task: {task_id}")
                if complexity:
                    print(f"   Complexity: {complexity}")
                if model_used:
                    print(f"   Model: {model_used}")
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Task start failed: {e}")
    
    def set_routing_path(self, path: str):
        """Track which routing path was taken"""
        try:
            self.current_routing_path = path
            self.log_step("routing_decision", f"Route: {path}")
            if self.debug_mode:
                print(f"🔀 Routing path: {path}")
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Routing path setting failed: {e}")
    
    def set_complexity(self, complexity: str):
        """Track complexity detection"""
        try:
            self.current_complexity = complexity
            self.log_step("complexity_assessment", f"Complexity: {complexity}")
            if self.debug_mode:
                print(f"🧠 Complexity: {complexity}")
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Complexity setting failed: {e}")
    
    def set_similar_examples_count(self, count: int):
        """Track number of RAG examples used"""
        try:
            self.current_similar_examples_count = count
            self.log_step("rag_retrieval", f"Retrieved {count} examples")
            if self.debug_mode:
                print(f"📚 Similar examples: {count}")
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Examples count setting failed: {e}")
    
    def log_question_result(self, task_id: str, question: str, 
                           final_answer: str, total_steps: int, success: bool):
        """Log completed question with enhanced metadata and timing"""
        try:
            execution_time = 0.0
            if self.task_start_time:
                execution_time = (datetime.datetime.now() - self.task_start_time).total_seconds()
            
            self.log_step("question_complete", f"Final answer: {final_answer}")
            
            if hasattr(self.question_logger, 'log_question'):
                self.question_logger.log_question(
                    task_id=task_id,
                    question=question,
                    final_answer=final_answer,
                    total_steps=total_steps,
                    success=success,
                    complexity=self.current_complexity,
                    routing_path=self.current_routing_path,
                    execution_time=execution_time,
                    model_used=self.current_model_used,
                    similar_examples_count=self.current_similar_examples_count
                )
            
            if self.debug_mode:
                status = "✅" if success else "❌"
                print(f"{status} Question completed: {total_steps} steps in {execution_time:.1f}s")
        
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Question result logging failed: {e}")
    
    def log_evaluation_result(self, task_id: str, question: str, final_answer: str,
                             ground_truth: str, level: int, is_correct: bool,
                             strategy_used: str = None, selected_agent: str = None):
        """Log evaluation result matching the example format"""
        try:
            execution_time = 0.0
            if self.task_start_time:
                execution_time = (datetime.datetime.now() - self.task_start_time).total_seconds()
            
            self.log_step("evaluation_complete", f"Correct: {is_correct}")
            
            if hasattr(self.evaluation_logger, 'log_evaluation'):
                self.evaluation_logger.log_evaluation(
                    task_id=task_id,
                    question=question,
                    final_answer=final_answer,
                    ground_truth=ground_truth,
                    level=level,
                    is_correct=is_correct,
                    execution_time=execution_time,
                    strategy_used=strategy_used or self.current_routing_path,
                    selected_agent=selected_agent,
                    model_used=self.current_model_used,
                    similar_examples_count=self.current_similar_examples_count
                )
            
            if self.debug_mode:
                status = "✅" if is_correct else "❌"
                print(f"{status} Evaluation logged: Level {level}, {execution_time:.1f}s")
        
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Evaluation result logging failed: {e}")
    
    @property
    def current_log_files(self) -> Dict[str, str]:
        """Get current log file paths"""
        try:
            return {
                "steps": str(getattr(self.step_logger, 'log_file', 'mock')),
                "questions": str(getattr(self.question_logger, 'log_file', 'mock')), 
                "evaluation": str(getattr(self.evaluation_logger, 'log_file', 'mock'))
            }
        except Exception as e:
            return {
                "steps": "error",
                "questions": "error",
                "evaluation": "error",
                "error": str(e)
            }