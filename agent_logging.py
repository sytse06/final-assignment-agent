# agent_logging.py
# CSV logging functionality for GAIA agent

import csv
import datetime
from pathlib import Path
from typing import Dict
from rich.console import Console
from smolagents import AgentLogger, LogLevel

# ============================================================================
# CSV LOGGERS
# ============================================================================

class StepLogger:
    """Logs individual agent steps to CSV"""
    
    def __init__(self, log_file: str = "gaia_steps.csv"):
        self.log_file = Path(log_file)
        self._ensure_csv_header()
    
    def _ensure_csv_header(self):
        """Create CSV with header if it doesn't exist"""
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
    """Logs completed questions to CSV"""
    
    def __init__(self, log_file: str = "gaia_questions.csv"):
        self.log_file = Path(log_file)
        self._ensure_header()
    
    def _ensure_header(self):
        """Create CSV with header if it doesn't exist"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'task_id', 'question', 'final_answer', 
                    'total_steps', 'success'
                ])
    
    def log_question(self, task_id: str, question: str, final_answer: str, 
                    total_steps: int, success: bool):
        """Log completed question result"""
        row = [
            datetime.datetime.now().isoformat(),
            task_id,
            question[:100],  # Truncate long questions
            final_answer,
            total_steps,
            success
        ]
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

# ============================================================================
# AGENT LOGGING HELPER
# ============================================================================

class AgentLoggingSetup:
    """Sets up SmolagAgent logging with step capture"""
    
    def __init__(self, debug_mode: bool = True, 
                 step_log_file: str = "gaia_steps.csv",
                 question_log_file: str = "gaia_questions.csv"):
        
        # Rich console for beautiful output
        self.console = Console(record=True)
        
        # SmolagAgent logger
        self.logger = AgentLogger(
            level=LogLevel.DEBUG if debug_mode else LogLevel.INFO,
            console=self.console
        )
        
        # Step tracking
        self.steps_buffer = []
        self.current_task_id = None
        self.step_counter = 0
        
        # CSV loggers
        self.step_logger = StepLogger(step_log_file)
        self.question_logger = QuestionLogger(question_log_file)
        
        self.debug_mode = debug_mode
    
    def capture_step_log(self, step):
        """Capture step information for logging"""
        self.step_counter += 1
        
        # Extract step information
        step_data = {
            'action': getattr(step, 'action', 'unknown'),
            'tool_name': getattr(step, 'tool_name', ''),
            'input': getattr(step, 'input', ''),
            'output': getattr(step, 'observation', '')
        }
        
        # Log to CSV
        if self.current_task_id:
            self.step_logger.log_step(
                task_id=self.current_task_id,
                step_number=self.step_counter,
                step_data=step_data
            )
        
        # Capture console output
        step_text = self.console.export_text(clear=True)
        self.steps_buffer.append(step_text)
        
        if self.debug_mode:
            print(f"Step {self.step_counter}: {step_data['action']}")
    
    def start_task(self, task_id: str):
        """Start tracking a new task"""
        self.current_task_id = task_id
        self.step_counter = 0
        self.steps_buffer.clear()
    
    def log_question_result(self, task_id: str, question: str, 
                           final_answer: str, total_steps: int, success: bool):
        """Log completed question"""
        self.question_logger.log_question(
            task_id=task_id,
            question=question,
            final_answer=final_answer,
            total_steps=total_steps,
            success=success
        )