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
# ENHANCED LOGGER FOR TASK TRACKING
# ============================================================================

class EnhancedAgentLogger:
    """Enhanced logger that wraps SmolagAgent logger with additional functionality"""
    
    def __init__(self, smolag_logger: AgentLogger, console: Console):
        self.smolag_logger = smolag_logger
        self.console = console
        self.current_task_context = {}
    
    def log_task(self, content: str, title: str = None, subtitle: str = None):
        """Log task start with context"""
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
            "timestamp": datetime.datetime.now().isoformat()
        }
    
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
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped logger"""
        return getattr(self.smolag_logger, name)

# ============================================================================
# AGENT LOGGING HELPER
# ============================================================================

class AgentLoggingSetup:
    """Sets up SmolagAgent logging with timestamped files and enhanced functionality"""
    
    def __init__(self, debug_mode: bool = True, 
                 step_log_file: str = None,
                 question_log_file: str = None,
                 evaluation_log_file: str = None):
        
        # Rich console for beautiful output
        self.console = Console(record=True)
        
        # SmolagAgent logger
        smolag_logger = AgentLogger(
            level=LogLevel.DEBUG if debug_mode else LogLevel.INFO,
            console=self.console
        )
        
        # Enhanced logger wrapper
        self.logger = EnhancedAgentLogger(smolag_logger, self.console)
        
        # Step tracking
        self.steps_buffer = []
        self.current_task_id = None
        self.step_counter = 0
        self.task_start_time = None
        
        # Routing tracking
        self.current_complexity = None
        self.current_routing_path = None
        self.current_model_used = None
        self.current_similar_examples_count = 0
        
        # CSV loggers with timestamped files
        self.step_logger = StepLogger(step_log_file)
        self.question_logger = QuestionLogger(question_log_file)
        self.evaluation_logger = EvaluationLogger(evaluation_log_file)
        
        self.debug_mode = debug_mode
        
        print(f"🚀 Logging setup complete:")
        print(f"   Steps: {self.step_logger.log_file}")
        print(f"   Questions: {self.question_logger.log_file}")
        print(f"   Evaluation: {self.evaluation_logger.log_file}")
    
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
    
    def start_task(self, task_id: str, complexity: str = None, model_used: str = None):
        """Start tracking a new task with timing"""
        self.current_task_id = task_id
        self.step_counter = 0
        self.steps_buffer.clear()
        self.current_complexity = complexity
        self.current_routing_path = None
        self.current_model_used = model_used
        self.current_similar_examples_count = 0
        self.task_start_time = datetime.datetime.now()
        
        if self.debug_mode:
            print(f"🚀 Starting task: {task_id}")
            if complexity:
                print(f"   Complexity: {complexity}")
            if model_used:
                print(f"   Model: {model_used}")
    
    def set_routing_path(self, path: str):
        """Track which routing path was taken"""
        self.current_routing_path = path
        if self.debug_mode:
            print(f"🔀 Routing path: {path}")
    
    def set_complexity(self, complexity: str):
        """Track complexity detection"""
        self.current_complexity = complexity
        if self.debug_mode:
            print(f"🧠 Complexity: {complexity}")
    
    def set_similar_examples_count(self, count: int):
        """Track number of RAG examples used"""
        self.current_similar_examples_count = count
        if self.debug_mode:
            print(f"📚 Similar examples: {count}")
    
    def log_question_result(self, task_id: str, question: str, 
                           final_answer: str, total_steps: int, success: bool):
        """Log completed question with enhanced metadata and timing"""
        execution_time = 0.0
        if self.task_start_time:
            execution_time = (datetime.datetime.now() - self.task_start_time).total_seconds()
        
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
    
    def log_evaluation_result(self, task_id: str, question: str, final_answer: str,
                             ground_truth: str, level: int, is_correct: bool,
                             strategy_used: str = None, selected_agent: str = None):
        """Log evaluation result matching the example format"""
        execution_time = 0.0
        if self.task_start_time:
            execution_time = (datetime.datetime.now() - self.task_start_time).total_seconds()
        
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
    
    @property
    def current_log_files(self) -> Dict[str, str]:
        """Get current log file paths"""
        return {
            "steps": str(self.step_logger.log_file),
            "questions": str(self.question_logger.log_file), 
            "evaluation": str(self.evaluation_logger.log_file)
        }

# ============================================================================
# PERFORMANCE ANALYTICS
# ============================================================================

class PerformanceAnalyzer:
    """Analyze logged performance data from timestamped files"""
    
    def __init__(self, log_pattern: str = "gaia_questions"):
        self.log_pattern = log_pattern
        self.latest_log_file = get_latest_log_file(log_pattern)
    
    def analyze_routing_performance(self, log_file: str = None) -> Dict:
        """Analyze performance by routing path"""
        if log_file is None:
            log_file = self.latest_log_file
        
        if not log_file or not Path(log_file).exists():
            return {"error": "No log file found"}
        
        stats = {
            "one_shot": {"count": 0, "success": 0, "avg_steps": 0, "avg_time": 0},
            "manager": {"count": 0, "success": 0, "avg_steps": 0, "avg_time": 0},
            "unknown": {"count": 0, "success": 0, "avg_steps": 0, "avg_time": 0}
        }
        
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                routing_path = row.get('routing_path', 'unknown')
                if routing_path not in stats:
                    routing_path = 'unknown'
                
                stats[routing_path]["count"] += 1
                
                if row.get('success', '').lower() == 'true':
                    stats[routing_path]["success"] += 1
                
                try:
                    steps = int(row.get('total_steps', 0))
                    exec_time = float(row.get('execution_time', 0))
                    
                    # Running average for steps
                    current_avg = stats[routing_path]["avg_steps"]
                    count = stats[routing_path]["count"]
                    stats[routing_path]["avg_steps"] = ((current_avg * (count - 1)) + steps) / count
                    
                    # Running average for time
                    current_avg_time = stats[routing_path]["avg_time"]
                    stats[routing_path]["avg_time"] = ((current_avg_time * (count - 1)) + exec_time) / count
                    
                except (ValueError, ZeroDivisionError):
                    pass
        
        # Calculate success rates
        for path_stats in stats.values():
            if path_stats["count"] > 0:
                path_stats["success_rate"] = path_stats["success"] / path_stats["count"]
            else:
                path_stats["success_rate"] = 0
        
        return stats
    
    def get_recent_performance(self, hours: int = 24, log_file: str = None) -> Dict:
        """Get performance stats for recent time period"""
        if log_file is None:
            log_file = self.latest_log_file
            
        if not log_file or not Path(log_file).exists():
            return {"error": "No log file found"}
        
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        recent_count = 0
        recent_success = 0
        total_time = 0
        
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    timestamp = datetime.datetime.fromisoformat(row.get('timestamp', ''))
                    if timestamp >= cutoff_time:
                        recent_count += 1
                        if row.get('success', '').lower() == 'true':
                            recent_success += 1
                        total_time += float(row.get('execution_time', 0))
                except (ValueError, TypeError):
                    continue
        
        return {
            "hours": hours,
            "total_questions": recent_count,
            "successful_questions": recent_success,
            "success_rate": recent_success / recent_count if recent_count > 0 else 0,
            "avg_execution_time": total_time / recent_count if recent_count > 0 else 0,
            "log_file": log_file
        }
    
    def list_available_logs(self) -> Dict:
        """List all available timestamped log files"""
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return {"error": "No logs directory found"}
        
        log_types = {
            "steps": list(logs_dir.glob("gaia_steps_*.csv")),
            "questions": list(logs_dir.glob("gaia_questions_*.csv")),
            "evaluation": list(logs_dir.glob("evaluation_*.csv"))
        }
        
        # Sort by modification time (newest first)
        for log_type in log_types:
            log_types[log_type] = sorted(
                log_types[log_type], 
                key=lambda f: f.stat().st_mtime, 
                reverse=True
            )
        
        return {
            log_type: [str(f) for f in files] 
            for log_type, files in log_types.items()
        }
        
# ============================================================================
# EXPORT INTERFACE
# ============================================================================

def analyze_performance(log_pattern: str = "gaia_questions"):
    """Quick performance analysis function with timestamped log support"""
    analyzer = PerformanceAnalyzer(log_pattern)
    
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 30)
    
    if not analyzer.latest_log_file:
        print("❌ No log files found")
        print("💡 Make sure you have run some questions first")
        return {"error": "No log files found"}
    
    print(f"📁 Using log file: {analyzer.latest_log_file}")
    
    # Routing performance
    routing_stats = analyzer.analyze_routing_performance()
    if "error" not in routing_stats:
        print("\n🔀 Routing Performance:")
        for path, stats in routing_stats.items():
            if stats["count"] > 0:
                print(f"  {path.upper()}:")
                print(f"    Questions: {stats['count']}")
                print(f"    Success rate: {stats['success_rate']:.1%}")
                print(f"    Avg steps: {stats['avg_steps']:.1f}")
                print(f"    Avg time: {stats['avg_time']:.1f}s")
    
    # Recent performance
    recent_stats = analyzer.get_recent_performance()
    if "error" not in recent_stats:
        print(f"\n⏱️  Recent Performance (24h):")
        print(f"  Questions: {recent_stats['total_questions']}")
        print(f"  Success rate: {recent_stats['success_rate']:.1%}")
        print(f"  Avg execution time: {recent_stats['avg_execution_time']:.1f}s")
    
    return {
        "routing_stats": routing_stats,
        "recent_stats": recent_stats,
        "log_file": analyzer.latest_log_file
    }

def list_all_logs():
    """List all available timestamped log files"""
    analyzer = PerformanceAnalyzer()
    logs = analyzer.list_available_logs()
    
    if "error" in logs:
        print("❌ No logs directory found")
        return logs
    
    print("📁 AVAILABLE LOG FILES")
    print("=" * 30)
    
    for log_type, files in logs.items():
        if files:
            print(f"\n📊 {log_type.upper()} LOGS:")
            for i, file in enumerate(files[:5], 1):  # Show latest 5
                file_path = Path(file)
                timestamp = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                print(f"  {i}. {file_path.name}")
                print(f"     Modified: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if len(files) > 5:
                print(f"     ... and {len(files) - 5} more files")
    
    return logs

def get_log_summary(log_file: str = None):
    """Get summary of specific log file"""
    if log_file is None:
        analyzer = PerformanceAnalyzer()
        log_file = analyzer.latest_log_file
    
    if not log_file or not Path(log_file).exists():
        print("❌ Log file not found")
        return {"error": "Log file not found"}
    
    row_count = 0
    earliest_time = None
    latest_time = None
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                row_count += 1
                
                try:
                    timestamp = datetime.datetime.fromisoformat(row.get('timestamp', ''))
                    if earliest_time is None or timestamp < earliest_time:
                        earliest_time = timestamp
                    if latest_time is None or timestamp > latest_time:
                        latest_time = timestamp
                except (ValueError, TypeError):
                    continue
        
        duration = None
        if earliest_time and latest_time:
            duration = (latest_time - earliest_time).total_seconds()
        
        print(f"📋 LOG SUMMARY: {Path(log_file).name}")
        print("=" * 40)
        print(f"Total entries: {row_count}")
        if earliest_time:
            print(f"First entry: {earliest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if latest_time:
            print(f"Last entry: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if duration:
            print(f"Duration: {duration/3600:.1f} hours")
        
        return {
            "file": log_file,
            "total_entries": row_count,
            "earliest_time": earliest_time.isoformat() if earliest_time else None,
            "latest_time": latest_time.isoformat() if latest_time else None,
            "duration_seconds": duration
        }
        
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return {"error": str(e)}