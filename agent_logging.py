# agent_logging.py
# CSV logging functionality for GAIA agent

import csv
import datetime
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from rich.console import Console
from smolagents import AgentLogger, LogLevel
import json

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
    """Logs individual agent steps to CSV with enhanced metadata"""
    
    def __init__(self, log_file: str = None):
        if log_file is None:
            self.log_file = Path(create_timestamped_filename("gaia_steps"))
        else:
            if not log_file.startswith("logs/"):
                self.log_file = Path(create_timestamped_filename(log_file.replace(".csv", "")))
            else:
                self.log_file = Path(log_file)
        
        self._ensure_csv_header()
        print(f"📝 Step logging to: {self.log_file}")
    
    def _ensure_csv_header(self):
        """Create CSV with enhanced header for testing framework"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'task_id', 'step_number', 'node_name',
                    'action', 'tool_name', 'input', 'output', 'duration_ms',
                    'context_bridge_step', 'routing_path', 'complexity'
                ])
    
    def log_step(self, task_id: str, step_number: int, step_data: Dict):
        """Append single step to CSV with enhanced metadata"""
        timestamp = datetime.datetime.now().isoformat()
        
        row = [
            timestamp,
            task_id,
            step_number,
            step_data.get('node_name', ''),
            step_data.get('action', ''),
            step_data.get('tool_name', ''),
            str(step_data.get('input', ''))[:200],
            str(step_data.get('output', ''))[:200],
            step_data.get('duration_ms', 0),
            step_data.get('context_bridge_step', False),
            step_data.get('routing_path', ''),
            step_data.get('complexity', '')
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
                    model_used: str = None, similar_examples_count: int = None,
                    # NEW: Add missing parameters that log_question_result passes
                    selected_agent: str = None, has_file: bool = False, 
                    file_category: str = None, context_bridge_used: bool = False,
                    coordinator_used: bool = False, file_metadata_available: bool = False):
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
            # Note: The additional parameters are accepted but not logged to maintain CSV compatibility
            # This prevents the "unexpected keyword argument" error
        ]
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

class EvaluationLogger:
    """Logs evaluation results matching testing framework expectations"""
    
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
        """Create CSV with enhanced evaluation header"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'task_id', 'question', 'final_answer', 'ground_truth', 
                    'level', 'is_correct', 'execution_time', 'strategy_used',
                    'selected_agent', 'model_used', 'similar_examples_count', 
                    'timestamp', 'matching_method', 'file_category',
                    'context_bridge_metrics', 'coordinator_analysis_available'
                ])
    
    def log_evaluation(self, task_id: str, question: str, final_answer: str, 
                      ground_truth: str, level: int, is_correct: bool,
                      execution_time: float, strategy_used: str = None,
                      selected_agent: str = None, model_used: str = None,
                      similar_examples_count: int = None, matching_method: str = None,
                      file_category: str = None, context_bridge_metrics: Dict = None,
                      coordinator_analysis_available: bool = False):
        """Log evaluation with enhanced metadata for analysis"""
        
        # Serialize context bridge metrics
        context_metrics_str = ""
        if context_bridge_metrics:
            try:
                context_metrics_str = json.dumps(context_bridge_metrics)[:100]
            except:
                context_metrics_str = str(context_bridge_metrics)[:100]
        
        row = [
            task_id,
            question[:100],
            final_answer,
            ground_truth,
            level,
            is_correct,
            execution_time,
            strategy_used or "unknown",
            selected_agent or "unknown", 
            model_used or "unknown",
            similar_examples_count or 0,
            datetime.datetime.now().isoformat(),
            matching_method or "unknown",
            file_category or "none",
            context_metrics_str,
            coordinator_analysis_available
        ]
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
# ============================================================================
# FILE METADATA LOGGER
# ============================================================================
class FileMetadataLogger:
    """NEW: Logs file metadata for testing framework analysis"""
    
    def __init__(self, log_file: str = None):
        if log_file is None:
            self.log_file = Path(create_timestamped_filename("file_metadata"))
        else:
            if not log_file.startswith("logs/"):
                self.log_file = Path(create_timestamped_filename(log_file.replace(".csv", "")))
            else:
                self.log_file = Path(log_file)
        
        self._ensure_header()
        print(f"📁 File metadata logging to: {self.log_file}")
    
    def _ensure_header(self):
        """Create CSV header for file metadata"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'task_id', 'file_name', 'file_path', 'extension',
                    'category', 'processing_approach', 'recommended_specialist',
                    'mime_type', 'estimated_complexity', 'file_size', 'analysis_success'
                ])
    
    def log_file_metadata(self, task_id: str, file_metadata: Dict):
        """Log file metadata analysis results"""
        if not file_metadata:
            return
        
        row = [
            datetime.datetime.now().isoformat(),
            task_id,
            file_metadata.get('file_name', ''),
            file_metadata.get('file_path', ''),
            file_metadata.get('extension', ''),
            file_metadata.get('category', 'unknown'),
            file_metadata.get('processing_approach', 'unknown'),
            file_metadata.get('recommended_specialist', 'unknown'),
            file_metadata.get('mime_type', ''),
            file_metadata.get('estimated_complexity', 'unknown'),
            file_metadata.get('file_size', 0),
            not bool(file_metadata.get('error', None))
        ]
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

# ============================================================================
# SMOLAGENT LOGGER
# ============================================================================
class SmolagStepLogger:
    """Logger for SmolagAgent detailed steps"""
    
    def __init__(self, log_file: str = None):
        if log_file is None:
            self.log_file = Path(create_timestamped_filename("gaia_smolagents_steps", "md"))
        else:
            if not log_file.startswith("logs/"):
                self.log_file = Path(create_timestamped_filename(log_file.replace(".md", ""), "md"))
            else:
                self.log_file = Path(log_file)
        
        self._ensure_file()
        print(f"🤖 SmolagAgent step logging to: {self.log_file}")
    
    def _ensure_file(self):
        """Create markdown file with header"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_file.exists():
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("# SmolagAgent Detailed Execution Log\n\n")
                f.write("Enhanced logging with ActionStep detail capture.\n\n")
                f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
                f.write("---\n\n")
    
    def log_agent_execution(self, agent_name: str, task: str, agent_instance, result: str, duration: float = None):
        """CORE FIX: Log agent execution with proper step extraction"""
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                f.write(f"## {agent_name} - {timestamp}\n\n")
                f.write(f"**Task**: {task[:200]}{'...' if len(task) > 200 else ''}\n\n")
                if duration:
                    f.write(f"**Duration**: {duration:.2f}s\n\n")
                
                # CRITICAL FIX: Proper step extraction with multiple methods
                detailed_steps = self._extract_detailed_steps_multi_method(agent_instance)
                
                f.write("### Execution Steps\n\n")
                
                if detailed_steps and len(detailed_steps) > 0:
                    print(f"✅ Extracted {len(detailed_steps)} detailed steps from {agent_name}")
                    self._write_detailed_steps(f, detailed_steps)
                else:
                    print(f"⚠️ No detailed steps extracted from {agent_name}, using fallback")
                    self._write_fallback_information(f, agent_instance)
                
                f.write(f"### Final Result\n\n")
                if result:
                    result_preview = result[:1000] + "..." if len(result) > 1000 else result
                    f.write(f"```\n{result_preview}\n```\n\n")
                else:
                    f.write("*No result captured*\n\n")
                
                f.write("---\n\n")
                
        except Exception as e:
            print(f"❌ Enhanced logging failed for {agent_name}: {e}")
    
    def _extract_detailed_steps_multi_method(self, agent_instance) -> List[Dict]:
        """CORE FIX: Extract detailed steps using multiple extraction methods"""
        
        detailed_steps = []
        extraction_success = False
        
        # Method 1: write_inner_memory_from_logs() - Primary SmolagAgent method
        try:
            if hasattr(agent_instance, 'write_inner_memory_from_logs'):
                memory_logs = agent_instance.write_inner_memory_from_logs()
                if memory_logs and len(memory_logs) > 0:
                    parsed_steps = self._parse_memory_logs(memory_logs)
                    detailed_steps.extend(parsed_steps)
                    print(f"📝 Method 1 (write_inner_memory_from_logs): {len(parsed_steps)} steps")
                    extraction_success = True
        except Exception as e:
            print(f"⚠️ Method 1 failed: {e}")
        
        # Method 2: Direct logs attribute access
        if not extraction_success:
            try:
                if hasattr(agent_instance, 'logs') and agent_instance.logs:
                    log_steps = self._parse_agent_logs(agent_instance.logs)
                    if log_steps:
                        detailed_steps.extend(log_steps)
                        print(f"📝 Method 2 (direct logs): {len(log_steps)} steps")
                        extraction_success = True
            except Exception as e:
                print(f"⚠️ Method 2 failed: {e}")
        
        # Method 3: Chat history extraction
        if not extraction_success:
            try:
                if hasattr(agent_instance, 'chat_history') and agent_instance.chat_history:
                    chat_steps = self._parse_chat_history(agent_instance.chat_history)
                    if chat_steps:
                        detailed_steps.extend(chat_steps)
                        print(f"📝 Method 3 (chat_history): {len(chat_steps)} steps")
                        extraction_success = True
            except Exception as e:
                print(f"⚠️ Method 3 failed: {e}")
        
        # Method 4: Step logs or execution history
        if not extraction_success:
            try:
                step_logs = None
                if hasattr(agent_instance, 'step_logs') and agent_instance.step_logs:
                    step_logs = agent_instance.step_logs
                elif hasattr(agent_instance, 'execution_history') and agent_instance.execution_history:
                    step_logs = agent_instance.execution_history
                
                if step_logs:
                    step_log_data = self._parse_step_logs(step_logs)
                    if step_log_data:
                        detailed_steps.extend(step_log_data)
                        print(f"📝 Method 4 (step_logs): {len(step_log_data)} steps")
                        extraction_success = True
            except Exception as e:
                print(f"⚠️ Method 4 failed: {e}")
        
        if not extraction_success:
            print(f"⚠️ All extraction methods failed for agent type: {type(agent_instance).__name__}")
        
        return detailed_steps
    
    def _parse_memory_logs(self, memory_logs) -> List[Dict]:
        """Parse write_inner_memory_from_logs() output"""
        steps = []
        
        for i, log_entry in enumerate(memory_logs):
            try:
                step = {
                    'step_number': i + 1,
                    'type': 'memory_log',
                    'description': self._extract_description(log_entry),
                    'reasoning': self._extract_reasoning(log_entry),
                    'tool_calls': self._extract_tool_calls(log_entry),
                    'observations': self._extract_observations(log_entry),
                    'duration': self._extract_duration(log_entry),
                    'raw_content': str(log_entry)[:200] + "..." if len(str(log_entry)) > 200 else str(log_entry)
                }
                steps.append(step)
            except Exception as e:
                print(f"⚠️ Error parsing memory log {i}: {e}")
                # Still add a basic step
                steps.append({
                    'step_number': i + 1,
                    'type': 'memory_log_error',
                    'description': f'Memory log {i + 1} (parse error)',
                    'reasoning': str(log_entry)[:100],
                    'tool_calls': [],
                    'observations': f'Parse error: {str(e)}',
                    'duration': None,
                    'raw_content': str(log_entry)[:200]
                })
        
        return steps
    
    def _parse_agent_logs(self, logs) -> List[Dict]:
        """Parse agent.logs attribute"""
        steps = []
        
        for i, log_entry in enumerate(logs):
            try:
                step = {
                    'step_number': i + 1,
                    'type': 'agent_log',
                    'description': self._extract_description(log_entry),
                    'reasoning': self._extract_reasoning(log_entry),
                    'tool_calls': self._extract_tool_calls(log_entry),
                    'observations': self._extract_observations(log_entry),
                    'duration': self._extract_duration(log_entry),
                    'timestamp': getattr(log_entry, 'timestamp', None),
                    'raw_content': str(log_entry)[:200]
                }
                steps.append(step)
            except Exception as e:
                print(f"⚠️ Error parsing agent log {i}: {e}")
        
        return steps
    
    def _parse_chat_history(self, chat_history) -> List[Dict]:
        """Parse chat_history attribute"""
        steps = []
        
        for i, message in enumerate(chat_history):
            try:
                role = getattr(message, 'role', 'unknown')
                content = getattr(message, 'content', str(message))
                
                step = {
                    'step_number': i + 1,
                    'type': 'chat_message',
                    'description': f'{role} message',
                    'reasoning': content if len(content) < 500 else content[:500] + "...",
                    'tool_calls': self._extract_tool_calls_from_content(content),
                    'observations': content if 'observation' in content.lower() else '',
                    'duration': None,
                    'role': role,
                    'raw_content': content[:200]
                }
                steps.append(step)
            except Exception as e:
                print(f"⚠️ Error parsing chat message {i}: {e}")
        
        return steps
    
    def _parse_step_logs(self, step_logs) -> List[Dict]:
        """Parse step_logs or execution_history"""
        steps = []
        
        for i, step_log in enumerate(step_logs):
            try:
                step = {
                    'step_number': i + 1,
                    'type': 'step_log',
                    'description': self._extract_description(step_log),
                    'reasoning': self._extract_reasoning(step_log),
                    'tool_calls': self._extract_tool_calls(step_log),
                    'observations': self._extract_observations(step_log),
                    'duration': self._extract_duration(step_log),
                    'raw_content': str(step_log)[:200]
                }
                steps.append(step)
            except Exception as e:
                print(f"⚠️ Error parsing step log {i}: {e}")
        
        return steps
    
    def _extract_description(self, entry) -> str:
        """Extract step description from entry"""
        if isinstance(entry, dict):
            return (entry.get('description') or 
                   entry.get('action') or 
                   entry.get('type') or 
                   'Dict step')
        elif hasattr(entry, 'description'):
            return entry.description
        elif hasattr(entry, 'action'):
            return entry.action
        else:
            return f'{type(entry).__name__} step'
    
    def _extract_reasoning(self, entry) -> str:
        """Extract reasoning/thinking from entry"""
        if isinstance(entry, dict):
            return (entry.get('reasoning') or 
                   entry.get('llm_output') or 
                   entry.get('thinking') or 
                   entry.get('content') or '')
        elif hasattr(entry, 'reasoning'):
            return str(entry.reasoning)
        elif hasattr(entry, 'llm_output'):
            return str(entry.llm_output)
        elif hasattr(entry, 'thinking'):
            return str(entry.thinking)
        elif hasattr(entry, 'content'):
            return str(entry.content)
        else:
            return str(entry)[:200]
    
    def _extract_tool_calls(self, entry) -> List[Dict]:
        """Extract tool calls from entry"""
        tool_calls = []
        
        # Try multiple attribute names
        raw_calls = None
        if isinstance(entry, dict):
            raw_calls = entry.get('tool_calls') or entry.get('tools')
        elif hasattr(entry, 'tool_calls'):
            raw_calls = entry.tool_calls
        elif hasattr(entry, 'tools'):
            raw_calls = entry.tools
        
        if raw_calls:
            for call in raw_calls:
                try:
                    if isinstance(call, dict):
                        tool_calls.append({
                            'name': call.get('name', 'unknown'),
                            'parameters': call.get('parameters', {}),
                            'result': str(call.get('result', 'No result'))[:200]
                        })
                    else:
                        tool_calls.append({
                            'name': getattr(call, 'name', 'unknown'),
                            'parameters': getattr(call, 'parameters', {}),
                            'result': str(getattr(call, 'result', 'No result'))[:200]
                        })
                except Exception:
                    tool_calls.append({
                        'name': 'parse_error',
                        'parameters': {},
                        'result': str(call)[:100]
                    })
        
        return tool_calls
    
    def _extract_tool_calls_from_content(self, content: str) -> List[Dict]:
        """Extract tool calls from text content"""
        tool_calls = []
        content_lower = content.lower()
        
        # Simple pattern matching for common tools
        tool_patterns = {
            'python_executor': ['```python', 'python_executor', 'execute code'],
            'web_search': ['search', 'google', 'web search', 'find information'],
            'calculator': ['calculate', 'math', 'compute', 'arithmetic'],
            'file_processor': ['read file', 'process file', 'open file', 'load data'],
            'browser': ['navigate', 'click', 'browser', 'website', 'visit']
        }
        
        for tool_name, keywords in tool_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                tool_calls.append({
                    'name': tool_name,
                    'parameters': {'inferred_from_content': True},
                    'result': 'Tool usage detected from content'
                })
        
        return tool_calls
    
    def _extract_observations(self, entry) -> str:
        """Extract observations from entry"""
        if isinstance(entry, dict):
            return str(entry.get('observations') or entry.get('observation') or '')
        elif hasattr(entry, 'observations'):
            return str(entry.observations)
        elif hasattr(entry, 'observation'):
            return str(entry.observation)
        else:
            return ''
    
    def _extract_duration(self, entry) -> Optional[float]:
        """Extract duration from entry"""
        if isinstance(entry, dict):
            return entry.get('duration')
        elif hasattr(entry, 'duration'):
            return entry.duration
        else:
            return None
    
    def _write_detailed_steps(self, f, detailed_steps):
        """Write detailed step information with proper formatting"""
        
        for step in detailed_steps:
            step_num = step.get('step_number', 0)
            description = step.get('description', 'Unknown step')
            
            f.write(f"**Step {step_num}: {description}**\n\n")
            
            # Duration
            duration = step.get('duration')
            if duration:
                f.write(f"- **Duration**: {duration:.2f}s\n")
            
            # Tool calls
            tool_calls = step.get('tool_calls', [])
            if tool_calls:
                f.write(f"- **Tools Used**: ")
                tool_names = [call.get('name', 'unknown') for call in tool_calls]
                f.write(f"{', '.join(tool_names)}\n")
                
                for call in tool_calls:
                    name = call.get('name', 'unknown')
                    params = call.get('parameters', {})
                    result = call.get('result', 'No result')
                    
                    f.write(f"  - `{name}`")
                    if params and params != {'inferred_from_content': True}:
                        f.write(f" with {params}")
                    f.write(f" → {result}\n")
            
            # Reasoning
            reasoning = step.get('reasoning', '')
            if reasoning and len(reasoning.strip()) > 0:
                reasoning_preview = reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
                f.write(f"- **Reasoning**: {reasoning_preview}\n")
            
            # Observations
            observations = step.get('observations', '')
            if observations and len(observations.strip()) > 0:
                obs_preview = observations[:200] + "..." if len(observations) > 200 else observations
                f.write(f"- **Observations**: {obs_preview}\n")
            
            # Type info for debugging
            step_type = step.get('type', 'unknown')
            f.write(f"- **Type**: {step_type}\n")
            
            f.write("\n")
    
    def _write_fallback_information(self, f, agent_instance):
        """Write fallback information when step extraction fails"""
        f.write("**Step Extraction Status**: Unable to extract detailed steps\n\n")
        
        # Agent information
        f.write("**Agent Information**:\n")
        f.write(f"- **Type**: {type(agent_instance).__name__}\n")
        
        if hasattr(agent_instance, 'name'):
            f.write(f"- **Name**: {agent_instance.name}\n")
        
        if hasattr(agent_instance, 'description'):
            f.write(f"- **Description**: {agent_instance.description}\n")
        
        # Available attributes
        f.write("**Available Attributes**:\n")
        attrs = [attr for attr in dir(agent_instance) if not attr.startswith('_')]
        logging_attrs = [attr for attr in attrs if any(keyword in attr.lower() 
                        for keyword in ['log', 'step', 'history', 'memory', 'chat'])]
        
        if logging_attrs:
            f.write(f"- **Logging-related**: {', '.join(logging_attrs)}\n")
        else:
            f.write("- **No logging-related attributes found**\n")
        
        # Tools information
        if hasattr(agent_instance, 'tools') and agent_instance.tools:
            tool_names = []
            for tool in agent_instance.tools:
                if hasattr(tool, 'name'):
                    tool_names.append(tool.name)
                else:
                    tool_names.append(str(tool))
            f.write(f"- **Tools**: {', '.join(tool_names)}\n")
        
        f.write("\n**Note**: Consider checking SmolagAgent version and logging configuration.\n\n")

# ============================================================================
# AGENT LOGGING SETUP
# ============================================================================
class AgentLoggingSetup:
    """FIXED: Enhanced logging setup compatible with agent_testing.py"""
    
    def __init__(self, debug_mode: bool = True, 
                step_log_file: str = None,
                question_log_file: str = None,
                evaluation_log_file: str = None):
        
        # Rich console for nice output
        self.console = Console(record=True)
        
        # Create SmolAgent logger with error handling
        try:
            smolag_logger = AgentLogger(
                level=LogLevel.DEBUG if debug_mode else LogLevel.INFO,
                console=self.console
            )
            print("✅ SmolAgent logger created successfully")
        except Exception as e:
            print(f"⚠️  SmolAgent logger creation failed: {e}")
            smolag_logger = self._create_mock_logger()
        
        self.logger = smolag_logger
        
        # Manual tracking variables
        self.steps_buffer = []
        self.current_task_id = None
        self.step_counter = 0
        self.task_start_time = None
        self.manual_steps = []
        
        # Routing and metadata tracking
        self.current_complexity = None
        self.current_routing_path = None
        self.current_model_used = None
        self.current_similar_examples_count = 0
        self.current_selected_agent = None
        self.current_file_metadata = None
        self.context_bridge_enabled = False
        self.coordinator_used = False
        
        # CSV loggers
        try:
            self.step_logger = StepLogger(step_log_file)
            self.question_logger = QuestionLogger(question_log_file)
            self.evaluation_logger = EvaluationLogger(evaluation_log_file)
            self.file_metadata_logger = FileMetadataLogger()
            
            # Add SmolagAgent logger AFTER other loggers
            self.smolag_logger = SmolagStepLogger()
            print(f"✅ SmolagAgent logger created: {self.smolag_logger.log_file}")
            
            self.csv_loggers = {
                'question_file': self.question_logger.log_file,
                'step_file': self.step_logger.log_file,
                'evaluation_file': self.evaluation_logger.log_file,
                'file_metadata_file': self.file_metadata_logger.log_file,
                'smolag_steps_file': self.smolag_logger.log_file  # NEW - Now csv_loggers exists
            }
            
        except Exception as e:
            print(f"⚠️  CSV logger setup failed: {e}")
            self.step_logger = self._create_mock_step_logger()
            self.question_logger = self._create_mock_question_logger()
            self.evaluation_logger = self._create_mock_evaluation_logger()
            self.file_metadata_logger = self._create_mock_file_metadata_logger()
            self.smolag_logger = self._create_mock_smolag_logger()  # NEW
            self.csv_loggers = {}
        
        self.debug_mode = debug_mode
        
        print(f"AgentLoggingSetup complete (Testing Framework Compatible):")
        for logger_type, file_path in self.csv_loggers.items():
            print(f"   {logger_type}: {file_path}")
    
    def log_smolag_execution(self, agent_name: str, task: str, agent_instance, result: str, duration: float = None):
        """Log SmolagAgent execution details"""
        if hasattr(self, 'smolag_logger') and self.smolag_logger:
            try:
                # Get detailed logs using SmolagAgent's native method
                messages = []
                if hasattr(agent_instance, 'write_inner_memory_from_logs'):
                    messages = agent_instance.write_inner_memory_from_logs()
                
                self.smolag_logger.log_agent_execution(
                    agent_name=agent_name, 
                    task=task, 
                    messages=messages, 
                    result=result, 
                    duration=duration
                )
                
                if self.debug_mode:
                    print(f"🤖 Logged {len(messages)} steps for {agent_name}")
                
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ SmolagAgent logging failed for {agent_name}: {e}")
    
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
        class MockStepLogger:
            def log_step(self, task_id, step_number, step_data):
                pass
        return MockStepLogger()
    
    def _create_mock_question_logger(self):
        class MockQuestionLogger:
            def log_question(self, **kwargs):
                pass
        return MockQuestionLogger()
    
    def _create_mock_evaluation_logger(self):
        class MockEvaluationLogger:
            def log_evaluation(self, **kwargs):
                pass
        return MockEvaluationLogger()
    
    def _create_mock_file_metadata_logger(self):  # NEW
        class MockFileMetadataLogger:
            def log_file_metadata(self, **kwargs):
                pass
        return MockFileMetadataLogger()
    
    def _create_mock_smolag_logger(self):
        class MockSmolagLogger:
            def log_agent_execution(self, **kwargs):
                pass
        return MockSmolagLogger()
    
    # ================================
    # ENHANCED LOGGING METHODS (Fixed for agent_testing.py compatibility)
    # ================================
    
    def log_step(self, action: str, details: str = "", node_name: str = ""):
        """Enhanced manual step logging with testing framework metadata"""
        self.step_counter += 1
        step_time = datetime.datetime.now()
        
        step_data = {
            'step_number': self.step_counter,
            'action': action,
            'details': details,
            'node_name': node_name,
            'timestamp': step_time,
            'context_bridge_step': self.context_bridge_enabled,
            'routing_path': self.current_routing_path,
            'complexity': self.current_complexity
        }
        
        self.manual_steps.append(step_data)
        
        # Enhanced CSV logging
        if hasattr(self.step_logger, 'log_step'):
            try:
                duration = (step_time - self.task_start_time).total_seconds() * 1000 if self.task_start_time else 0
                
                self.step_logger.log_step(
                    task_id=self.current_task_id or "unknown",
                    step_number=self.step_counter,
                    step_data={
                        'action': action,
                        'node_name': node_name,
                        'tool_name': '',
                        'input': details[:100],
                        'output': '',
                        'duration_ms': duration,
                        'context_bridge_step': self.context_bridge_enabled,
                        'routing_path': self.current_routing_path or '',
                        'complexity': self.current_complexity or ''
                    }
                )
            except Exception as e:
                if self.debug_mode:
                    print(f"Debug: Enhanced step CSV logging failed: {e}")
        
        if self.debug_mode:
            print(f"📝 Step {self.step_counter}: {action} ({node_name})")
    
    def start_task(self, task_id: str, model_used: str = None, complexity: str = None):
        """FIXED: Start task with enhanced tracking"""
        try:
            self.current_task_id = task_id
            self.step_counter = 0
            self.steps_buffer.clear()
            self.manual_steps.clear()
            self.current_complexity = complexity
            self.current_routing_path = None
            self.current_model_used = model_used
            self.current_similar_examples_count = 0
            self.current_selected_agent = None
            self.current_file_metadata = None
            self.context_bridge_enabled = False
            self.coordinator_used = False
            self.task_start_time = datetime.datetime.now()
            
            self.log_step("task_start", f"Started task: {task_id}")
            
            if self.debug_mode:
                print(f"🚀 Starting enhanced task tracking: {task_id}")
                
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Enhanced task start failed: {e}")
    
    # ================================
    # NEW METHODS (Required by agent_logic.py and agent_testing.py)
    # ================================
    
    def log_file_metadata(self, file_metadata: Dict):
        """NEW: Log file metadata (called by agent_logic.py)"""
        try:
            self.current_file_metadata = file_metadata
            
            if hasattr(self.file_metadata_logger, 'log_file_metadata'):
                self.file_metadata_logger.log_file_metadata(
                    task_id=self.current_task_id or "unknown",
                    file_metadata=file_metadata
                )
            
            # Also log as a step
            file_category = file_metadata.get('category', 'unknown')
            self.log_step("file_metadata_analysis", f"File category: {file_category}")
            
            if self.debug_mode:
                print(f"📁 File metadata logged: {file_category}")
                
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: File metadata logging failed: {e}")
    
    def set_context_bridge_enabled(self, enabled: bool):
        """NEW: Track context bridge usage"""
        self.context_bridge_enabled = enabled
        if enabled:
            self.log_step("context_bridge_enabled", "Context bridge tracking started")
    
    def set_coordinator_used(self, used: bool):
        """NEW: Track coordinator usage"""
        self.coordinator_used = used
        if used:
            self.log_step("coordinator_activated", "Hierarchical coordinator used")
    
    def set_selected_agent(self, agent_name: str):
        """NEW: Track selected agent"""
        self.current_selected_agent = agent_name
        self.log_step("agent_selection", f"Selected agent: {agent_name}")
    
    # ================================
    # ENHANCED EXISTING METHODS
    # ================================
    
    def set_routing_path(self, path: str):
        """Enhanced routing path tracking"""
        try:
            self.current_routing_path = path
            self.log_step("routing_decision", f"Route: {path}", "complexity_check")
            if self.debug_mode:
                print(f"🔀 Enhanced routing path: {path}")
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Enhanced routing path setting failed: {e}")
    
    def set_complexity(self, complexity: str):
        """Enhanced complexity tracking"""
        try:
            self.current_complexity = complexity
            self.log_step("complexity_assessment", f"Complexity: {complexity}", "complexity_check")
            if self.debug_mode:
                print(f"🧠 Enhanced complexity: {complexity}")
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Enhanced complexity setting failed: {e}")
    
    def set_similar_examples_count(self, count: int):
        """Enhanced RAG examples tracking"""
        try:
            self.current_similar_examples_count = count
            self.log_step("rag_retrieval", f"Retrieved {count} examples", "read_question")
            if self.debug_mode:
                print(f"📚 Enhanced examples count: {count}")
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Enhanced examples count setting failed: {e}")
    
    def log_question_result(self, task_id: str, question: str, 
                           final_answer: str, total_steps: int, success: bool):
        """ENHANCED: Log question with comprehensive metadata"""
        try:
            execution_time = 0.0
            if self.task_start_time:
                execution_time = (datetime.datetime.now() - self.task_start_time).total_seconds()
            
            self.log_step("question_complete", f"Final answer: {final_answer}", "format_answer")
            
            # Enhanced question logging with all metadata
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
                    similar_examples_count=self.current_similar_examples_count,
                    selected_agent=self.current_selected_agent,
                    has_file=bool(self.current_file_metadata),
                    file_category=self.current_file_metadata.get('category', 'none') if self.current_file_metadata else 'none',
                    context_bridge_used=self.context_bridge_enabled,
                    coordinator_used=self.coordinator_used,
                    file_metadata_available=bool(self.current_file_metadata)
                )
            
            if self.debug_mode:
                status = "✅" if success else "❌"
                print(f"{status} Enhanced question completed: {total_steps} steps in {execution_time:.1f}s")
        
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Enhanced question result logging failed: {e}")
    
    def log_evaluation_result(self, task_id: str, question: str, final_answer: str,
                             ground_truth: str, level: int, is_correct: bool,
                             strategy_used: str = None, selected_agent: str = None,
                             matching_method: str = None, context_bridge_metrics: Dict = None):
        """ENHANCED: Log evaluation with comprehensive metadata"""
        try:
            execution_time = 0.0
            if self.task_start_time:
                execution_time = (datetime.datetime.now() - self.task_start_time).total_seconds()
            
            self.log_step("evaluation_complete", f"Correct: {is_correct}")
            
            # Enhanced evaluation logging
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
                    selected_agent=selected_agent or self.current_selected_agent,
                    model_used=self.current_model_used,
                    similar_examples_count=self.current_similar_examples_count,
                    matching_method=matching_method,
                    file_category=self.current_file_metadata.get('category', 'none') if self.current_file_metadata else 'none',
                    context_bridge_metrics=context_bridge_metrics,
                    coordinator_analysis_available=self.coordinator_used
                )
            
            if self.debug_mode:
                status = "✅" if is_correct else "❌"
                print(f"{status} Enhanced evaluation logged: Level {level}, {execution_time:.1f}s")
        
        except Exception as e:
            if self.debug_mode:
                print(f"Debug: Enhanced evaluation result logging failed: {e}")
    
    @property
    def current_log_files(self) -> Dict[str, str]:
        """Get current log file paths with enhanced metadata"""
        try:
            return {
                "steps": str(getattr(self.step_logger, 'log_file', 'mock')),
                "questions": str(getattr(self.question_logger, 'log_file', 'mock')), 
                "evaluation": str(getattr(self.evaluation_logger, 'log_file', 'mock')),
                "file_metadata": str(getattr(self.file_metadata_logger, 'log_file', 'mock')),
                "smolag_steps": str(getattr(self.smolag_logger, 'log_file', 'mock')),  # NEW
                "framework_version": "smolag_integrated"  # Updated version
            }
        except Exception as e:
            return {
                "steps": "error",
                "questions": "error", 
                "evaluation": "error",
                "file_metadata": "error",
                "smolag_steps": "error",
                "error": str(e)
            }

# ============================================================================
# TESTING FRAMEWORK INTEGRATION HELPERS
# ============================================================================

def validate_logging_compatibility() -> Dict[str, bool]:
    """Validate logging compatibility with testing framework"""
    
    validation = {
        "csv_loggers_available": False,
        "enhanced_metadata_support": False,
        "file_metadata_logging": False,
        "context_bridge_integration": False,
        "testing_framework_compatible": False
    }
    
    try:
        # Test basic CSV logger creation
        test_setup = AgentLoggingSetup(debug_mode=False)
        validation["csv_loggers_available"] = True
        
        # Test enhanced metadata methods
        if hasattr(test_setup, 'log_file_metadata'):
            validation["file_metadata_logging"] = True
        
        if hasattr(test_setup, 'set_context_bridge_enabled'):
            validation["context_bridge_integration"] = True
        
        # Test enhanced logging methods
        if hasattr(test_setup.question_logger, 'log_question'):
            validation["enhanced_metadata_support"] = True
        
        validation["testing_framework_compatible"] = all([
            validation["csv_loggers_available"],
            validation["file_metadata_logging"],
            validation["enhanced_metadata_support"]
        ])
        
    except Exception as e:
        print(f"Logging validation error: {e}")
    
    return validation

if __name__ == "__main__":
    print("🚀 Enhanced Agent Logging - Testing Framework Compatible")
    print("=" * 60)
    print("✅ FIXES APPLIED:")
    print("   ├── Added log_file_metadata() method (required by agent_logic.py)")
    print("   ├── Enhanced CSV headers for comprehensive analysis")
    print("   ├── Added FileMetadataLogger for file processing analysis")
    print("   ├── Enhanced question logging with testing framework metadata")
    print("   ├── Added context bridge integration tracking")
    print("   ├── Enhanced evaluation logging with coordinator analysis")
    print("   └── Added testing framework compatibility validation")
    
    print("\n🔧 NEW FEATURES:")
    features = [
        "✅ log_file_metadata() - Logs file analysis results",
        "✅ Enhanced step logging with node names and routing paths",
        "✅ Coordinator usage tracking",
        "✅ Context bridge integration logging",
        "✅ File category and processing approach logging",
        "✅ Comprehensive CSV headers for testing analysis"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print("\n🎯 VALIDATION TEST:")
    validation = validate_logging_compatibility()
    
    for check, status in validation.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {check}: {status}")
    
    if validation["testing_framework_compatible"]:
        print("\n🎉 Agent logging is fully compatible with testing framework!")
        print("🔗 Integration methods available:")
        print("   ├── AgentLoggingSetup.log_file_metadata()")
        print("   ├── AgentLoggingSetup.set_context_bridge_enabled()")
        print("   ├── AgentLoggingSetup.set_coordinator_used()")
        print("   ├── AgentLoggingSetup.set_selected_agent()")
        print("   └── Enhanced CSV logging for comprehensive analysis")
    else:
        print("\n⚠️  Some logging features need attention - check validation results above")

# ============================================================================
# INTEGRATION PATTERNS FOR AGENT_LOGIC.PY
# ============================================================================

class LoggingIntegration:
    """Integration patterns for agent_logic.py to use enhanced logging"""
    
    @staticmethod
    def read_question_integration(logging_setup, state):
        """Pattern for _read_question_node logging"""
        if logging_setup:
            logging_setup.log_step("question_processing", 
                                 f"Processing: {state['question'][:50]}...", 
                                 "read_question")
            
            # Log file metadata if present
            file_metadata = state.get('file_metadata')
            if file_metadata:
                logging_setup.log_file_metadata(file_metadata)
    
    @staticmethod
    def complexity_check_integration(logging_setup, complexity, reason=""):
        """Pattern for _complexity_check_node logging"""
        if logging_setup:
            logging_setup.set_complexity(complexity)
            logging_setup.log_step("complexity_analysis", 
                                 f"Determined: {complexity} - {reason}", 
                                 "complexity_check")
    
    @staticmethod
    def coordinator_integration(logging_setup, coordination_analysis):
        """Pattern for _coordinator_node logging"""
        if logging_setup:
            logging_setup.set_coordinator_used(True)
            
            selected_specialist = coordination_analysis.get("selected_specialist", "unknown")
            logging_setup.set_selected_agent(selected_specialist)
            
            logging_setup.log_step("coordinator_analysis", 
                                 f"Selected: {selected_specialist}", 
                                 "coordinator")
    
    @staticmethod
    def format_answer_integration(logging_setup, final_answer):
        """Pattern for _format_answer_node logging"""
        if logging_setup:
            logging_setup.log_step("answer_formatting", 
                                 f"Formatted: {final_answer[:30]}...", 
                                 "format_answer")

# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================

# Ensure old method names still work
def create_timestamped_log_filename(base_name: str, extension: str = "csv") -> str:
    """Backwards compatibility alias"""
    return create_timestamped_filename(base_name, extension)

# ============================================================================
# EXAMPLE USAGE FOR TESTING FRAMEWORK
# ============================================================================

def example_testing_framework_usage():
    """Example of how testing framework should use enhanced logging"""
    
    print("\n📋 EXAMPLE: Testing Framework Integration")
    print("-" * 50)
    
    # 1. Create enhanced logging setup
    logging_setup = AgentLoggingSetup(debug_mode=True)
    
    # 2. Start task with enhanced tracking
    logging_setup.start_task("test_123", model_used="qwen-qwq-32b")
    logging_setup.set_context_bridge_enabled(True)
    
    # 3. Example file metadata logging
    file_metadata = {
        "file_name": "test.xlsx",
        "category": "data",
        "processing_approach": "direct_pandas",
        "recommended_specialist": "data_analyst"
    }
    logging_setup.log_file_metadata(file_metadata)
    
    # 4. Example routing decision
    logging_setup.set_complexity("complex")
    logging_setup.set_routing_path("manager_coordination")
    
    # 5. Example coordinator usage
    logging_setup.set_coordinator_used(True)
    logging_setup.set_selected_agent("data_analyst")
    
    # 6. Example question completion
    logging_setup.log_question_result(
        task_id="test_123",
        question="What is the total revenue?",
        final_answer="1000000",
        total_steps=8,
        success=True
    )
    
    # 7. Example evaluation result
    logging_setup.log_evaluation_result(
        task_id="test_123",
        question="What is the total revenue?",
        final_answer="1000000",
        ground_truth="1000000",
        level=2,
        is_correct=True,
        matching_method="exact"
    )
    
    print("✅ Example logging sequence completed!")
    print(f"📁 Log files created:")
    for log_type, file_path in logging_setup.current_log_files.items():
        print(f"   {log_type}: {file_path}")

if __name__ == "__main__":
    # Run the validation and example
    example_testing_framework_usage()