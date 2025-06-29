from contextvars import ContextVar
from typing import Dict, Optional
import time


class ContextBridge:
    """
    Simple context bridge for execution tracking.
    That's it. Nothing more.
    """
    
    # Real-time execution tracking
    current_operation: ContextVar[Optional[str]] = ContextVar('current_operation', default=None)
    step_counter: ContextVar[int] = ContextVar('step_counter', default=0)
    execution_start: ContextVar[Optional[float]] = ContextVar('execution_start', default=None)
    active_task_id: ContextVar[Optional[str]] = ContextVar('active_task_id', default=None)
    last_error: ContextVar[Optional[str]] = ContextVar('last_error', default=None)
    
    @classmethod
    def start_task_execution(cls, task_id: str):
        """Start tracking task execution"""
        cls.active_task_id.set(task_id)
        cls.execution_start.set(time.time())
        cls.step_counter.set(0)
        cls.last_error.set(None)
        print(f"🚀 Started execution tracking: {task_id}")
    
    @classmethod
    def track_operation(cls, operation: str):
        """Track current operation"""
        cls.current_operation.set(operation)
        current_step = cls.step_counter.get(0)
        cls.step_counter.set(current_step + 1)
        print(f"📝 Step {current_step + 1}: {operation}")
    
    @classmethod
    def track_error(cls, error: str):
        """Track error occurrence"""
        cls.last_error.set(error)
        cls.track_operation(f"ERROR: {error}")
        print(f"❌ Error tracked: {error}")
    
    @classmethod
    def get_execution_metrics(cls) -> Dict:
        """Get current execution metrics"""
        start_time = cls.execution_start.get()
        return {
            "execution_time": time.time() - start_time if start_time else 0,
            "steps_executed": cls.step_counter.get(0),
            "current_operation": cls.current_operation.get(),
            "last_error": cls.last_error.get()
        }
    
    @classmethod
    def clear_tracking(cls):
        """Clear execution tracking"""
        task_id = cls.active_task_id.get()
        metrics = cls.get_execution_metrics()
        
        cls.active_task_id.set(None)
        cls.execution_start.set(None)
        cls.step_counter.set(0)
        cls.current_operation.set(None)
        cls.last_error.set(None)
        
        print(f"🏁 Execution complete: {task_id}, {metrics['steps_executed']} steps, {metrics['execution_time']:.2f}s")

# ============================================================================
# SIMPLE INTEGRATION (Just use this in agent_logic.py)
# ============================================================================

def track(operation: str, config):
    """Simple tracking - just one function"""
    if config.enable_context_bridge:
        try:
            ContextBridge.track_operation(operation)
        except:
            pass  # Silent fail - don't care about errors