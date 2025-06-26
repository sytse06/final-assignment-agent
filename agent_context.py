# agent_context.py

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
    
    @classmethod
    def start_task_execution(cls, task_id: str):
        """Start tracking task execution"""
        cls.active_task_id.set(task_id)
        cls.execution_start.set(time.time())
        cls.step_counter.set(0)
        print(f"🚀 Started execution tracking: {task_id}")
    
    @classmethod
    def track_operation(cls, operation: str):
        """Track current operation"""
        cls.current_operation.set(operation)
        current_step = cls.step_counter.get(0)
        cls.step_counter.set(current_step + 1)
        print(f"📝 Step {current_step + 1}: {operation}")
    
    @classmethod
    def get_task_id(cls) -> Optional[str]:
        """Get active task ID"""
        return cls.active_task_id.get()
    
    @classmethod
    def get_execution_metrics(cls) -> Dict:
        """Get current execution metrics"""
        start_time = cls.execution_start.get()
        return {
            "execution_time": time.time() - start_time if start_time else 0,
            "steps_executed": cls.step_counter.get(0),
            "current_operation": cls.current_operation.get()
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
        
        print(f"🏁 Execution complete: {task_id}, {metrics['steps_executed']} steps, {metrics['execution_time']:.2f}s")

if __name__ == "__main__":
    print("🧹 CONTEXT BRIDGE")
    print("=" * 25)
    print("✅ Ready for execution tracking")
    
    # Quick demo
    ContextBridge.start_task_execution("demo_123")
    ContextBridge.track_operation("Processing question")
    ContextBridge.track_operation("Executing agent")
    
    metrics = ContextBridge.get_execution_metrics()
    print(f"\nMetrics: {metrics}")
    
    ContextBridge.clear_tracking()
    print("Demo complete!")