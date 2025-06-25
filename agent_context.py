# agent_context.py
# Context Bridge for LangGraph ↔ SmolagAgents Integration
# Architecture principle: Static info from LangGraph state, dynamic tracking via context bridge

from contextvars import ContextVar
from typing import Dict, Any, Optional, List
import time
from datetime import datetime
import uuid
import inspect

# Import SmolagAgents Tool base class
from smolagents import Tool

# Import for tool availability check
try:
    from tools import GetAttachmentTool, ContentRetrieverTool
    CUSTOM_TOOLS_AVAILABLE = True
except ImportError:
    CUSTOM_TOOLS_AVAILABLE = False

class ContextBridge:
    """
    Context bridge for real-time tracking and cross-boundary communication.
    Static information (file paths, task metadata) comes from LangGraph state.
    Dynamic information (execution steps, timing) tracked here.
    """
    
    # Real-time execution tracking (dynamic)
    current_operation: ContextVar[Optional[str]] = ContextVar('current_operation', default=None)
    step_counter: ContextVar[int] = ContextVar('step_counter', default=0)
    execution_start: ContextVar[Optional[float]] = ContextVar('execution_start', default=None)
    
    # Cross-boundary data (for SmolagAgents)
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
        """Get active task ID for tool configuration"""
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

class StateAwareGetAttachmentTool(Tool):
    """
    GetAttachmentTool that gets configuration from LangGraph state.
    Uses context bridge for task_id as fallback.
    """
    
    name = "get_attachment"
    description = """
    Get attachment files for a GAIA task. Gets task_id from state or context bridge.
    Use this when you need to access files associated with the current task.
    """
    
    inputs = {
        "task_id": {
            "type": "string", 
            "description": "Task ID to get attachments for. Auto-filled from state/context.",
            "nullable": True
        }
    }
    
    output_type = "string"
    
    def __init__(self, original_tool, **kwargs):
        super().__init__(**kwargs)
        self.original_tool = original_tool
        self._configured_task_id = None
    
    def configure_from_state(self, task_id: str, file_path: str = None):
        """Configure tool from LangGraph state (called by workflow node)"""
        self._configured_task_id = task_id
        if hasattr(self.original_tool, 'attachment_for'):
            self.original_tool.attachment_for(task_id)
        if file_path and hasattr(self.original_tool, 'set_file_path'):
            self.original_tool.set_file_path(file_path)
        print(f"🔧 GetAttachmentTool configured from state: {task_id}")
    
    def forward(self, task_id: str = None) -> str:
        """
        Get attachments with state/context fallback.
        """
        # Priority: explicit parameter > state configuration > context bridge
        if task_id is None:
            if self._configured_task_id:
                task_id = self._configured_task_id
                print(f"🔧 Using state-configured task_id: {task_id}")
            else:
                # Fallback to context bridge
                context_task_id = ContextBridge.get_task_id()
                if context_task_id:
                    task_id = context_task_id
                    print(f"🔧 Using context bridge task_id: {task_id}")
                else:
                    raise ValueError("No task_id provided and none found in state or context")
        
        try:
            result = self.original_tool(task_id)
            ContextBridge.track_operation(f"GetAttachment successful: {task_id}")
            return result
        except Exception as e:
            error_msg = f"GetAttachmentTool failed for task {task_id}: {str(e)}"
            ContextBridge.track_operation(f"GetAttachment failed: {str(e)}")
            return error_msg

class StateAwareContentRetrieverTool(Tool):
    """
    ContentRetrieverTool that can access question from state.
    """
    
    name = "content_retriever"
    description = """
    Retrieve and process content from various sources with state awareness.
    Gets context question from state when available.
    """
    
    inputs = {
        "query": {
            "type": "string",
            "description": "Query or content to retrieve"
        },
        "context_question": {
            "type": "string", 
            "description": "Context question for better retrieval.",
            "nullable": True
        }
    }
    
    output_type = "string"
    
    def __init__(self, original_tool, **kwargs):
        super().__init__(**kwargs)
        self.original_tool = original_tool
        self._state_question = None
    
    def configure_from_state(self, question: str):
        """Configure tool from LangGraph state (called by workflow node)"""
        self._state_question = question
        print(f"🔧 ContentRetriever configured with question: {question[:50]}...")
    
    def forward(self, query: str, context_question: str = None) -> str:
        """
        Retrieve content with state-aware context.
        """
        # Use state-configured question if available
        if context_question is None and self._state_question:
            context_question = self._state_question
            print(f"🔧 Using state-configured question for context")
        
        try:
            result = self.original_tool(query, context_question=context_question)
            ContextBridge.track_operation(f"ContentRetriever successful: {query[:30]}...")
            return result
        except Exception as e:
            error_msg = f"ContentRetrieverTool failed: {str(e)}"
            ContextBridge.track_operation(f"ContentRetriever failed: {str(e)}")
            return error_msg

def create_state_aware_tools(shared_tools: Dict) -> List:
    """Create state-aware tool wrappers for LangGraph integration"""
    
    state_aware_tools = []
    
    for tool_name, base_tool in shared_tools.items():
        if tool_name == "get_attachment":
            state_tool = StateAwareGetAttachmentTool(base_tool)
        elif tool_name == "content_retriever":
            state_tool = StateAwareContentRetrieverTool(base_tool)
        else:
            # Generic state-aware wrapper for other tools
            state_tool = create_generic_state_wrapper(base_tool)
        
        state_aware_tools.append(state_tool)
        print(f"✅ Created state-aware {tool_name}")
    
    return state_aware_tools

def create_generic_state_wrapper(base_tool):
    """Create generic state-aware wrapper for other tools"""
    
    class GenericStateAwareTool(Tool):
        """Generic state-aware tool wrapper"""
        
        def __init__(self, base_tool):
            self.base_tool = base_tool
            
            # Copy attributes from base tool
            self.name = getattr(base_tool, 'name', 'unknown_tool')
            self.description = getattr(base_tool, 'description', 'State-aware tool wrapper')
            self.inputs = getattr(base_tool, 'inputs', {
                "query": {"type": "string", "description": "Input query", "nullable": True}
            })
            self.output_type = getattr(base_tool, 'output_type', "string")
            
            # Ensure nullable flags
            self._ensure_inputs_nullable()
            super().__init__()
            self._create_forward_method()
        
        def _ensure_inputs_nullable(self):
            """Ensure inputs have nullable flags"""
            for key, schema in self.inputs.items():
                if 'nullable' not in schema:
                    schema['nullable'] = True
        
        def _create_forward_method(self):
            """Create compliant forward method"""
            input_keys = list(self.inputs.keys())
            
            def forward_impl(self, **kwargs):
                # Filter to valid parameters
                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                 if k in input_keys and v is not None}
                
                try:
                    result = self.base_tool.forward(**filtered_kwargs)
                    ContextBridge.track_operation(f"{self.name} successful")
                    return result
                except Exception as e:
                    error_msg = f"{self.name} failed: {str(e)}"
                    ContextBridge.track_operation(error_msg)
                    return error_msg
            
            # Set method signature
            if input_keys:
                import inspect
                parameters = [inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
                for key in input_keys:
                    param = inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY, default=None)
                    parameters.append(param)
                forward_impl.__signature__ = inspect.Signature(parameters)
            
            import types
            self.forward = types.MethodType(forward_impl, self)
    
    return GenericStateAwareTool(base_tool)

# Compatibility layer for existing code
class ContextVariableFlow:
    """
    Compatibility layer that redirects to ContextBridge.
    """
    
    @classmethod
    def set_task_context(cls, task_id: str, question: str, metadata: Dict[str, Any] = None):
        """Compatibility: redirect to simplified bridge"""
        ContextBridge.start_task_execution(task_id)
        if metadata and metadata.get('routing_path'):
            ContextBridge.track_operation(f"Routing: {metadata['routing_path']}")
    
    @classmethod
    def get_task_id(cls) -> Optional[str]:
        """Compatibility: get task ID"""
        return ContextBridge.get_task_id()
    
    @classmethod
    def add_workflow_step(cls, task_id: str, step_name: str, details: str = ""):
        """Compatibility: track workflow step"""
        ContextBridge.track_operation(f"{step_name}: {details}")
    
    @classmethod
    def update_routing_path(cls, path: str):
        """Compatibility: track routing update"""
        ContextBridge.track_operation(f"Routing updated: {path}")
    
    @classmethod
    def update_complexity(cls, complexity: str):
        """Compatibility: track complexity update"""
        ContextBridge.track_operation(f"Complexity: {complexity}")
    
    @classmethod
    def get_context_summary(cls) -> str:
        """Compatibility: get execution summary"""
        task_id = ContextBridge.get_task_id()
        metrics = ContextBridge.get_execution_metrics()
        
        if not task_id:
            return "No active context"
        
        return f"Task: {task_id} | Steps: {metrics['steps_executed']} | Time: {metrics['execution_time']:.2f}s"
    
    @classmethod
    def clear_context(cls):
        """Compatibility: clear context"""
        ContextBridge.clear_tracking()
    
    @classmethod
    def get_step_count(cls, task_id: str) -> Dict:
        """Compatibility: get step metrics"""
        metrics = ContextBridge.get_execution_metrics()
        return {
            "workflow_steps": 0,  # Not tracked in simplified version
            "agent_steps": metrics['steps_executed'],
            "total_steps": metrics['steps_executed'],
            "step_log": []  # Simplified logging
        }

# Alias for backward compatibility
create_context_aware_tools = create_state_aware_tools

if __name__ == "__main__":
    print("🌉 Simplified Context Bridge - Hybrid Approach")
    print("=" * 50)
    
    # Demo the simplified approach
    print("\n📚 Demo: Simplified Context Bridge")
    
    # Start task execution
    ContextBridge.start_task_execution("demo_123")
    
    # Track some operations
    ContextBridge.track_operation("Processing question")
    ContextBridge.track_operation("Selecting agent")
    ContextBridge.track_operation("Executing agent")
    
    # Show metrics
    metrics = ContextBridge.get_execution_metrics()
    print(f"Execution metrics: {metrics}")
    
    # Clear tracking
    ContextBridge.clear_tracking()
    print("Context cleared successfully")