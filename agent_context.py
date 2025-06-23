# agent_context.py
# Context Variables Infrastructure for LangGraph ↔ SmolagAgents Bridging

from contextvars import ContextVar
from typing import Dict, Any, Optional, List
import time
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

class ContextVariableFlow:
    """
    Thread-safe context management for LangGraph → SmolagAgents information flow.
    Solves the critical problem of passing task_id and context from LangGraph state
    to SmolagAgent tools that need this information to function correctly.
    """
    
    # Core context variables
    current_task_id: ContextVar[Optional[str]] = ContextVar('current_task_id', default=None)
    current_question: ContextVar[Optional[str]] = ContextVar('current_question', default=None)
    current_metadata: ContextVar[Optional[Dict[str, Any]]] = ContextVar('current_metadata', default=None)
    
    # Execution tracking
    execution_start_time: ContextVar[Optional[float]] = ContextVar('execution_start_time', default=None)
    routing_path: ContextVar[Optional[str]] = ContextVar('routing_path', default=None)
    complexity_level: ContextVar[Optional[str]] = ContextVar('complexity_level', default=None)
    
    @classmethod
    def set_task_context(cls, task_id: str, question: str, metadata: Dict[str, Any] = None):
        """
        Set context for current task execution (thread-safe).
        
        Args:
            task_id: Unique identifier for the GAIA task
            question: The question being processed
            metadata: Additional context like complexity, routing_path, etc.
        """
        cls.current_task_id.set(task_id)
        cls.current_question.set(question)
        cls.current_metadata.set(metadata or {})
        cls.execution_start_time.set(time.time())
        
        # Extract common metadata
        if metadata:
            if 'complexity' in metadata:
                cls.complexity_level.set(metadata['complexity'])
            if 'routing_path' in metadata:
                cls.routing_path.set(metadata['routing_path'])
        
        print(f"🔧 Context set: task_id={task_id}, question='{question[:30]}...', metadata={metadata}")
    
    @classmethod
    def get_task_context(cls) -> Dict[str, Any]:
        """
        Get current task context (thread-safe).
        
        Returns:
            Dictionary with all current context information
        """
        return {
            "task_id": cls.current_task_id.get(),
            "question": cls.current_question.get(),
            "metadata": cls.current_metadata.get() or {},
            "execution_start_time": cls.execution_start_time.get(),
            "routing_path": cls.routing_path.get(),
            "complexity": cls.complexity_level.get()
        }
    
    @classmethod
    def get_task_id(cls) -> Optional[str]:
        """Quick access to current task_id"""
        return cls.current_task_id.get()
    
    @classmethod
    def get_question(cls) -> Optional[str]:
        """Quick access to current question"""
        return cls.current_question.get()
    
    @classmethod
    def update_routing_path(cls, path: str):
        """Update the routing path during execution"""
        cls.routing_path.set(path)
        
        # Also update in metadata
        current_metadata = cls.current_metadata.get() or {}
        current_metadata['routing_path'] = path
        cls.current_metadata.set(current_metadata)
        
        print(f"🛤️  Routing path updated: {path}")
    
    @classmethod
    def update_complexity(cls, complexity: str):
        """Update complexity assessment during execution"""
        cls.complexity_level.set(complexity)
        
        # Also update in metadata
        current_metadata = cls.current_metadata.get() or {}
        current_metadata['complexity'] = complexity
        cls.current_metadata.set(current_metadata)
        
        print(f"📊 Complexity updated: {complexity}")
    
    @classmethod
    def get_execution_time(cls) -> Optional[float]:
        """Get current execution time in seconds"""
        start_time = cls.execution_start_time.get()
        if start_time:
            return time.time() - start_time
        return None
    
    @classmethod
    def clear_context(cls):
        """Clear all context variables"""
        cls.current_task_id.set(None)
        cls.current_question.set(None)
        cls.current_metadata.set(None)
        cls.execution_start_time.set(None)
        cls.routing_path.set(None)
        cls.complexity_level.set(None)
        
        print("🧹 Context variables cleared")
    
    @classmethod
    def is_context_active(cls) -> bool:
        """Check if context is currently set"""
        return cls.current_task_id.get() is not None
    
    @classmethod
    def get_context_summary(cls) -> str:
        """Get a human-readable summary of current context"""
        if not cls.is_context_active():
            return "No active context"
        
        context = cls.get_task_context()
        execution_time = cls.get_execution_time()
        
        summary = [
            f"Task ID: {context['task_id']}",
            f"Question: {context['question'][:50]}..." if context['question'] else "No question",
            f"Complexity: {context['complexity']}" if context['complexity'] else "Unknown complexity",
            f"Routing: {context['routing_path']}" if context['routing_path'] else "Unknown routing",
            f"Runtime: {execution_time:.2f}s" if execution_time else "Unknown runtime"
        ]
        
        return " | ".join(summary)


class ContextAwareGetAttachmentTool(Tool):
    """
    Context-aware wrapper for GetAttachmentTool that automatically uses
    task_id from context variables when not explicitly provided.
    
    This is a proper SmolagAgents Tool that inherits from Tool base class.
    """
    
    name = "get_attachment"
    description = """
    Get attachment files for a GAIA task. Automatically uses task_id from context if not provided.
    Use this when you need to access files associated with the current task.
    """
    
    inputs = {
        "task_id": {
            "type": "string", 
            "description": "Task ID to get attachments for. If not provided, will use current context task_id.",
            "nullable": True
        }
    }
    
    output_type = "string"
    
    def __init__(self, original_tool, **kwargs):
        super().__init__(**kwargs)
        self.original_tool = original_tool
    
    def forward(self, task_id: str = None) -> str:
        """
        Enhanced call with automatic context fallback.
        
        Args:
            task_id: Optional task ID. If not provided, will try to get from context.
            
        Returns:
            Result from original GetAttachmentTool
            
        Raises:
            ValueError: If no task_id provided and none found in context
        """
        if task_id is None:
            # Try to get from context variables
            context_task_id = ContextVariableFlow.get_task_id()
            
            if context_task_id:
                task_id = context_task_id
                print(f"🔧 GetAttachmentTool auto-detected task_id: {task_id}")
            else:
                available_context = ContextVariableFlow.get_context_summary()
                raise ValueError(
                    f"No task_id provided and none found in context. "
                    f"Current context: {available_context}"
                )
        
        try:
            # Call original tool
            result = self.original_tool(task_id)
            print(f"✅ GetAttachmentTool successful for task: {task_id}")
            return result
        except Exception as e:
            error_msg = f"GetAttachmentTool failed for task {task_id}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg


class ContextAwareContentRetrieverTool(Tool):
    """
    Context-aware wrapper for ContentRetrieverTool that can access
    question context when needed.
    
    This is a proper SmolagAgents Tool that inherits from Tool base class.
    """
    
    name = "content_retriever"
    description = """
    Retrieve and process content from various sources with context awareness.
    Automatically includes current question context for better content retrieval.
    """
    
    inputs = {
        "query": {
            "type": "string",
            "description": "Query or content to retrieve"
        },
        "context_question": {
            "type": "string", 
            "description": "Context question for better retrieval. Auto-filled from current context.",
            "nullable": True
        }
    }
    
    output_type = "string"
    
    def __init__(self, original_tool, **kwargs):
        super().__init__(**kwargs)
        self.original_tool = original_tool
    
    def forward(self, query: str, context_question: str = None) -> str:
        """
        Enhanced call with context awareness.
        
        Args:
            query: Main query for content retrieval
            context_question: Optional context question, auto-filled from context
            
        Returns:
            Retrieved content
        """
        # Get context information
        if context_question is None:
            context_question = ContextVariableFlow.get_question()
        
        context_task_id = ContextVariableFlow.get_task_id()
        
        print(f"🔧 ContentRetrieverTool context: task_id={context_task_id}, question='{context_question[:30] if context_question else 'None'}...'")
        
        try:
            # Call original tool with context
            result = self.original_tool(
                query, 
                context_question=context_question,
                context_task_id=context_task_id
            )
            print(f"✅ ContentRetrieverTool successful")
            return result
        except Exception as e:
            error_msg = f"ContentRetrieverTool failed: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

def create_context_aware_tools(shared_tools: Dict) -> List:
    """Create context-aware tool wrappers with automatic parameter filtering"""
    
    context_aware_tools = []
    
    for tool_name, base_tool in shared_tools.items():
        context_tool = create_safe_context_wrapper(base_tool)
        context_aware_tools.append(context_tool)
        print(f"✅ Created context-aware {tool_name} with parameter filtering")
    
    return context_aware_tools

def create_safe_context_wrapper(base_tool):
    """Create context-aware wrapper that automatically filters unknown parameters"""
    
    class SafeContextAwareTool(Tool):
        def __init__(self, base_tool):
            self.base_tool = base_tool
            self.name = getattr(base_tool, 'name', 'unknown_tool')
            self.description = getattr(base_tool, 'description', '')
            
            # Required Tool attributes - copy from base tool or use defaults
            self.inputs = getattr(base_tool, 'inputs', {
                "query": {"type": "string", "description": "Input for the tool"}
            })
            self.output_type = getattr(base_tool, 'output_type', "string")
            
            # Initialize base Tool class AFTER setting required attributes
            super().__init__()
            
            # Cache the base tool's parameter signature for efficiency
            self._cached_signature = None
            
            # Dynamically create forward method with correct signature
            self._create_dynamic_forward_method()
        
        def _create_dynamic_forward_method(self):
            """Dynamically create forward method that matches inputs exactly"""
            
            # Get the expected parameters from inputs
            expected_params = list(self.inputs.keys())
            
            # Create parameter string for the method signature
            if expected_params:
                params_str = ", ".join(f"{param}=None" for param in expected_params)
                method_code = f"""
def forward(self, {params_str}):
    return self._execute_with_context({', '.join(f'{param}={param}' for param in expected_params)})
"""
            else:
                method_code = """
def forward(self):
    return self._execute_with_context()
"""
            
            # Execute the dynamic method creation
            namespace = {'self': self}
            exec(method_code, namespace)
            
            # Bind the method to this instance
            import types
            self.forward = types.MethodType(namespace['forward'], self)
        
        def _execute_with_context(self, **kwargs):
            """Execute the base tool with context and parameter filtering"""
            
            # Get context information
            try:
                context = ContextVariableFlow.get_task_context()
                task_id = context.get('task_id') if context else None
                question = context.get('question') if context else None
            except:
                task_id = None
                question = None
            
            # Set task_id for tools that need it (like GetAttachmentTool)
            if task_id and hasattr(self.base_tool, 'attachment_for'):
                try:
                    self.base_tool.attachment_for(task_id)
                    print(f"🔧 Set task_id {task_id} for {self.name}")
                except Exception as e:
                    print(f"⚠️ Could not set task_id for {self.name}: {e}")
            
            # Filter kwargs to only include parameters the base tool accepts
            valid_params = self._get_valid_parameters()
            filtered_kwargs = {}
            
            for param_name, param_value in kwargs.items():
                if param_value is not None and param_name in valid_params:
                    filtered_kwargs[param_name] = param_value
                elif param_value is not None:
                    # Log filtered parameters for debugging
                    print(f"🔧 {self.name}: Filtered parameter '{param_name}'")
            
            # Enhance query with context if possible
            if 'query' in valid_params and question:
                if 'query' not in filtered_kwargs or not filtered_kwargs['query']:
                    filtered_kwargs['query'] = question
            
            try:
                # Call base tool with filtered parameters
                result = self.base_tool.forward(**filtered_kwargs)
                return result
                
            except Exception as e:
                print(f"❌ {self.name} execution failed: {e}")
                raise
        
        def _get_valid_parameters(self):
            """Get the parameter names that the base tool's forward method accepts"""
            if self._cached_signature is None:
                try:
                    sig = inspect.signature(self.base_tool.forward)
                    self._cached_signature = set(sig.parameters.keys())
                except Exception:
                    # Fallback: assume common parameters
                    self._cached_signature = {'url', 'query', 'task', 'question', 'fmt', 'format'}
            
            return self._cached_signature
    
    return SafeContextAwareTool(base_tool)


# Convenience functions for common patterns
def with_context(func):
    """
    Decorator to ensure function runs with proper context handling.
    """
    def wrapper(*args, **kwargs):
        if not ContextVariableFlow.is_context_active():
            print("⚠️  Function called without active context")
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            current_context = ContextVariableFlow.get_context_summary()
            print(f"❌ Function failed with context: {current_context}")
            raise
    
    return wrapper


def ensure_task_context(task_id: str = None, question: str = None):
    """
    Ensure minimum context is available, creating if necessary.
    """
    if not ContextVariableFlow.is_context_active():
        if not task_id:
            task_id = str(uuid.uuid4())
        if not question:
            question = "Context restoration"
        
        ContextVariableFlow.set_task_context(task_id, question, {"restored": True})
        print(f"🔄 Context restored: {task_id}")
    
    return ContextVariableFlow.get_task_context()


if __name__ == "__main__":
    print("🌉 Context Bridge Module - FIXED VERSION")
    print("=" * 50)
    
    # Demo usage
    print("\n📚 Demo: Context Variable Flow")
    
    # Set context
    ContextVariableFlow.set_task_context(
        task_id="demo_123",
        question="What is the capital of France?",
        metadata={"complexity": "simple", "routing_path": "one_shot"}
    )
    
    # Show context access
    print(f"Context summary: {ContextVariableFlow.get_context_summary()}")
    print(f"Task ID: {ContextVariableFlow.get_task_id()}")
    print(f"Question: {ContextVariableFlow.get_question()}")
    
    # Update routing
    ContextVariableFlow.update_routing_path("manager_coordination")
    print(f"Updated context: {ContextVariableFlow.get_context_summary()}")
    
    # Clear context
    ContextVariableFlow.clear_context()
    print(f"After clear: {ContextVariableFlow.get_context_summary()}")