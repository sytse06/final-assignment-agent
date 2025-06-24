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
    Passes task_id and context from LangGraph state to SmolagAgent tools.
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
        context_tool = create_context_wrapper(base_tool)
        context_aware_tools.append(context_tool)
        print(f"✅ Created context-aware {tool_name} with parameter filtering")
    
    return context_aware_tools

def create_context_wrapper(base_tool):
    """Create SmolagAgents-compliant context-aware tool wrapper"""
    
    class ContextAwareTool(Tool):
        """SmolagAgents-compliant Tool wrapper with context bridge integration"""
        
        def __init__(self, base_tool):
            self.base_tool = base_tool
            
            # Set required Tool attributes BEFORE super().__init__()
            self.name = getattr(base_tool, 'name', 'unknown_tool')
            self.description = getattr(base_tool, 'description', 'Context-aware tool wrapper')
            self.inputs = getattr(base_tool, 'inputs', {
                "query": {"type": "string", "description": "Input query"}
            })
            self.output_type = getattr(base_tool, 'output_type', "string")
            
            # Ensure inputs have proper nullable flags for SmolagAgents validation
            self._ensure_inputs_have_nullable_flags()
            
            # Initialize parent Tool class (validation occurs here)
            super().__init__()
            
            # Create compliant forward method
            self._create_compliant_forward_method()
        
        def _ensure_inputs_have_nullable_flags(self):
            """Ensure all inputs have proper nullable flags for SmolagAgents validation"""
            updated_inputs = {}
            
            for key, input_schema in self.inputs.items():
                updated_schema = input_schema.copy()
                
                # Add nullable flag if missing (assume nullable for compatibility)
                if 'nullable' not in updated_schema:
                    updated_schema['nullable'] = True
                    
                updated_inputs[key] = updated_schema
            
            self.inputs = updated_inputs
        
        def _create_compliant_forward_method(self):
            """Create forward method with exact signature matching self.inputs"""
            input_keys = list(self.inputs.keys())
            
            if not input_keys:
                def forward_impl(self) -> Any:
                    return self._execute_with_context(**{})
            else:
                def forward_impl(self, **kwargs) -> Any:
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in input_keys}
                    return self._execute_with_context(**filtered_kwargs)
            
            # Set signature that matches SmolagAgents validation requirements
            if input_keys:
                import inspect
                parameters = [inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
                
                for key in input_keys:
                    # Check if input is nullable in schema
                    input_schema = self.inputs.get(key, {})
                    is_nullable = input_schema.get('nullable', False)
                    
                    if is_nullable:
                        # If input is nullable, use default=None
                        param = inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY, default=None)
                    else:
                        # If input is required, no default value
                        param = inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY)
                    
                    parameters.append(param)
                
                # Set signature on function before binding
                forward_impl.__signature__ = inspect.Signature(parameters)
            
            # Bind method to instance
            import types
            self.forward = types.MethodType(forward_impl, self)
        
        def _execute_with_context(self, **kwargs):
            """Execute base tool with context bridge and parameter filtering"""
            
            # Get context from bridge
            try:
                context = ContextVariableFlow.get_task_context()
                task_id = context.get('task_id')
                question = context.get('question')
            except Exception:
                task_id = None
                question = None
            
            # Set task_id for GetAttachmentTool
            if task_id and hasattr(self.base_tool, 'attachment_for'):
                try:
                    self.base_tool.attachment_tool(task_id)
                    print(f"🔧 Set task_id {task_id} for {self.name}")
                except Exception as e:
                    print(f"⚠️ Could not set task_id for {self.name}: {e}")
            
            # Filter parameters to base tool's signature
            valid_params = self._get_base_tool_parameters()
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params and v is not None}
            
            # Add context query if needed
            if 'query' in valid_params and question and 'query' not in filtered_kwargs:
                filtered_kwargs['query'] = question
            
            # Execute with fallback strategy
            try:
                return self.base_tool.forward(**filtered_kwargs)
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    return self._fallback_execution(**filtered_kwargs)
                else:
                    raise
        
        def _get_base_tool_parameters(self):
            """Get base tool's accepted parameters"""
            try:
                import inspect
                sig = inspect.signature(self.base_tool.forward)
                return {name for name in sig.parameters.keys() if name != 'self'}
            except Exception:
                # Fallback based on tool type
                tool_name = self.name.lower()
                if 'search' in tool_name or 'web' in tool_name:
                    return {'query', 'url'}
                elif 'attachment' in tool_name or 'file' in tool_name:
                    return {'query', 'task_id'}
                elif 'content' in tool_name:
                    return {'query', 'url', 'format'}
                else:
                    return {'query', 'question', 'input'}
        
        def _fallback_execution(self, **kwargs):
            """Progressive parameter fallback"""
            param_order = ['query', 'question', 'input', 'url', 'task_id']
            
            for num_params in range(len(kwargs), 0, -1):
                test_params = {}
                for param in param_order:
                    if param in kwargs and len(test_params) < num_params:
                        test_params[param] = kwargs[param]
                
                try:
                    return self.base_tool.forward(**test_params)
                except TypeError:
                    continue
            
            # Final attempt with no parameters
            try:
                return self.base_tool.forward()
            except Exception as e:
                raise RuntimeError(f"All parameter combinations failed for {self.name}: {e}")
    
    return ContextAwareTool(base_tool)

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