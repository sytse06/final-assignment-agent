# agent_context.py - FIXED VERSION
# Context Variables Infrastructure for LangGraph ↔ SmolagAgents Bridging

from contextvars import ContextVar
from typing import Dict, Any, Optional, List
import time
import uuid

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
    Enhanced call with automatic context fallback and error handling.
    
    Args:
        task_id: Optional task ID. If not provided, will try to get from context.
        
    Returns:
        Result from original GetAttachmentTool
    """
    # Try to get task_id from context if not provided
    if task_id is None:
        context_task_id = ContextVariableFlow.get_task_id()
        
        if context_task_id:
            task_id = context_task_id
            print(f"🔧 GetAttachmentTool auto-detected task_id: {task_id}")
        else:
            # More detailed error with context debugging
            available_context = ContextVariableFlow.get_context_summary()
            error_msg = (
                f"GetAttachmentTool called without task_id and none found in context.\n"
                f"Current context: {available_context}\n"
                f"Context active: {ContextVariableFlow.is_context_active()}"
            )
            print(f"❌ {error_msg}")
            return f"Error: {error_msg}"
    
    try:
        # Clean up task_id format - remove hyphens and convert to uppercase
        # The original tool seems to expect a specific format
        clean_task_id = task_id.replace('-', '').upper()
        
        print(f"🔧 Calling GetAttachmentTool with cleaned task_id: {clean_task_id}")
        
        # Try with cleaned task_id first
        result = self.original_tool(clean_task_id)
        print(f"✅ GetAttachmentTool successful with cleaned task_id")
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ GetAttachmentTool failed: {error_msg}")
        
        # If it's a format error, provide helpful guidance
        if "Invalid format" in error_msg:
            return (
                f"GetAttachmentTool format error: {error_msg}\n"
                f"Task ID provided: {task_id}\n"
                f"Cleaned task ID: {clean_task_id}\n"
                f"The tool expects: URL, DATA_URL, LOCAL_FILE_PATH, or TEXT format"
            )
        else:
            return f"GetAttachmentTool error: {error_msg}"

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


def create_context_aware_tools(original_tools: Dict[str, Any]) -> List[Tool]:
    """
    Create context-aware versions of tools that are proper SmolagAgents Tools.
    
    Args:
        original_tools: Dictionary of original tool instances
        
    Returns:
        List of context-aware tool instances (proper Tool objects)
    """
    context_aware_tools = []
    
    if not original_tools:
        print("⚠️  No original tools provided to create_context_aware_tools")
        return context_aware_tools
    
    for tool_name, tool_instance in original_tools.items():
        if tool_name == 'get_attachment' and tool_instance:
            try:
                context_tool = ContextAwareGetAttachmentTool(tool_instance)
                context_aware_tools.append(context_tool)
                print(f"✅ Created context-aware GetAttachmentTool")
            except Exception as e:
                print(f"❌ Failed to create context-aware GetAttachmentTool: {e}")
            
        elif tool_name == 'content_retriever' and tool_instance:
            try:
                context_tool = ContextAwareContentRetrieverTool(tool_instance)
                context_aware_tools.append(context_tool)
                print(f"✅ Created context-aware ContentRetrieverTool")
            except Exception as e:
                print(f"❌ Failed to create context-aware ContentRetrieverTool: {e}")
    
    print(f"🔧 Created {len(context_aware_tools)} context-aware tools total")
    return context_aware_tools

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