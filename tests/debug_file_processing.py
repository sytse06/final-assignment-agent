# debug_file_processing.py
# Quick test script to debug file processing

def test_file_info_extraction():
    """Test file info extraction with sample task"""
    from agent_logic import extract_file_info_from_task_id
    
    # Test with clean task_id
    task_id = "9318445f-fe6a-4e1b-acbf-c68228c9906a"
    file_info = extract_file_info_from_task_id(task_id)
    
    print(f"Task ID: {task_id}")
    print(f"File Info: {file_info}")
    
    # Test with filename contamination  
    dirty_task_id = "9318445f-fe6a-4e1b-acbf-c68228c9906a.png"
    cleaned_info = extract_file_info_from_task_id(dirty_task_id)
    
    print(f"Dirty Task ID: {dirty_task_id}")
    print(f"Cleaned Info: {cleaned_info}")

def test_tool_configuration():
    """Test tool configuration independently - FIXED VERSION"""
    from tools import GetAttachmentTool
    
    tool = GetAttachmentTool()
    
    # INSPECT: Check what methods the tool actually has
    print("🔍 GetAttachmentTool available methods:")
    methods = [method for method in dir(tool) if not method.startswith('_')]
    for method in methods:
        print(f"   - {method}")
    
    # FIXED: Check for different possible method names
    task_id = "9318445f-fe6a-4e1b-acbf-c68228c9906a"
    
    if hasattr(tool, 'attachment_for'):
        print(f"✅ Using attachment_for method")
        tool.attachment_for(task_id)
    elif hasattr(tool, 'task_id'):
        print(f"✅ Setting task_id directly")
        tool.task_id = task_id
    elif hasattr(tool, 'configure_for_task'):
        print(f"✅ Using configure_for_task method") 
        tool.configure_for_task(task_id)
    else:
        print(f"⚠️  No known configuration method found")
        print(f"Available methods: {methods}")
        return None
    
    # Test with correct syntax
    print("\n🧪 Testing LOCAL_FILE_PATH format:")
    try:
        result = tool.forward(fmt="LOCAL_FILE_PATH")
        print(f"Path result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n🧪 Testing TEXT format:")
    try:
        text_result = tool.forward(fmt="TEXT")
        print(f"Text result: {text_result[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    return result

def test_state_creation():
    """Test GAIAState creation with file info"""
    from agent_logic import GAIAState, extract_file_info_from_task_id
    
    task_id = "9318445f-fe6a-4e1b-acbf-c68228c9906a"
    question = "Test question with image"
    
    # Extract file info
    file_info = extract_file_info_from_task_id(task_id)
    
    # Create state
    state = GAIAState(
        task_id=task_id,
        question=question,
        file_name=file_info.get("file_name", ""),
        file_path=file_info.get("file_path", ""),
        has_file=file_info.get("has_file", False),
        steps=[]
    )
    
    print("Created state:")
    for key, value in state.items():
        print(f"  {key}: {value}")

def test_content_retriever_safe():
    """Test ContentRetrieverTool with error handling"""
    from tools import ContentRetrieverTool
    
    print("🧪 Testing ContentRetrieverTool error handling:")
    tool = ContentRetrieverTool()
    
    # Test with invalid input (should return error string, not crash)
    result = tool.forward("nonexistent_file.pdf", "test query")
    
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    print(f"Has .strip(): {hasattr(result, 'strip')}")
    
    return result

def test_patched_tool():
    # Apply patch
    from tools import GetAttachmentTool
    
    def attachment_for(self, task_id: str):
        self.task_id = task_id
        print(f"🔗 Configured for task: {task_id}")
    
    GetAttachmentTool.attachment_for = attachment_for
    if not hasattr(GetAttachmentTool, 'task_id'):
        GetAttachmentTool.task_id = None
    
    # Test it
    tool = GetAttachmentTool()
    tool.attachment_for("9318445f-fe6a-4e1b-acbf-c68228c9906a")
    result = tool.forward(fmt="LOCAL_FILE_PATH")
    print(f"Patched result: {result}")

if __name__ == "__main__":
    print("🧪 Testing File Processing Components")
    print("=" * 50)
    
    print("\n1. Testing file info extraction:")
    test_file_info_extraction()
    
    print("\n2. Testing tool configuration:")
    try:
        test_tool_configuration()
    except Exception as e:
        print(f"❌ Tool configuration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Testing state creation:")
    test_state_creation()
    
    print("\n4. Testing ContentRetrieverTool error handling:")
    test_content_retriever_safe()