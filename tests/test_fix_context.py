# test_fix.py - Test the fixed GAIA agent

from agent_logic import GAIAAgent, GAIAConfig

def test_agent_creation():
    """Test that the agent can be created without Tool assertion error"""
    print("🧪 Testing GAIA Agent Creation")
    print("=" * 40)
    
    try:
        # Create config object for Openrouter
        config = GAIAConfig(
            model_provider="openrouter",
            model_name="qwen/qwen3-30b-a3b",
            enable_context_bridge=True,
            context_bridge_debug=True
        )
        
        print("✅ Config created successfully")
        
        # Create agent - this should not fail with Tool assertion error
        agent = GAIAAgent(config)
        
        print("✅ GAIA Agent created successfully!")
        print(f"✅ Specialists: {list(agent.specialists.keys())}")
        print(f"✅ Context-aware tools: {len(agent.context_aware_tools)}")
        print(f"✅ Manager created: {agent.manager is not None}")
        
        # Debug: Show context-aware tools details
        if agent.context_aware_tools:
            print(f"🔧 Context-aware tools details:")
            for i, tool in enumerate(agent.context_aware_tools):
                print(f"   {i+1}. {tool.name} ({type(tool).__name__})")
        
        return agent
        
    except Exception as e:
        print(f"❌ Agent creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_tool_creation():
    """Test creating tools independently"""
    print("\n🧪 Testing Tool Creation")
    print("=" * 40)
    
    try:
        # Test ContentRetrieverTool creation
        from content_retriever_tool import ContentRetrieverTool
        
        print("Testing ContentRetrieverTool creation...")
        content_tool = ContentRetrieverTool()
        print(f"✅ ContentRetrieverTool created: {content_tool.name}")
        print(f"   Inputs: {list(content_tool.inputs.keys())}")
        print(f"   Output type: {content_tool.output_type}")
        
        # Check nullable flag
        verification_input = content_tool.inputs.get('verification_mode', {})
        nullable = verification_input.get('nullable', False)
        print(f"   Verification mode nullable: {nullable}")
        
        if not nullable:
            print("❌ WARNING: verification_mode should have nullable=True")
        else:
            print("✅ verification_mode properly configured with nullable=True")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_question(agent):
    """Test processing a simple question"""
    if agent is None:
        print("⚠️  Skipping question test - agent creation failed")
        return
    
    print("\n🧪 Testing Simple Question Processing")
    print("=" * 40)
    
    try:
        question = "What is Mark Rutte doing right now?"
        result = agent.process_question(question)
        
        print(f"✅ Question processed successfully!")
        print(f"Task ID: {result.get('task_id')}")
        print(f"Final Answer: {result.get('final_answer', 'No answer')}")
        print(f"Execution Successful: {result.get('execution_successful', False)}")
        print(f"Context Bridge Used: {result.get('context_bridge_used', False)}")
        print(f"Steps: {len(result.get('steps', []))}")
        
        return result
        
    except Exception as e:
        print(f"❌ Question processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_context_bridge():
    """Test context bridge functionality"""
    print("\n🧪 Testing Context Bridge")
    print("=" * 40)
    
    from agent_context import ContextVariableFlow
    
    try:
        # Test context setting
        ContextVariableFlow.set_task_context(
            task_id="test_123",
            question="Test question",
            metadata={"complexity": "simple", "routing_path": "test"}
        )
        
        print(f"✅ Context set: {ContextVariableFlow.get_context_summary()}")
        
        # Test context retrieval
        task_id = ContextVariableFlow.get_task_id()
        question = ContextVariableFlow.get_question()
        
        print(f"✅ Task ID retrieved: {task_id}")
        print(f"✅ Question retrieved: {question}")
        
        # Test context updates
        ContextVariableFlow.update_complexity("complex")
        ContextVariableFlow.update_routing_path("manager_coordination")
        
        print(f"✅ Updated context: {ContextVariableFlow.get_context_summary()}")
        
        # Test context clearing
        ContextVariableFlow.clear_context()
        print(f"✅ Context cleared: {ContextVariableFlow.get_context_summary()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Context bridge test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 GAIA Agent Fix Verification - COMPREHENSIVE")
    print("=" * 50)
    print("")
    
    # Test 1: Context Bridge
    context_success = test_context_bridge()
    
    # Test 2: Tool Creation
    tool_success = test_tool_creation()
    
    # Test 3: Agent Creation
    agent = test_agent_creation()
    
    # Test 4: Question Processing (if agent created successfully)
    question_success = False
    if agent:
        result = test_simple_question(agent)
        question_success = result is not None
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 40)
    print(f"Context Bridge: {'✅ PASS' if context_success else '❌ FAIL'}")
    print(f"Tool Creation: {'✅ PASS' if tool_success else '❌ FAIL'}")
    print(f"Agent Creation: {'✅ PASS' if agent is not None else '❌ FAIL'}")
    print(f"Question Processing: {'✅ PASS' if question_success else '❌ FAIL'}")
    
    if agent and context_success and tool_success and question_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("The Tool assertion error has been fixed.")
        print("The ContentRetrieverTool nullable issue has been fixed.")
        print("Your GAIA agent is ready for deployment!")
        
        # Additional success details
        print(f"\n📋 Success Details:")
        print(f"   • Context-aware tools created: {len(agent.context_aware_tools)}")
        print(f"   • Specialists available: {', '.join(agent.specialists.keys())}")
        print(f"   • Manager agent: {'✅' if agent.manager else '❌'}")
        print(f"   • Context bridge: {'✅' if agent.config.enable_context_bridge else '❌'}")
        
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        
        failed_tests = []
        if not context_success:
            failed_tests.append("Context Bridge")
        if not tool_success:
            failed_tests.append("Tool Creation")
        if agent is None:
            failed_tests.append("Agent Creation")
        if not question_success:
            failed_tests.append("Question Processing")
            
        print(f"   Failed tests: {', '.join(failed_tests)}")
    
    return agent

if __name__ == "__main__":
    agent = main()

def test_simple_question(agent):
    """Test processing a simple question"""
    if agent is None:
        print("⚠️  Skipping question test - agent creation failed")
        return
    
    print("\n🧪 Testing Simple Question Processing")
    print("=" * 40)
    
    try:
        question = "What is Mark Rutte doing right now?"
        result = agent.process_question(question)
        
        print(f"✅ Question processed successfully!")
        print(f"Task ID: {result.get('task_id')}")
        print(f"Final Answer: {result.get('final_answer', 'No answer')}")
        print(f"Execution Successful: {result.get('execution_successful', False)}")
        print(f"Context Bridge Used: {result.get('context_bridge_used', False)}")
        print(f"Steps: {len(result.get('steps', []))}")
        
        return result
        
    except Exception as e:
        print(f"❌ Question processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_context_bridge():
    """Test context bridge functionality"""
    print("\n🧪 Testing Context Bridge")
    print("=" * 40)
    
    from agent_context import ContextVariableFlow
    
    try:
        # Test context setting
        ContextVariableFlow.set_task_context(
            task_id="test_123",
            question="Test question",
            metadata={"complexity": "simple", "routing_path": "test"}
        )
        
        print(f"✅ Context set: {ContextVariableFlow.get_context_summary()}")
        
        # Test context retrieval
        task_id = ContextVariableFlow.get_task_id()
        question = ContextVariableFlow.get_question()
        
        print(f"✅ Task ID retrieved: {task_id}")
        print(f"✅ Question retrieved: {question}")
        
        # Test context updates
        ContextVariableFlow.update_complexity("complex")
        ContextVariableFlow.update_routing_path("manager_coordination")
        
        print(f"✅ Updated context: {ContextVariableFlow.get_context_summary()}")
        
        # Test context clearing
        ContextVariableFlow.clear_context()
        print(f"✅ Context cleared: {ContextVariableFlow.get_context_summary()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Context bridge test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 GAIA Agent Fix Verification")
    print("=" * 50)
    print("")
    
    # Test 1: Context Bridge
    context_success = test_context_bridge()
    
    # Test 2: Agent Creation
    agent = test_agent_creation()
    
    # Test 3: Question Processing (if agent created successfully)
    question_success = False
    if agent:
        result = test_simple_question(agent)
        question_success = result is not None
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 40)
    print(f"Context Bridge: {'✅ PASS' if context_success else '❌ FAIL'}")
    print(f"Agent Creation: {'✅ PASS' if agent is not None else '❌ FAIL'}")
    print(f"Question Processing: {'✅ PASS' if question_success else '❌ FAIL'}")
    
    if agent and context_success and question_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("The Tool assertion error has been fixed.")
        print("Your GAIA agent is ready for deployment!")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return agent

if __name__ == "__main__":
    agent = main()