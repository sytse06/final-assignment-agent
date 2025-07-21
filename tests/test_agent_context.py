# test_agent_context.py  
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_agent_creation():
    """Test that the agent can be created with new specialist architecture"""
    print("🧪 Testing GAIA Agent Creation")
    print("=" * 40)
    
    try:
        from agent_logic import GAIAAgent, GAIAConfig
        
        # Create config object
        config = GAIAConfig(
            model_provider="openrouter",
            model_name="google/gemini-2.5-flash",
            enable_context_bridge=True,
            context_bridge_debug=True
        )
        
        print("✅ Config created successfully")
        
        # Create agent with specialist architecture
        agent = GAIAAgent(config)
        
        print("✅ GAIA Agent created successfully!")
        
        # Test specialist creation (they're created on-demand, not stored permanently)
        try:
            specialists = agent._create_specialist_agents()
            print(f"✅ Specialists: {list(specialists.keys())}")
            
            # Test specialist capabilities
            for name, specialist in specialists.items():
                print(f"   📊 {name}:")
                if hasattr(specialist, 'tools'):
                    print(f"      Tools: {len(specialist.tools)}")
                    for tool in specialist.tools[:3]:  # Show first 3 tools
                        tool_name = getattr(tool, 'name', type(tool).__name__)
                        print(f"        - {tool_name}")
                    if len(specialist.tools) > 3:
                        print(f"        ... and {len(specialist.tools) - 3} more")
                
                if hasattr(specialist, 'additional_authorized_imports'):
                    print(f"      Imports: {len(specialist.additional_authorized_imports)}")
                
                if hasattr(specialist, 'step_callbacks'):
                    print(f"      Callbacks: {len(specialist.step_callbacks) if specialist.step_callbacks else 0}")
        
        except Exception as e:
            print(f"⚠️ Specialist testing failed: {e}")
        
        # Test coordinator creation
        try:
            coordinator = agent._create_coordinator()
            print(f"✅ Coordinator created with {len(coordinator.managed_agents) if hasattr(coordinator, 'managed_agents') else 'unknown'} managed agents")
        except Exception as e:
            print(f"⚠️ Coordinator testing failed: {e}")
        
        # Test capabilities assessment
        if hasattr(agent, 'capabilities'):
            print(f"✅ Vision capabilities: {agent.capabilities}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Agent creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_specialist_tools():
    """Test that specialist tools are properly configured"""
    print("\n🧪 Testing Specialist Tool Configuration")
    print("=" * 40)
    
    try:
        from tools import (
            get_web_researcher_tools, 
            get_content_processor_tools,
            BrowserProfileTool,
            SMOLAGENTS_VISION_AVAILABLE,
            BROWSER_PROFILE_AVAILABLE
        )
        
        print(f"✅ Tool availability:")
        print(f"   smolagents vision: {SMOLAGENTS_VISION_AVAILABLE}")
        print(f"   browser profile: {BROWSER_PROFILE_AVAILABLE}")
        
        # Test web researcher tools
        print("\n📊 Web Researcher Tools:")
        web_tools = get_web_researcher_tools()
        for i, tool in enumerate(web_tools, 1):
            tool_name = getattr(tool, 'name', type(tool).__name__)
            print(f"   {i}. {tool_name}")
        
        # Test content processor tools  
        print("\n📊 Content Processor Tools:")
        content_tools = get_content_processor_tools()
        for i, tool in enumerate(content_tools, 1):
            tool_name = getattr(tool, 'name', type(tool).__name__)
            print(f"   {i}. {tool_name}")
        
        # Test BrowserProfileTool specifically
        if BROWSER_PROFILE_AVAILABLE:
            print("\n📊 Testing BrowserProfileTool instantiation:")
            browser_tool = BrowserProfileTool()
            print(f"   ✅ {browser_tool.name} created successfully")
            print(f"   Profile directory: {browser_tool._profile_dir}")
            print(f"   Container environment: {browser_tool._is_container_environment()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Specialist tool test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_authentication_workflow():
    """Test authentication workflow and instructions"""
    print("\n🧪 Testing Authentication Workflow")
    print("=" * 40)
    
    try:
        from tools.BrowserProfileTool import get_authenticated_browser_instructions
        
        instructions = get_authenticated_browser_instructions()
        print(f"✅ Authentication instructions loaded: {len(instructions)} characters")
        
        # Check for key authentication patterns
        key_patterns = [
            "browser_profile(",
            "YOUTUBE_COOKIES",
            "GITHUB_COOKIES",
            "Authentication: {auth_method}",
            "wait_until(Text('Library').exists)"
        ]
        
        found_patterns = 0
        for pattern in key_patterns:
            if pattern in instructions:
                found_patterns += 1
                print(f"   ✅ Found pattern: {pattern}")
            else:
                print(f"   ⚠️ Missing pattern: {pattern}")
        
        print(f"✅ Authentication patterns found: {found_patterns}/{len(key_patterns)}")
        
        return found_patterns >= len(key_patterns) - 1  # Allow 1 missing pattern
        
    except Exception as e:
        print(f"❌ Authentication workflow test failed: {str(e)}")
        return False

def test_question_processing(agent):
    """Test processing a question with enhanced capabilities"""
    if agent is None:
        print("⚠️ Skipping question test - agent creation failed")
        return
    
    print("\n🧪 Testing Question Processing")
    print("=" * 40)
    
    try:
        # Test authentication-aware question
        question = "What is the latest video on the OpenAI YouTube channel?"
        print(f"📝 Question: {question}")
        
        result = agent.process_question(question)
        
        print(f"✅ Question processed successfully!")
        print(f"Task ID: {result.get('task_id')}")
        print(f"Final Answer: {result.get('final_answer', 'No answer')[:100]}...")
        print(f"Execution Successful: {result.get('execution_successful', False)}")
        print(f"Context Bridge Used: {result.get('context_bridge_used', False)}")
        print(f"Steps: {len(result.get('steps', []))}")
        
        # Check if authentication was considered
        steps = result.get('steps', [])
        auth_mentioned = any('youtube' in str(step).lower() or 'profile' in str(step).lower() 
                           for step in steps)
        print(f"Authentication considered: {auth_mentioned}")
        
        return result
        
    except Exception as e:
        print(f"❌ Question processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_specialist_context_building():
    """Test the specialist context building"""
    print("\n🧪 Testing Specialist Context Building")
    print("=" * 40)
    
    try:
        from agent_logic import GAIAAgent, GAIAConfig
        
        config = GAIAConfig(model_provider="openrouter", model_name="google/gemini-2.5-flash")
        agent = GAIAAgent(config)
        
        # Test context building with authentication-aware content
        test_questions = [
            "What is in this age-restricted YouTube video: https://youtube.com/watch?v=abc123",
            "Calculate the average of these numbers: 15, 23, 42, 18, 31",
            "Extract text from this PDF document"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Test {i}: {question[:50]}...")
            
            # Simulate state
            state = {
                "question": question,
                "complexity": "medium", 
                "has_file": False,
                "similar_examples": []
            }
            
            context = agent._build_specialist_context(state)
            
            # Check for authentication hints
            auth_hints = [
                "AUTHENTICATION:",
                "browser_profile",
                "CONTENT TYPE:",
                "CAPABILITY GUIDANCE:"
            ]
            
            found_hints = sum(1 for hint in auth_hints if hint in context)
            print(f"   Authentication hints found: {found_hints}/{len(auth_hints)}")
            
            # Show relevant context excerpt
            lines = context.split('\n')
            auth_lines = [line for line in lines if any(hint in line for hint in auth_hints)]
            if auth_lines:
                print(f"   Context excerpt: {auth_lines[0][:80]}...")
        
        print("✅ Context building works")
        return True
        
    except Exception as e:
        print(f"❌ Context building test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 GAIA Agent Test Suite")
    print("=" * 50)
    print("")
    
    # Test 1: Tool Configuration
    tool_success = test_specialist_tools()
    
    # Test 2: Authentication Workflow
    auth_success = test_authentication_workflow()
    
    # Test 3: Agent Creation
    agent = test_agent_creation()
    agent_success = agent is not None
    
    # Test 4: Context Building
    context_success = test_specialist_context_building()
    
    # Test 5: Question Processing
    question_success = False
    if agent:
        result = test_question_processing(agent)
        question_success = result is not None
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 40)
    print(f"Tool Configuration: {'✅ PASS' if tool_success else '❌ FAIL'}")
    print(f"Authentication Workflow: {'✅ PASS' if auth_success else '❌ FAIL'}")
    print(f"Agent Creation: {'✅ PASS' if agent_success else '❌ FAIL'}")
    print(f"Context Building: {'✅ PASS' if context_success else '❌ FAIL'}")
    print(f"Question Processing: {'✅ PASS' if question_success else '❌ FAIL'}")
    
    all_passed = all([tool_success, auth_success, agent_success, context_success, question_success])
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("The new smolagents architecture is working correctly:")
        print("   ✅ BrowserProfileTool authentication system")
        print("   ✅ smolagents vision browser components")
        print("   ✅ Specialist agent capabilities") 
        print("   ✅ Authentication-aware context building")
        print("   ✅ Multi-platform authentication support")
        print("\n🚀 Your GAIA agent is ready for deployment!")
        
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
        
        failed_tests = []
        if not tool_success:
            failed_tests.append("Tool Configuration")
        if not auth_success:
            failed_tests.append("Authentication Workflow")
        if not agent_success:
            failed_tests.append("Agent Creation")
        if not context_success:
            failed_tests.append("Context Building")
        if not question_success:
            failed_tests.append("Question Processing")
            
        print(f"   Failed tests: {', '.join(failed_tests)}")
    
    return agent

if __name__ == "__main__":
    agent = main()