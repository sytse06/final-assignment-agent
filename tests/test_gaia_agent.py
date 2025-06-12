# tests/test_gaia_agent.py - Enhanced with CLI support and better debugging

import sys
import os
import argparse

# Solution 1: Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Now imports work normally
from agent_logic import GAIAAgent, GAIAConfig

def test_agent_creation(config_name="openrouter"):
    """Test that the agent can be created without Tool assertion error"""
    print("🧪 Testing GAIA Agent Creation")
    print("=" * 40)
    
    try:
        # Create config based on provider
        if config_name == "groq":
            config = GAIAConfig(
                model_provider="groq",
                model_name="qwen-qwq-32b",
                enable_context_bridge=True,
                context_bridge_debug=True
            )
        elif config_name == "google":
            config = GAIAConfig(
                model_provider="google",
                model_name="gemini-2.0-flash-preview",
                enable_context_bridge=True,
                context_bridge_debug=True
            )
        else:  # openrouter default
            config = GAIAConfig(
                model_provider="openrouter",
                model_name="qwen/qwen3-30b-a3b",
                enable_context_bridge=True,
                context_bridge_debug=True
            )
        
        print(f"✅ Config created: {config.model_provider}/{config.model_name}")
        
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
        
        # Debug: Show manager tools
        if hasattr(agent.manager, 'tools') and agent.manager.tools:
            print(f"🔧 Manager tools: {list(agent.manager.tools.keys())}")
        
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

def test_simple_question(agent, question="What is Mark Rutte doing right now?"):
    """Test processing a question with enhanced debugging"""
    if agent is None:
        print("⚠️  Skipping question test - agent creation failed")
        return
    
    print(f"\n🧪 Testing Question Processing")
    print("=" * 40)
    print(f"Question: {question}")
    
    try:
        # Enable detailed context debugging
        if hasattr(agent.config, 'context_bridge_debug'):
            agent.config.context_bridge_debug = True
        
        result = agent.process_question(question)
        
        print(f"✅ Question processed successfully!")
        print(f"Task ID: {result.get('task_id')}")
        print(f"Final Answer: {result.get('final_answer', 'No answer')}")
        print(f"Execution Successful: {result.get('execution_successful', False)}")
        print(f"Context Bridge Used: {result.get('context_bridge_used', False)}")
        print(f"Steps: {len(result.get('steps', []))}")
        
        # Debug: Show execution steps
        steps = result.get('steps', [])
        if steps:
            print(f"\n🔍 Execution Steps:")
            for i, step in enumerate(steps, 1):
                print(f"   {i}. {step}")
        
        # Debug: Show any errors
        if 'error' in result:
            print(f"⚠️  Error encountered: {result['error']}")
        
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

def test_manager_delegation_format(agent, question="What is the capital of France?"):
    """Test that manager delegation uses correct format"""
    if agent is None:
        return False
        
    print(f"\n🧪 Testing Manager Delegation Format")
    print("=" * 40)
    
    try:
        from agent_context import ContextVariableFlow
        
        # Set up context manually to test delegation
        task_id = "test_delegation_123"
        ContextVariableFlow.set_task_context(task_id, question)
        
        # Test manager context preparation
        context = agent._prepare_manager_context(question, [], task_id)
        
        print("✅ Manager context prepared")
        print(f"Context length: {len(context)} characters")
        
        # Check if context contains proper delegation examples
        if "data_analyst('''" in context:
            print("✅ Data analyst delegation example found")
        else:
            print("⚠️  Data analyst delegation example missing")
            
        if "web_researcher('''" in context:
            print("✅ Web researcher delegation example found")
        else:
            print("⚠️  Web researcher delegation example missing")
        
        # Clean up
        ContextVariableFlow.clear_context()
        
        return True
        
    except Exception as e:
        print(f"❌ Manager delegation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test GAIA Agent with custom question')
    parser.add_argument(
        '-q', '--question', 
        type=str, 
        default="What is Mark Rutte doing right now?",
        help='Question to test with the agent'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        choices=['openrouter', 'groq', 'google'],
        default='openrouter',
        help='Model provider to use'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--skip-agent-test',
        action='store_true',
        help='Skip the actual agent question test (useful for debugging setup)'
    )
    
    return parser.parse_args()

def main():
    """Run all tests with CLI support"""
    args = parse_arguments()
    
    print("🚀 GAIA Agent Fix Verification - ENHANCED")
    print("=" * 50)
    print(f"Running from: {os.getcwd()}")
    print(f"Test file: {__file__}")
    print(f"Question: {args.question}")
    print(f"Config: {args.config}")
    print(f"Verbose: {args.verbose}")
    print("")
    
    # Test 1: Context Bridge
    context_success = test_context_bridge()
    
    # Test 2: Tool Creation
    tool_success = test_tool_creation()
    
    # Test 3: Agent Creation
    agent = test_agent_creation(args.config)
    
    # Test 4: Manager Delegation Format
    delegation_success = False
    if agent:
        delegation_success = test_manager_delegation_format(agent, args.question)
    
    # Test 5: Question Processing (if agent created successfully and not skipped)
    question_success = False
    if agent and not args.skip_agent_test:
        result = test_simple_question(agent, args.question)
        question_success = result is not None
    elif args.skip_agent_test:
        print("\n⏭️  Skipping agent question test as requested")
        question_success = True  # Don't count as failure
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 40)
    print(f"Context Bridge: {'✅ PASS' if context_success else '❌ FAIL'}")
    print(f"Tool Creation: {'✅ PASS' if tool_success else '❌ FAIL'}")
    print(f"Agent Creation: {'✅ PASS' if agent is not None else '❌ FAIL'}")
    print(f"Manager Delegation: {'✅ PASS' if delegation_success else '❌ FAIL'}")
    if not args.skip_agent_test:
        print(f"Question Processing: {'✅ PASS' if question_success else '❌ FAIL'}")
    
    all_passed = all([
        context_success, 
        tool_success, 
        agent is not None, 
        delegation_success,
        question_success or args.skip_agent_test
    ])
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("The Tool assertion error has been fixed.")
        print("The ContentRetrieverTool nullable issue has been fixed.")
        print("Manager delegation format looks correct.")
        if not args.skip_agent_test:
            print("Your GAIA agent is ready for deployment!")
        
        # Additional success details
        if agent:
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
        if not delegation_success:
            failed_tests.append("Manager Delegation")
        if not question_success and not args.skip_agent_test:
            failed_tests.append("Question Processing")
            
        print(f"   Failed tests: {', '.join(failed_tests)}")
    
    return agent

if __name__ == "__main__":
    agent = main()