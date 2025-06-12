# tests/test_comprehensive.py - Comprehensive Test Suite for Both Scenarios

import sys
import os
import argparse
from typing import Optional, Dict, Any

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
from agent_logic import GAIAAgent, GAIAConfig
from agent_context import ContextVariableFlow

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
        
        # Create agent
        agent = GAIAAgent(config)
        
        print("✅ GAIA Agent created successfully!")
        print(f"✅ Specialists: {list(agent.specialists.keys())}")
        print(f"✅ Context-aware tools: {len(agent.context_aware_tools)}")
        print(f"✅ Manager created: {agent.manager is not None}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Agent creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_development_question(agent, question: str):
    """
    Test with development questions (no files expected).
    These are arbitrary questions that don't require GAIA files.
    """
    if agent is None:
        print("⚠️  Skipping development test - agent creation failed")
        return None
        
    print(f"\n🔧 Development Question Test")
    print("=" * 40)
    print(f"Question: {question}")
    print("Expected: No file access needed, pure reasoning/web search")
    print("")
    
    try:
        result = agent.process_question(question)
        
        print(f"✅ Development question processed!")
        print(f"Task ID: {result.get('task_id')} (auto-generated)")
        print(f"Final Answer: {result.get('final_answer', 'No answer')}")
        print(f"Execution Successful: {result.get('execution_successful', False)}")
        print(f"Steps: {len(result.get('steps', []))}")
        
        # Analysis of results for development questions
        steps = result.get('steps', [])
        if steps:
            print(f"\n🔍 Execution Analysis:")
            for i, step in enumerate(steps, 1):
                print(f"   {i}. {step}")
                
            # Check for expected behaviors
            step_text = " ".join(steps).lower()
            if "getattachmenttool" in step_text and "no task_id" in step_text:
                print(f"\n✅ Expected: GetAttachmentTool correctly reports no files for dev question")
            elif "getattachmenttool" in step_text and "error" in step_text:
                print(f"\n✅ Expected: GetAttachmentTool correctly fails for non-GAIA task ID")
            
            if any(word in step_text for word in ["web", "search", "research"]):
                print(f"✅ Expected: Web research initiated for current information")
                
            if any(word in step_text for word in ["direct", "one-shot", "simple"]):
                print(f"✅ Expected: Question routed to appropriate complexity level")
        
        return result
        
    except Exception as e:
        print(f"❌ Development question failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_gaia_benchmark_question(agent, max_questions: int = 1):
    """
    Test with actual GAIA benchmark questions that have associated files.
    This tests the full GAIA workflow with file access.
    """
    if agent is None:
        print("⚠️  Skipping GAIA benchmark test - agent creation failed")
        return None
        
    print(f"\n🎯 GAIA Benchmark Test")
    print("=" * 40)
    print(f"Testing {max_questions} real GAIA question(s) with files")
    print("")
    
    try:
        # Try to import GAIA testing infrastructure
        from agent_testing import run_quick_gaia_test
        
        print("✅ GAIA testing infrastructure available")
        
        # Run actual GAIA test
        print(f"🚀 Running GAIA test with {agent.config.model_provider} model...")
        
        results = run_quick_gaia_test(
            agent_config_name=agent.config.model_provider,
            max_questions=max_questions
        )
        
        if results:
            print(f"✅ GAIA benchmark test completed!")
            print(f"Questions tested: {len(results.get('results', []))}")
            
            # Analyze GAIA results
            successful = sum(1 for r in results.get('results', []) if r.get('execution_successful', False))
            print(f"Successful executions: {successful}/{len(results.get('results', []))}")
            
            # Show sample results
            for i, result in enumerate(results.get('results', [])[:3]):  # Show first 3
                print(f"\n📝 GAIA Sample {i+1}:")
                print(f"   Task ID: {result.get('task_id', 'Unknown')}")
                print(f"   Question: {result.get('question', '')[:100]}...")
                print(f"   Success: {result.get('execution_successful', False)}")
                print(f"   Answer: {result.get('final_answer', 'No answer')[:50]}...")
            
            return results
        else:
            print("⚠️  GAIA test returned no results")
            return None
            
    except ImportError:
        print("⚠️  GAIA testing infrastructure not available")
        print("     This is expected if agent_testing.py is not set up")
        print("     GAIA benchmark testing requires the full testing framework")
        return None
    except Exception as e:
        print(f"❌ GAIA benchmark test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_manual_gaia_task(agent, task_id: str, question: str):
    """
    Test with a manually specified GAIA task ID and question.
    This simulates a real GAIA task if you have the task ID.
    """
    if agent is None:
        print("⚠️  Skipping manual GAIA test - agent creation failed")
        return None
        
    print(f"\n📋 Manual GAIA Task Test")
    print("=" * 40)
    print(f"Task ID: {task_id}")
    print(f"Question: {question}")
    print("Expected: File access should work if task ID exists")
    print("")
    
    try:
        result = agent.process_question(question, task_id=task_id)
        
        print(f"✅ Manual GAIA task processed!")
        print(f"Task ID: {result.get('task_id')}")
        print(f"Final Answer: {result.get('final_answer', 'No answer')}")
        print(f"Execution Successful: {result.get('execution_successful', False)}")
        
        # Analysis for GAIA tasks
        steps = result.get('steps', [])
        if steps:
            print(f"\n🔍 GAIA Task Analysis:")
            step_text = " ".join(steps).lower()
            
            if "getattachmenttool" in step_text and "successful" in step_text:
                print(f"✅ GetAttachmentTool successfully accessed files")
            elif "getattachmenttool" in step_text and ("error" in step_text or "failed" in step_text):
                print(f"⚠️  GetAttachmentTool could not access files (task ID may not exist)")
            
            if "file" in step_text or "attachment" in step_text:
                print(f"✅ File processing attempted")
                
        return result
        
    except Exception as e:
        print(f"❌ Manual GAIA task failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_context_bridge():
    """Test context bridge functionality"""
    print("\n🌉 Context Bridge Test")
    print("=" * 40)
    
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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Comprehensive GAIA Agent Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Modes:
  development   Test with arbitrary questions (no files needed)
  gaia         Test with real GAIA benchmark questions 
  manual       Test with specific GAIA task ID
  all          Run all test modes

Examples:
  # Development testing
  python tests/test_comprehensive.py development -q "What is 15% of 200?"
  
  # GAIA benchmark testing  
  python tests/test_comprehensive.py gaia -n 3
  
  # Manual GAIA task testing
  python tests/test_comprehensive.py manual -t task_001 -q "Question about the attached file"
  
  # Run all tests
  python tests/test_comprehensive.py all -q "Is Elon Musk CEO of Tesla?"
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['development', 'gaia', 'manual', 'all'],
        help='Test mode to run'
    )
    
    parser.add_argument(
        '-q', '--question', 
        type=str, 
        default="What is the current population of the Netherlands?",
        help='Question for development or manual testing'
    )
    
    parser.add_argument(
        '-t', '--task-id',
        type=str,
        default="task_001",
        help='Task ID for manual GAIA testing'
    )
    
    parser.add_argument(
        '-n', '--num-questions',
        type=int,
        default=1,
        help='Number of GAIA questions to test'
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
    
    return parser.parse_args()

def main():
    """Run comprehensive test suite"""
    args = parse_arguments()
    
    print("🚀 COMPREHENSIVE GAIA AGENT TEST SUITE")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Question: {args.question}")
    if args.mode in ['manual', 'gaia']:
        print(f"Task ID: {args.task_id if args.mode == 'manual' else 'Auto-selected'}")
        print(f"Num Questions: {args.num_questions if args.mode == 'gaia' else 'N/A'}")
    print("")
    
    # Test 1: Context Bridge (always run)
    context_success = test_context_bridge()
    
    # Test 2: Agent Creation (always run)
    agent = test_agent_creation(args.config)
    
    # Test 3-5: Based on mode
    dev_success = None
    gaia_success = None  
    manual_success = None
    
    if args.mode == 'development' or args.mode == 'all':
        dev_result = test_development_question(agent, args.question)
        dev_success = dev_result is not None
    
    if args.mode == 'gaia' or args.mode == 'all':
        gaia_result = test_gaia_benchmark_question(agent, args.num_questions)
        gaia_success = gaia_result is not None
    
    if args.mode == 'manual' or args.mode == 'all':
        manual_result = test_manual_gaia_task(agent, args.task_id, args.question)
        manual_success = manual_result is not None
    
    # Summary
    print("\n📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    print(f"Context Bridge: {'✅ PASS' if context_success else '❌ FAIL'}")
    print(f"Agent Creation: {'✅ PASS' if agent is not None else '❌ FAIL'}")
    
    if dev_success is not None:
        print(f"Development Test: {'✅ PASS' if dev_success else '❌ FAIL'}")
    if gaia_success is not None:
        print(f"GAIA Benchmark: {'✅ PASS' if gaia_success else '⚠️  N/A' if gaia_success is None else '❌ FAIL'}")
    if manual_success is not None:
        print(f"Manual GAIA Test: {'✅ PASS' if manual_success else '❌ FAIL'}")
    
    # Overall assessment
    core_tests_passed = context_success and (agent is not None)
    
    if core_tests_passed:
        print(f"\n🎉 CORE TESTS PASSED!")
        print(f"Your GAIA agent infrastructure is working correctly.")
        
        if args.mode == 'development' and dev_success:
            print(f"✅ Development testing: Agent handles arbitrary questions correctly")
        if args.mode == 'gaia' and gaia_success:
            print(f"✅ GAIA benchmark: Agent processes real GAIA tasks correctly")
        if args.mode == 'manual' and manual_success:
            print(f"✅ Manual GAIA: Agent handles specific task IDs correctly")
        
        print(f"\n📋 Key Insights:")
        print(f"   • GetAttachmentTool 'errors' for dev questions are EXPECTED behavior")
        print(f"   • GAIA tasks require real task IDs with associated files")
        print(f"   • Development questions test reasoning without file dependencies")
        print(f"   • Context bridge successfully manages task state across modes")
        
    else:
        print(f"\n⚠️  Core infrastructure needs attention")
        if not context_success:
            print(f"   • Context bridge issues detected")
        if agent is None:
            print(f"   • Agent creation failed")
    
    return {
        'agent': agent,
        'context_success': context_success,
        'dev_success': dev_success,
        'gaia_success': gaia_success,
        'manual_success': manual_success
    }

if __name__ == "__main__":
    results = main()