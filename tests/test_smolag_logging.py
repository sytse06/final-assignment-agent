# test_smolag_logging.py - Create this file to test the integration

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent_logic import GAIAAgent, GAIAConfig
from agent_logging import AgentLoggingSetup, validate_logging_compatibility

def test_smolag_logging_integration():
    """Test the enhanced SmolagAgent logging integration"""
    
    print("🧪 Testing SmolagAgent Logging Integration")
    print("=" * 60)
    
    # 1. Test logging compatibility
    print("\n📋 Step 1: Validating logging compatibility...")
    validation = validate_logging_compatibility()
    
    for check, status in validation.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {check}: {status}")
    
    if not validation.get("testing_framework_compatible", False):
        print("❌ Logging compatibility failed - check agent_logging.py setup")
        return False
    
    # 2. Test agent creation with enhanced logging
    print("\n🤖 Step 2: Creating GAIA agent with enhanced logging...")
    
    try:
        config = GAIAConfig(
            model_provider="openrouter",
            model_name="google/gemini-2.5-flash", 
            enable_csv_logging=True,
            debug_mode=True
        )
        
        agent = GAIAAgent(config)
        print("✅ GAIA agent created successfully")
        
        # Check if smolag logger was created
        if hasattr(agent.logging, 'smolag_logger'):
            print(f"✅ SmolagAgent logger available: {agent.logging.smolag_logger.log_file}")
        else:
            print("⚠️ SmolagAgent logger not found")
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False
    
    # 3. Test simple question processing
    print("\n📝 Step 3: Testing question processing with logging...")
    
    try:
        test_question = "What is 15 + 27? Show your calculation."
        result = agent.process_question(test_question, task_id="test_smolag_001")
        
        print(f"✅ Question processed successfully")
        print(f"   Answer: {result['final_answer']}")
        print(f"   SmolagAgent executions: {result.get('smolag_executions', 0)}")
        print(f"   Total SmolagAgent steps: {result.get('smolag_steps', 0)}")
        print(f"   Agents used: {result.get('agents_used', [])}")
        
        # Check if logs were created
        log_files = agent.logging.current_log_files
        print(f"\n📁 Log files created:")
        for log_type, file_path in log_files.items():
            exists = "✅" if os.path.exists(file_path) else "❌"
            print(f"   {exists} {log_type}: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Question processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_bridge_functionality():
    """Test ContextBridge enhanced functionality"""
    
    print("\n🌉 Testing ContextBridge Enhanced Functionality")
    print("-" * 50)
    
    from agent_logic import ContextBridge
    
    # Test basic tracking
    ContextBridge.start_task_execution("test_context_001")
    ContextBridge.track_operation("Testing operation tracking")
    
    # Test metrics
    metrics = ContextBridge.get_execution_metrics()
    print(f"✅ Execution metrics: {metrics}")
    
    # Test SmolagAgent summary
    summary = ContextBridge.get_smolag_summary()
    print(f"✅ SmolagAgent summary: {summary}")
    
    # Clear tracking
    final_metrics = ContextBridge.clear_tracking()
    print(f"✅ Final metrics: {final_metrics}")
    
    return True

def validate_log_file_formats():
    """Validate that log files are created in the expected format"""
    
    print("\n📄 Validating Log File Formats")
    print("-" * 40)
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("⚠️ No logs directory found - run a test first")
        return False
    
    # Check for SmolagAgent log files
    smolag_files = list(logs_dir.glob("gaia_smolagents_steps_*.md"))
    
    if smolag_files:
        latest_file = max(smolag_files, key=lambda f: f.stat().st_mtime)
        print(f"✅ Found SmolagAgent log: {latest_file}")
        
        # Validate file content
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "# SmolagAgent Detailed Execution Log" in content:
                print("✅ File header format correct")
            else:
                print("❌ File header format incorrect")
                
            if "## " in content:  # Should have agent execution sections
                print("✅ Agent execution sections found")
            else:
                print("⚠️ No agent execution sections found")
                
            print(f"📊 File size: {len(content)} characters")
            
        except Exception as e:
            print(f"❌ Error reading log file: {e}")
            return False
    else:
        print("⚠️ No SmolagAgent log files found")
        return False
    
    return True

def run_comprehensive_test():
    """Run comprehensive test of the integration"""
    
    print("🚀 COMPREHENSIVE SMOLAGENT LOGGING TEST")
    print("=" * 80)
    
    tests = [
        ("Logging Integration", test_smolag_logging_integration),
        ("ContextBridge Functionality", test_context_bridge_functionality),
        ("Log File Formats", validate_log_file_formats)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*20} TEST SUMMARY {'='*20}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name}: {'PASSED' if result else 'FAILED'}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SmolagAgent logging integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)