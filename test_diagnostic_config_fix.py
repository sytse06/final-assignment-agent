# extended_diagnostic_interface.py - Check both agent_testing.py AND agent_interface.py

import sys
import traceback
import inspect
import importlib

def inspect_agent_interface_functions():
    """Inspect agent_interface.py for GAIAConfig handling issues"""
    
    print("\n🔍 INSPECTING AGENT_INTERFACE MODULE")
    print("=" * 40)
    
    try:
        import agent_interface
        
        # Get all callable functions
        functions = [name for name in dir(agent_interface) 
                    if callable(getattr(agent_interface, name)) and not name.startswith('_')]
        
        print(f"Functions found: {len(functions)}")
        for func in sorted(functions):
            print(f"  - {func}")
        
        problems_found = []
        
        # Check create_gaia_agent specifically - this is likely the culprit
        if hasattr(agent_interface, 'create_gaia_agent'):
            print(f"\n🔍 Inspecting create_gaia_agent source:")
            try:
                source = inspect.getsource(agent_interface.create_gaia_agent)
                
                # Look for problematic patterns
                if "isinstance(config_overrides, dict)" in source:
                    print("⚠️  POTENTIAL ISSUE: create_gaia_agent checks isinstance(config_overrides, dict)")
                    
                    # Check if it properly handles GAIAConfig objects
                    if "isinstance(config_overrides, GAIAConfig)" not in source:
                        print("❌ PROBLEM: create_gaia_agent doesn't handle GAIAConfig objects properly")
                        problems_found.append("create_gaia_agent missing GAIAConfig handling")
                
                if ".get(" in source:
                    print("❌ PROBLEM: create_gaia_agent uses .get() method")
                    problems_found.append("create_gaia_agent uses .get()")
                
                if "config_overrides.items()" in source:
                    print("⚠️  POTENTIAL ISSUE: create_gaia_agent calls .items() - assumes dict")
                    problems_found.append("create_gaia_agent assumes dict with .items()")
                
                # Check the type annotation
                if "Union[Dict, str]" in source:
                    print("⚠️  TYPE ISSUE: create_gaia_agent annotated as Union[Dict, str] but receives GAIAConfig")
                    problems_found.append("create_gaia_agent type annotation excludes GAIAConfig")
                    
                print("✅ create_gaia_agent source inspection complete")
                    
            except Exception as e:
                print(f"   Could not inspect create_gaia_agent source: {e}")
        else:
            print("❌ create_gaia_agent not found!")
            problems_found.append("create_gaia_agent missing")
        
        # Check config functions for correct return types
        config_functions = ['get_groq_config', 'get_google_config', 'get_openrouter_config', 'get_ollama_config']
        
        print(f"\n🔍 Checking config function return type annotations:")
        for func_name in config_functions:
            if hasattr(agent_interface, func_name):
                try:
                    func = getattr(agent_interface, func_name)
                    source = inspect.getsource(func)
                    
                    if "-> Dict" in source:
                        print(f"❌ {func_name}: Returns -> Dict (should be -> GAIAConfig)")
                        problems_found.append(f"{func_name} annotated as -> Dict")
                    elif "-> GAIAConfig" in source:
                        print(f"✅ {func_name}: Returns -> GAIAConfig (correct)")
                    else:
                        print(f"⚠️  {func_name}: No return type annotation")
                        
                except Exception as e:
                    print(f"   Could not inspect {func_name}: {e}")
            else:
                print(f"❌ {func_name}: Not found")
        
        return problems_found
        
    except Exception as e:
        print(f"❌ Could not inspect agent_interface: {e}")
        traceback.print_exc()
        return ["inspection_failed"]

def test_create_gaia_agent_with_gaiaconfig():
    """Test calling create_gaia_agent with a GAIAConfig object (the failing scenario)"""
    
    print("\n🎯 TESTING create_gaia_agent WITH GAIAConfig OBJECT")
    print("=" * 50)
    
    try:
        from agent_interface import create_gaia_agent, get_ollama_config
        from agent_logic import GAIAConfig
        
        # Step 1: Get a GAIAConfig object
        print("Step 1: Getting GAIAConfig object...")
        gaia_config = get_ollama_config()
        print(f"✅ Got GAIAConfig: {type(gaia_config)}")
        
        # Step 2: Call create_gaia_agent with GAIAConfig object
        print("Step 2: Calling create_gaia_agent(gaia_config)...")
        print("This is what GAIATestExecutor does and where it likely fails!")
        
        # This is the exact call that fails
        agent = create_gaia_agent(gaia_config)
        
        print("✅ SUCCESS: create_gaia_agent works with GAIAConfig objects!")
        return True, None
        
    except AttributeError as e:
        if "'GAIAConfig' object has no attribute" in str(e):
            print("❌ REPRODUCED THE ERROR!")
            print(f"Error: {e}")
            
            tb = traceback.format_exc()
            
            analysis = {
                "error_found": True,
                "error_message": str(e),
                "traceback": tb,
                "location": "agent_interface.create_gaia_agent"
            }
            
            print("\n🔍 Error Analysis:")
            if ".items()" in str(e):
                print("❌ Root cause: create_gaia_agent calls .items() on GAIAConfig")
                analysis["root_cause"] = "gaiaconfig_items_call"
            elif ".get(" in str(e):
                print("❌ Root cause: create_gaia_agent uses .get() on GAIAConfig")
                analysis["root_cause"] = "gaiaconfig_get_call"
            else:
                print("❌ Root cause: Unknown attribute access on GAIAConfig")
                analysis["root_cause"] = "unknown_attribute_access"
                
            print(f"\n📋 Full traceback:")
            print(tb)
            
            return False, analysis
        else:
            print(f"❌ Different AttributeError: {e}")
            return False, {"error_found": True, "error_message": str(e), "error_type": "other_attribute_error"}
            
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False, {"error_found": True, "error_message": str(e), "error_type": type(e).__name__}

def test_gaiatestexecutor_flow():
    """Test the exact flow that GAIATestExecutor follows"""
    
    print("\n🔄 TESTING GAIATESTEXECUTOR FLOW")
    print("=" * 35)
    
    try:
        # Step 1: get_agent_config_by_name (from agent_testing)
        print("Step 1: get_agent_config_by_name('ollama')")
        from agent_testing import get_agent_config_by_name
        gaia_config = get_agent_config_by_name("ollama")
        print(f"✅ Got: {type(gaia_config)}")
        
        # Step 2: create_gaia_agent (from agent_interface) 
        print("Step 2: create_gaia_agent(gaia_config)")
        from agent_interface import create_gaia_agent
        agent = create_gaia_agent(gaia_config)
        print(f"✅ Created: {type(agent)}")
        
        # Step 3: Full GAIATestExecutor
        print("Step 3: GAIATestExecutor('ollama')")
        from agent_testing import GAIATestExecutor
        executor = GAIATestExecutor("ollama")
        print(f"✅ Executor: {type(executor)}")
        
        return True, None
        
    except Exception as e:
        print(f"❌ Flow failed at: {e}")
        
        # Analyze which step failed
        step_analysis = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        if "get_agent_config_by_name" in traceback.format_exc():
            step_analysis["failing_step"] = "step_1_get_config"
        elif "create_gaia_agent" in traceback.format_exc():
            step_analysis["failing_step"] = "step_2_create_agent"
        elif "GAIATestExecutor" in traceback.format_exc():
            step_analysis["failing_step"] = "step_3_executor"
        else:
            step_analysis["failing_step"] = "unknown"
        
        print(f"📋 Full traceback:")
        print(traceback.format_exc())
        
        return False, step_analysis

def generate_interface_fix_recommendations(interface_problems, test_results):
    """Generate specific fix recommendations for agent_interface.py"""
    
    print("\n🔧 AGENT_INTERFACE.PY FIX RECOMMENDATIONS")
    print("=" * 45)
    
    if not interface_problems and not test_results:
        print("✅ No problems detected in agent_interface.py!")
        return
    
    recommendations = []
    
    # Analyze interface problems
    if "create_gaia_agent missing GAIAConfig handling" in interface_problems:
        recommendations.append(
            "1. FIX create_gaia_agent() to handle GAIAConfig objects\n"
            "   Add: elif isinstance(config_overrides, GAIAConfig): config = config_overrides"
        )
    
    if "create_gaia_agent assumes dict with .items()" in interface_problems:
        recommendations.append(
            "2. FIX create_gaia_agent() .items() call\n"
            "   Only call .items() when config_overrides is actually a dict"
        )
    
    if "create_gaia_agent type annotation excludes GAIAConfig" in interface_problems:
        recommendations.append(
            "3. FIX create_gaia_agent() type annotation\n"
            "   Change: Union[Dict, str] -> Union[Dict, str, GAIAConfig]"
        )
    
    if any("annotated as -> Dict" in p for p in interface_problems):
        recommendations.append(
            "4. FIX config function return type annotations\n"
            "   Change all: -> Dict to -> GAIAConfig"
        )
    
    # Analyze test results
    if test_results and test_results.get("root_cause") == "gaiaconfig_items_call":
        recommendations.append(
            "5. CRITICAL: Fix .items() call on GAIAConfig in create_gaia_agent\n"
            "   GAIAConfig objects don't have .items() method"
        )
    
    if test_results and test_results.get("root_cause") == "gaiaconfig_get_call":
        recommendations.append(
            "6. CRITICAL: Fix .get() call on GAIAConfig in create_gaia_agent\n"
            "   GAIAConfig objects don't have .get() method"
        )
    
    # Print recommendations
    for rec in recommendations:
        print(rec)
    
    if not recommendations:
        print("No specific recommendations - check the detailed error output above.")

def main():
    """Run complete extended diagnostic including agent_interface.py"""
    
    print("🧪 EXTENDED GAIACONFIG DIAGNOSTIC")
    print("=" * 55)
    print("Checking BOTH agent_testing.py AND agent_interface.py")
    print()
    
    # Test 1: Basic imports
    print("=" * 55)
    print("TEST 1: Basic Imports")
    try:
        from agent_interface import get_ollama_config, get_groq_config, create_gaia_agent
        from agent_logic import GAIAConfig
        import agent_testing
        print("✅ All imports successful")
        imports_ok = True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        imports_ok = False
    
    if not imports_ok:
        print("\n❌ Cannot proceed - basic imports failed")
        return
    
    # Test 2: Inspect agent_interface
    print("=" * 55)
    print("TEST 2: Agent Interface Inspection")
    interface_problems = inspect_agent_interface_functions()
    
    # Test 3: Test create_gaia_agent with GAIAConfig
    print("=" * 55)
    print("TEST 3: create_gaia_agent with GAIAConfig")
    create_agent_ok, create_agent_error = test_create_gaia_agent_with_gaiaconfig()
    
    # Test 4: Test full GAIATestExecutor flow
    print("=" * 55)
    print("TEST 4: Full GAIATestExecutor Flow")
    flow_ok, flow_error = test_gaiatestexecutor_flow()
    
    # Summary
    print("\n" + "=" * 55)
    print("🎯 EXTENDED DIAGNOSTIC SUMMARY")
    print("=" * 30)
    
    print(f"Basic imports: {'✅' if imports_ok else '❌'}")
    print(f"Interface problems: {len(interface_problems) if interface_problems else 0}")
    print(f"create_gaia_agent test: {'✅' if create_agent_ok else '❌'}")
    print(f"Full flow test: {'✅' if flow_ok else '❌'}")
    
    # Detailed analysis
    if interface_problems:
        print(f"\n⚠️  Interface problems detected:")
        for problem in interface_problems:
            print(f"  - {problem}")
    
    if create_agent_error:
        print(f"\n❌ create_gaia_agent error:")
        print(f"  Error: {create_agent_error.get('error_message', 'Unknown')}")
        print(f"  Root cause: {create_agent_error.get('root_cause', 'Unknown')}")
    
    if flow_error:
        print(f"\n❌ Flow error:")
        print(f"  Failing step: {flow_error.get('failing_step', 'Unknown')}")
        print(f"  Error: {flow_error.get('error', 'Unknown')}")
    
    # Generate fix recommendations
    generate_interface_fix_recommendations(interface_problems, create_agent_error or flow_error)
    
    # Final verdict
    print(f"\n🎯 FINAL VERDICT")
    print("=" * 20)
    
    if flow_ok:
        print("✅ ALL TESTS PASSED - Your GAIATestExecutor should work!")
    elif create_agent_ok:
        print("⚠️  create_gaia_agent works but GAIATestExecutor flow fails")
    else:
        print("❌ create_gaia_agent fails - this is likely the root issue")
        print("   The problem is in agent_interface.py, not agent_testing.py")
        print("   Focus on fixing create_gaia_agent() function")

if __name__ == "__main__":
    main()