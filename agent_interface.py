# gaia_interface.py
# Public interface and convenience functions for GAIA agent

from typing import Dict, List, Union
from agent_logic import GAIAAgent, GAIAConfig

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_gaia_agent(config_overrides: Union[Dict, str] = None) -> GAIAAgent:
    """
    Factory function for creating GAIA agent
    
    Args:
        config_overrides: Can be either:
            - Dict: Configuration overrides
            - str: Configuration preset name ('groq', 'google', 'performance', etc.)
            - None: Use default configuration
    """
    config = GAIAConfig()
    
    # Handle string preset names
    if isinstance(config_overrides, str):
        preset_name = config_overrides.lower()
        
        if preset_name in ["groq", "qwen", "qwen3_32b"]:
            config_overrides = get_groq_config()
        elif preset_name in ["google", "gemini"]:
            config_overrides = get_google_config()
        elif preset_name in ["openrouter", "or"]:
            config_overrides = get_openrouter_config()
        elif preset_name in ["ollama", "local"]:
            config_overrides = get_ollama_config()
        elif preset_name == "performance":
            config_overrides = get_performance_config()
        elif preset_name == "accuracy":
            config_overrides = get_accuracy_config()
        elif preset_name == "debug":
            config_overrides = get_debug_config()
        else:
            print(f"⚠️  Unknown preset '{preset_name}', using default groq config")
            config_overrides = get_groq_config()
    
    # Apply configuration overrides
    if config_overrides and isinstance(config_overrides, dict):
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"⚠️  Unknown config key: {key}")
    
    return GAIAAgent(config)

def run_single_question_enhanced(agent: 'GAIAAgent', question: str, task_id: str = None) -> Dict:
    """
    Enhanced single question runner with comprehensive tracking
    
    This is the recommended method for testing and evaluation as it provides
    detailed execution metrics and strategy information.
    """
    import time
    import uuid
    
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    start_time = time.time()
    
    try:
        # Run the core agent method
        result = agent.run_single_question(question, task_id)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Extract strategy information from the workflow state
        strategy_used = "unknown"
        selected_agent = "unknown"
        
        # Try to determine strategy from various result fields
        if "complexity" in result:
            complexity = result["complexity"]
            if complexity in ["simple", "direct"]:
                strategy_used = "one_shot"
            elif complexity in ["complex", "manager"]:
                strategy_used = "manager"
            else:
                strategy_used = complexity
        
        # Try to extract agent information from steps
        steps = result.get("steps", [])
        if steps:
            for step in steps:
                if isinstance(step, dict) and "agent" in step:
                    selected_agent = step["agent"]
                    break
                elif isinstance(step, str) and "specialist" in step.lower():
                    # Extract agent name from step description
                    for agent_name in ["data_analyst", "web_researcher", "document_processor", "general_assistant"]:
                        if agent_name in step.lower():
                            selected_agent = agent_name
                            break
        
        # Enhance the result with additional tracking information
        enhanced_result = {
            **result,  # Include all original fields
            "execution_time": execution_time,
            "selected_strategy": strategy_used,
            "selected_agent": selected_agent,
            "enhanced_tracking": True,  # Flag to indicate this was enhanced
            "timestamp": time.time()
        }
        
        return enhanced_result
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Enhanced execution failed: {str(e)}"
        
        return {
            "task_id": task_id,
            "question": question,
            "final_answer": "Execution failed",
            "error": error_msg,
            "execution_time": execution_time,
            "selected_strategy": "error",
            "selected_agent": "none",
            "enhanced_tracking": True,
            "timestamp": time.time(),
            "steps": []
        }

def quick_test(question: str = "Calculate 15% of 1000", 
               config_overrides: Union[Dict, str] = None) -> Dict:
    """Quick test function with custom config"""
    try:
        agent = create_gaia_agent(config_overrides)
        result = agent.run_single_question(question)
        
        print(f"\n🎯 TEST RESULT")
        print("=" * 30)
        print(f"Question: {question}")
        print(f"Final Answer: {result.get('final_answer', 'No answer')}")
        print(f"Steps: {len(result.get('steps', []))}")
        print(f"Success: {'✅' if 'error' not in result else '❌'}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return {"error": str(e)}

def test_specialist_coordination():
    """Test that manager properly delegates to specialists"""
    questions = [
        "Calculate 25% of 400",           # Should use data_analyst
        "What is current population of Tokyo?", # Should use web_researcher  
        "Process attached Excel file"     # Should use document_processor
    ]
    return test_multiple_questions(questions)

def test_multiple_questions(questions: List[str], 
                           config_overrides: Union[Dict, str] = None) -> List[Dict]:
    """Test multiple questions with enhanced tracking"""
    agent = create_gaia_agent(config_overrides)
    results = []
    
    print(f"🧪 Testing {len(questions)} questions...")
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}: {question[:50]}...")
        try:
            result = run_single_question_enhanced(agent, question)
            results.append(result)
            
            # Show enhanced information
            print(f"✅ Answer: {result.get('final_answer', 'No answer')}")
            print(f"🔄 Strategy: {result.get('selected_strategy', 'unknown')}")
            print(f"🤖 Agent: {result.get('selected_agent', 'unknown')}")
            print(f"⏱️ Time: {result.get('execution_time', 0):.2f}s")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({"question": question, "error": str(e)})
    
    return results

# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

def get_groq_config(model_name: str = "qwen-qwq-32b") -> Dict:
    """Get Groq configuration with smart routing"""
    return {
        "model_provider": "groq",
        "model_name": model_name,
        "temperature": 0.3,
        "max_agent_steps": 15,
        "enable_smart_routing": True,
        "skip_rag_for_simple": True
    }

def get_openrouter_config(model_name: str = "qwen/qwen-2.5-coder-32b-instruct:free") -> Dict:
    """Get OpenRouter configuration with smart routing"""
    return {
        "model_provider": "openrouter", 
        "model_name": model_name,
        "temperature": 0.3,
        "max_agent_steps": 20,
        "enable_smart_routing": True,
        "skip_rag_for_simple": True
    }

def get_ollama_config(model_name: str = "qwen2.5-coder:32b", num_ctx: int = 32768) -> Dict:
    """Get Ollama configuration with smart routing"""
    return {
        "model_provider": "ollama",
        "model_name": model_name,
        "temperature": 0.3,
        "num_ctx": num_ctx,
        "max_agent_steps": 15,
        "enable_smart_routing": True,
        "skip_rag_for_simple": True
    }

def get_google_config(model_name: str = "gemini-2.0-flash-preview") -> Dict:
    """Get Google configuration with smart routing"""
    return {
        "model_provider": "google",
        "model_name": model_name,
        "temperature": 0.4,
        "max_agent_steps": 15,
        "enable_smart_routing": True,
        "skip_rag_for_simple": True
    }

def get_debug_config(enable_logging: bool = True) -> Dict:
    """Get debug configuration"""
    return {
        "debug_mode": enable_logging,
        "enable_csv_logging": enable_logging,
        "max_agent_steps": 10,  # Shorter for debugging
        "enable_smart_routing": False,  # Disable for predictable debugging
        "skip_rag_for_simple": False
    }

def get_performance_config() -> Dict:
    """Get performance-optimized configuration"""
    return {
        "model_provider": "groq",
        "model_name": "qwen-qwq-32b",
        "enable_smart_routing": True,
        "skip_rag_for_simple": True,
        "max_agent_steps": 12,  # Optimized step count
        "planning_interval": 2,  # More frequent planning
        "temperature": 0.2  # Lower for consistency
    }

def get_accuracy_config() -> Dict:
    """Get accuracy-focused configuration"""
    return {
        "model_provider": "groq",
        "model_name": "qwen-qwq-32b",
        "enable_smart_routing": False,  # Always use full pipeline
        "skip_rag_for_simple": False,   # Always use RAG
        "max_agent_steps": 25,  # More steps for thoroughness
        "planning_interval": 5,  # Less frequent planning
        "rag_examples_count": 5,  # More examples
        "temperature": 0.4  # Higher for creativity
    }

# ============================================================================
# TESTING HELPERS
# ============================================================================

def test_basic_math():
    """Test basic math capabilities"""
    questions = [
        "What is 25% of 400?",
        "Calculate 15 * 23",
        "What is the square root of 144?"
    ]
    return test_multiple_questions(questions)

def test_web_search():
    """Test web search capabilities"""
    questions = [
        "What is the current population of Tokyo?",
        "When was the Eiffel Tower built?",
        "Who won the 2024 FIFA World Cup?"
    ]
    return test_multiple_questions(questions)

def test_reasoning():
    """Test reasoning capabilities"""
    questions = [
        "If a train travels 60 km/h for 2.5 hours, how far does it go?",
        "Explain why the sky is blue in simple terms",
        "What are the primary colors?"
    ]
    return test_multiple_questions(questions)

def test_routing_behavior():
    """Test smart routing behavior"""
    print("🧪 Testing routing behavior...")
    
    # Test simple questions (should use one-shot)
    simple_config = get_groq_config()
    simple_questions = [
        "What is 25% of 400?",
        "What are the primary colors?",
        "Calculate 15 + 23"
    ]
    
    print("\n📊 Testing simple questions with smart routing:")
    simple_results = test_multiple_questions(simple_questions, simple_config)
    
    # Test complex questions (should use manager)
    complex_questions = [
        "What is the current population of Tokyo and how has it changed since 2020?",
        "Analyze the attached spreadsheet and find the average",
        "Search for recent AI developments and summarize key trends"
    ]
    
    print("\n🧠 Testing complex questions with smart routing:")
    complex_results = test_multiple_questions(complex_questions, simple_config)
    
    return {
        "simple_results": simple_results,
        "complex_results": complex_results
    }

# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

def compare_agent_configs(config_names: List[str], test_questions: List[str] = None) -> Dict:
    """Compare performance across different configurations"""
    
    if test_questions is None:
        test_questions = [
            "What is 25% of 400?",
            "Calculate 15 * 23",
            "What are the primary colors?",
            "What is the current year?"
        ]
    
    results = {}
    
    for config_name in config_names:
        print(f"\n🧪 Testing {config_name} configuration...")
        
        try:
            agent = create_gaia_agent(config_name)
            config_results = []
            
            for question in test_questions:
                try:
                    result = agent.run_single_question(question)
                    config_results.append({
                        "question": question,
                        "answer": result.get("final_answer", ""),
                        "success": "error" not in result,
                        "steps": len(result.get("steps", [])),
                        "complexity": result.get("complexity", "unknown")
                    })
                except Exception as e:
                    config_results.append({
                        "question": question,
                        "error": str(e),
                        "success": False
                    })
            
            # Calculate summary stats
            total_questions = len(config_results)
            successful = sum(1 for r in config_results if r.get("success", False))
            avg_steps = sum(r.get("steps", 0) for r in config_results) / total_questions
            
            results[config_name] = {
                "results": config_results,
                "summary": {
                    "success_rate": successful / total_questions,
                    "avg_steps": avg_steps,
                    "total_questions": total_questions
                }
            }
            
            print(f"✅ {config_name}: {successful}/{total_questions} success ({successful/total_questions:.1%})")
            
        except Exception as e:
            print(f"❌ {config_name} failed: {e}")
            results[config_name] = {"error": str(e)}
    
    return results

# ============================================================================
# MAIN TESTING INTERFACE
# ============================================================================

def run_comprehensive_test(config_name: str = "groq") -> Dict:
    """Run comprehensive test suite"""
    
    print(f"🚀 Running comprehensive test with {config_name} config")
    print("=" * 50)
    
    results = {
        "config": config_name,
        "basic_math": [],
        "web_search": [],
        "reasoning": [],
        "routing_test": None,
        "summary": {}
    }
    
    try:
        # Test categories
        print("\n📊 Testing basic math...")
        results["basic_math"] = test_multiple_questions([
            "What is 25% of 400?",
            "Calculate 15 * 23"
        ], config_name)
        
        print("\n🌐 Testing web search...")
        results["web_search"] = test_multiple_questions([
            "What is the current year?",
            "When was Python programming language created?"
        ], config_name)
        
        print("\n🧠 Testing reasoning...")
        results["reasoning"] = test_multiple_questions([
            "What are the primary colors?",
            "If a train travels 60 km/h for 2 hours, how far does it go?"
        ], config_name)
        
        # Test routing if available
        try:
            agent = create_gaia_agent(config_name)
            if hasattr(agent.config, 'enable_smart_routing') and agent.config.enable_smart_routing:
                print("\n🔀 Testing smart routing...")
                results["routing_test"] = test_routing_behavior()
        except:
            pass
        
        # Calculate summary
        all_results = (results["basic_math"] + 
                      results["web_search"] + 
                      results["reasoning"])
        
        total_tests = len(all_results)
        successful_tests = sum(1 for r in all_results if "error" not in r)
        
        results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "config_used": config_name
        }
        
        print(f"\n📈 SUMMARY")
        print("=" * 20)
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Success rate: {results['summary']['success_rate']:.1%}")
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        results["error"] = str(e)
    
    return results

# ============================================================================
# USAGE EXAMPLES AND HELP
# ============================================================================

def show_usage_examples():
    """Show usage examples for different scenarios"""
    print("🚀 GAIA Agent Usage Examples")
    print("=" * 40)
    
    print("\n1️⃣ BASIC USAGE:")
    print("   agent = create_gaia_agent()  # Default configuration")
    print("   agent = create_gaia_agent('groq')  # Use preset")
    print("   agent = create_gaia_agent({'model_provider': 'google'})  # Custom config")
    
    print("\n2️⃣ AVAILABLE PRESETS:")
    print("   - 'groq' or 'qwen3_32b': Fast Groq models")
    print("   - 'google' or 'gemini': Google's Gemini models")
    print("   - 'openrouter': Free OpenRouter models")
    print("   - 'ollama': Local Ollama models")
    print("   - 'performance': Optimized for speed")
    print("   - 'accuracy': Optimized for accuracy")
    print("   - 'debug': Debug mode with logging")
    
    print("\n3️⃣ TESTING FUNCTIONS:")
    print("   quick_test('What is 25% of 400?', 'groq')")
    print("   run_comprehensive_test('performance')")
    print("   compare_agent_configs(['groq', 'google'])")
    
    print("\n4️⃣ EXAMPLE CORRECT USAGE:")
    print("   # Your original code had this error:")
    print("   # agent = create_gaia_agent('qwen3_32b')  # ❌ This works now!")
    print("   ")
    print("   # All these are now valid:")
    print("   agent = create_gaia_agent('qwen3_32b')  # ✅ Preset name")
    print("   agent = create_gaia_agent('groq')       # ✅ Preset name")
    print("   agent = create_gaia_agent({'temperature': 0.5})  # ✅ Dict config")
    print("   agent = create_gaia_agent()             # ✅ Default config")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI interface"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            question = sys.argv[2] if len(sys.argv) > 2 else "Calculate 15% of 1000"
            config = sys.argv[3] if len(sys.argv) > 3 else "groq"
            quick_test(question, config)
        elif command == "comprehensive":
            config_name = sys.argv[2] if len(sys.argv) > 2 else "groq"
            run_comprehensive_test(config_name)
        elif command == "compare":
            configs = sys.argv[2:] if len(sys.argv) > 2 else ["groq", "google"]
            compare_agent_configs(configs)
        elif command == "examples":
            show_usage_examples()
        elif command == "math":
            test_basic_math()
        elif command == "web":
            test_web_search()
        elif command == "reasoning":
            test_reasoning()
        elif command == "routing":
            test_routing_behavior()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: test, comprehensive, compare, examples, math, web, reasoning, routing")
    else:
        show_usage_examples()

if __name__ == "__main__":
    main()