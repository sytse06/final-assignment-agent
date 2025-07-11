# agent_interface.py - STREAMLINED VERSION
# Focus on real added value: presets, testing, and comparison

from typing import Dict, List
from agent_logic import GAIAAgent, GAIAConfig

# ============================================================================
# CONFIGURATION PRESETS (✅ Real Value)
# ============================================================================

def get_openrouter_config() -> GAIAConfig:
    """Openrouter gemini-2.5-flash configuration"""
    return GAIAConfig(
        model_provider="openrouter",
        model_name="google/gemini-2.5-flash",
        temperature=0.1,
        enable_smart_routing=True,
        enable_csv_logging=True
    )

def get_google_config() -> GAIAConfig:
    """Google Gemini configuration"""
    return GAIAConfig(
        model_provider="google", 
        model_name="gemini-2.0-flash-preview",
        temperature=0.3,
        enable_smart_routing=True,
        enable_csv_logging=True
    )

def get_groq_config() -> GAIAConfig:
    """Groq configuration"""
    return GAIAConfig(
        model_provider="groq",
        model_name="qwen/qwen3-32b",
        temperature=0.1,
        enable_smart_routing=True,
        enable_csv_logging=True
    )
    
def get_ollama_config(model_name: str = "devstral-16k") -> GAIAConfig:
    """Get Ollama configuration with enhanced defaults"""
    return GAIAConfig(
        model_provider="ollama",
        model_name=model_name,
        temperature=0.1,
        max_agent_steps=15,
        enable_smart_routing=True,
        skip_rag_for_simple=True,
        enable_csv_logging=True,
        debug_mode=True
    )

def get_performance_config() -> GAIAConfig:
    """Speed-optimized configuration"""
    return GAIAConfig(
        model_provider="groq",
        model_name="qwen/qwen3-32b", 
        temperature=0.1,
        max_agent_steps=12,
        enable_smart_routing=True,
        debug_mode=False
    )

def get_accuracy_config() -> GAIAConfig:
    """Accuracy-optimized configuration"""
    return GAIAConfig(
        model_provider="google",
        model_name="gemini-2.0-flash-preview",
        temperature=0.2,
        max_agent_steps=20,
        skip_rag_for_simple=False,
        rag_examples_count=5
    )

# ============================================================================
# AGENT FACTORY (✅ Real Value - String Presets)
# ============================================================================

def create_gaia_agent(preset_or_config=None) -> GAIAAgent:
    """Create agent from preset name or custom config"""
    
    if isinstance(preset_or_config, str):
        presets = {
            "groq": get_groq_config,
            "qwen": get_groq_config,  
            "google": get_google_config,
            "gemini": get_google_config,
            "openrouter": get_openrouter_config,
            "performance": get_performance_config,
            "accuracy": get_accuracy_config
        }
        
        if preset_or_config.lower() in presets:
            config = presets[preset_or_config.lower()]()
        else:
            print(f"⚠️  Unknown preset '{preset_or_config}', using groq")
            config = get_groq_config()
    
    elif isinstance(preset_or_config, dict):
        # Apply overrides to default config
        config = get_groq_config()  # Start with default
        for key, value in preset_or_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    elif preset_or_config is None:
        config = get_groq_config()  # Default
    
    else:
        config = preset_or_config  # Already a GAIAConfig
    
    return GAIAAgent(config)

# ============================================================================
# TESTING UTILITIES (✅ Real Value - Multi-Agent Comparison)
# ============================================================================

def compare_configs(config_names: List[str], test_question: str = "What is 25% of 400?") -> Dict:
    """Compare different configurations on same question"""
    
    print(f"⚖️  Comparing: {', '.join(config_names)}")
    print(f"Question: {test_question}")
    
    results = {}
    
    for config_name in config_names:
        try:
            agent = create_gaia_agent(config_name)
            result = agent.process_question(test_question)
            
            results[config_name] = {
                "answer": result.get("final_answer", "No answer"),
                "success": result.get("execution_successful", False),
                "steps": len(result.get("steps", [])),
                "time": result.get("performance_metrics", {}).get("total_execution_time", 0)
            }
            
            status = "✅" if results[config_name]["success"] else "❌"
            print(f"{status} {config_name}: {results[config_name]['answer']}")
            
        except Exception as e:
            results[config_name] = {"error": str(e)}
            print(f"❌ {config_name}: Failed")
    
    return results

def test_routing(config_name: str = "groq") -> Dict:
    """Test smart routing behavior"""
    
    test_cases = [
        {"question": "What is 2 + 2?", "expected": "simple"},
        {"question": "What is the current population of Tokyo?", "expected": "complex"},
        {"question": "Calculate 15% of 200", "expected": "simple"},
        {"question": "Analyze the attached spreadsheet", "expected": "complex"}
    ]
    
    print(f"🔀 Testing routing with {config_name}")
    
    agent = create_gaia_agent(config_name)
    results = []
    
    for test_case in test_cases:
        result = agent.process_question(test_case["question"])
        
        # Get routing info (simplified)
        complexity = result.get("complexity", "unknown")
        actual_routing = "simple" if complexity == "simple" else "complex"
        
        routing_correct = actual_routing == test_case["expected"]
        
        results.append({
            "question": test_case["question"],
            "expected": test_case["expected"], 
            "actual": actual_routing,
            "correct": routing_correct
        })
        
        status = "✅" if routing_correct else "❌"
        print(f"{status} {test_case['question'][:40]}... → {actual_routing}")
    
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"📊 Routing accuracy: {accuracy:.1%}")
    
    return {"accuracy": accuracy, "results": results}

def quick_test(question: str = "What is 25% of 400?", config: str = "groq") -> bool:
    """Quick test - returns True/False for success"""
    try:
        agent = create_gaia_agent(config)
        result = agent.process_question(question)
        
        success = result.get("execution_successful", False)
        answer = result.get("final_answer", "No answer")
        
        print(f"{'✅' if success else '❌'} {config}: {answer}")
        return success
        
    except Exception as e:
        print(f"❌ {config}: Failed - {e}")
        return False

# ============================================================================
# BATCH TESTING (✅ Real Value - Multi-Question Testing)
# ============================================================================

def test_questions(questions: List[str], config: str = "groq") -> Dict:
    """Test multiple questions with one configuration"""
    
    print(f"🧪 Testing {len(questions)} questions with {config}")
    
    agent = create_gaia_agent(config)
    results = []
    
    for i, question in enumerate(questions, 1):
        try:
            result = agent.process_question(question)
            
            success = result.get("execution_successful", False)
            answer = result.get("final_answer", "No answer")
            
            results.append({
                "question": question,
                "answer": answer,
                "success": success
            })
            
            status = "✅" if success else "❌"
            print(f"{status} {i}/{len(questions)}: {answer}")
            
        except Exception as e:
            results.append({
                "question": question,
                "error": str(e),
                "success": False
            })
            print(f"❌ {i}/{len(questions)}: Failed")
    
    success_rate = sum(r["success"] for r in results) / len(results)
    print(f"📊 Success rate: {success_rate:.1%}")
    
    return {
        "config": config,
        "success_rate": success_rate,
        "results": results
    }

# ============================================================================
# CONVENIENCE TESTING SUITES (✅ Real Value - Predefined Test Sets)
# ============================================================================

def test_math_suite(config: str = "groq"):
    """Test mathematical reasoning"""
    math_questions = [
        "What is 25% of 400?",
        "Calculate 15 * 23", 
        "What is the square root of 144?",
        "If I buy 3 items at $12 each, what's the total?"
    ]
    return test_questions(math_questions, config)

def test_knowledge_suite(config: str = "groq"):
    """Test factual knowledge"""
    knowledge_questions = [
        "What are the primary colors?",
        "What is the capital of France?",
        "How many days are in a leap year?",
        "What is the freezing point of water in Celsius?"
    ]
    return test_questions(knowledge_questions, config)

def run_comprehensive_test(config: str = "groq"):
    """Run all test suites"""
    print(f"🚀 Comprehensive test with {config}")
    print("=" * 40)
    
    math_results = test_math_suite(config)
    knowledge_results = test_knowledge_suite(config)
    routing_results = test_routing(config)
    
    overall_success = (math_results["success_rate"] + knowledge_results["success_rate"]) / 2
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"Math: {math_results['success_rate']:.1%}")
    print(f"Knowledge: {knowledge_results['success_rate']:.1%}")
    print(f"Routing: {routing_results['accuracy']:.1%}")
    print(f"Overall: {overall_success:.1%}")
    
    return {
        "config": config,
        "math": math_results,
        "knowledge": knowledge_results,
        "routing": routing_results,
        "overall_success": overall_success
    }

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("🚀 GAIA Agent Interface - Streamlined")
    print("=" * 40)
    print("✅ Key functions:")
    print("   create_gaia_agent('groq')           # Create with preset")
    print("   quick_test('What is 2+2?', 'groq') # Quick test")
    print("   compare_configs(['groq', 'google']) # Compare configs")
    print("   test_routing('performance')         # Test routing")
    print("   run_comprehensive_test('accuracy')  # Full test suite")
    print("")
    print("💡 Example usage:")
    print("   agent = create_gaia_agent('performance')")
    print("   result = agent.process_question('What is 25% of 400?')")
    print("   print(result['final_answer'])")