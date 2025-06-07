# gaia_interface.py
# Public interface and convenience functions for GAIA agent

from typing import Dict, List
from agent_logic import GAIAAgent, GAIAConfig

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_gaia_agent(config_overrides: Dict = None) -> GAIAAgent:
    """Factory function for creating GAIA agent"""
    config = GAIAConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"⚠️  Unknown config key: {key}")
    
    return GAIAAgent(config)

def quick_test(question: str = "Calculate 15% of 1000", 
               config_overrides: Dict = None) -> Dict:
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
                           config_overrides: Dict = None) -> List[Dict]:
    """Test multiple questions with same agent"""
    agent = create_gaia_agent(config_overrides)
    results = []
    
    print(f"🧪 Testing {len(questions)} questions...")
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}: {question[:50]}...")
        try:
            result = agent.run_single_question(question)
            results.append(result)
            print(f"✅ Answer: {result.get('final_answer', 'No answer')}")
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({"question": question, "error": str(e)})
    
    return results

# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

def get_groq_config(model_name: str = "qwen-qwq-32b") -> Dict:
    """Get Groq configuration"""
    return {
        "model_provider": "groq",
        "model_name": model_name,
        "temperature": 0.3,
        "max_agent_steps": 15
    }

def get_openrouter_config(model_name: str = "qwen/qwen-2.5-coder-32b-instruct:free") -> Dict:
    """Get OpenRouter configuration"""
    return {
        "model_provider": "openrouter", 
        "model_name": model_name,
        "temperature": 0.3,
        "max_agent_steps": 20
    }

def get_ollama_config(model_name: str = "qwen3:14b", num_ctx: int = 32768) -> Dict:
    """Get Ollama configuration"""
    return {
        "model_provider": "ollama",
        "model_name": model_name,
        "temperature": 0.3,
        "num_ctx": num_ctx,
        "max_agent_steps": 15
    }

def get_google_config(model_name: str = "gemini-2.0-flash-preview") -> Dict:
    """Get Google configuration"""
    return {
        "model_provider": "google",
        "model_name": model_name,
        "temperature": 0.4,
        "max_agent_steps": 15
    }

def get_debug_config(enable_logging: bool = True) -> Dict:
    """Get debug configuration"""
    return {
        "debug_mode": enable_logging,
        "enable_csv_logging": enable_logging,
        "max_agent_steps": 10  # Shorter for debugging
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

# ============================================================================
# MAIN TESTING INTERFACE
# ============================================================================

def run_comprehensive_test(config_name: str = "groq") -> Dict:
    """Run comprehensive test suite"""
    
    # Get config
    if config_name == "groq":
        config = get_groq_config()
    elif config_name == "openrouter":
        config = get_openrouter_config()
    elif config_name == "google":
        config = get_google_config()
    else:
        config = {}
    
    print(f"🚀 Running comprehensive test with {config_name} config")
    print("=" * 50)
    
    results = {
        "config": config_name,
        "basic_math": [],
        "web_search": [],
        "reasoning": [],
        "summary": {}
    }
    
    try:
        # Test categories
        print("\n📊 Testing basic math...")
        results["basic_math"] = test_multiple_questions([
            "What is 25% of 400?",
            "Calculate 15 * 23"
        ], config)
        
        print("\n🌐 Testing web search...")
        results["web_search"] = test_multiple_questions([
            "What is the current year?",
            "When was Python programming language created?"
        ], config)
        
        print("\n🧠 Testing reasoning...")
        results["reasoning"] = test_multiple_questions([
            "What are the primary colors?",
            "If a train travels 60 km/h for 2 hours, how far does it go?"
        ], config)
        
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
            "config_used": config
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
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI interface"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            quick_test()
        elif command == "comprehensive":
            config_name = sys.argv[2] if len(sys.argv) > 2 else "groq"
            run_comprehensive_test(config_name)
        elif command == "math":
            test_basic_math()
        elif command == "web":
            test_web_search()
        elif command == "reasoning":
            test_reasoning()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: test, comprehensive, math, web, reasoning")
    else:
        print("🚀 GAIA Agent Interface")
        print("=" * 30)
        print("Available functions:")
        print("- create_gaia_agent(config)")
        print("- quick_test(question)")
        print("- run_comprehensive_test(config_name)")
        print("- test_basic_math()")
        print("- test_web_search()")
        print("- test_reasoning()")

if __name__ == "__main__":
    main()