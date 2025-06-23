# agent_interface.py  - Interface and utility methods
# Public interface and convenience functions for GAIA agent 
# including testing and verification

import uuid
import datetime
from typing import Dict, List, Optional, Union
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

def run_single_question_with_logging(agent: GAIAAgent, question: str, task_id: str = None) -> Dict:
    """Execute single question using the new process_question() method with enhanced metadata"""
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    print(f"🔍 Processing: {question[:60]}...")
    
    # Use the core process_question method (which already includes logging)
    result = agent.process_question(question, task_id)
    
    # Add enhanced metadata for interface layer
    if agent.logging:
        execution_time = (datetime.datetime.now() - agent.logging.task_start_time).total_seconds() if agent.logging.task_start_time else 0
        manual_steps = len(agent.logging.manual_steps) if hasattr(agent.logging, 'manual_steps') else 0
        
        # Enhance the result with additional interface-level metadata
        result.update({
            "complexity": getattr(agent.logging, 'current_complexity', 'unknown'),
            "routing_path": getattr(agent.logging, 'current_routing_path', 'unknown'),
            "total_manual_steps": manual_steps,
            "execution_time": execution_time,
            "model_used": getattr(agent.logging, 'current_model_used', 'unknown'),
            "similar_examples_count": getattr(agent.logging, 'current_similar_examples_count', 0)
        })
    
    return result

def verify_logging_coverage(agent: GAIAAgent) -> Dict:
    """Verify that all workflow nodes have logging coverage"""
    
    workflow_nodes = [
        "_read_question_node",
        "_complexity_check_node", 
        "_one_shot_answering_node",
        "_manager_execution_node",
        "_format_answer_node",
        "_llm_complexity_check",
        "_route_by_complexity",
        "_extract_final_answer",
        "_apply_gaia_formatting",
        "_prepare_manager_context"
    ]
    
    coverage_report = {
        "total_nodes": len(workflow_nodes),
        "nodes_with_logging": 0,
        "missing_logging": [],
        "has_logging": []
    }
    
    for node_name in workflow_nodes:
        if hasattr(agent, node_name):
            node_method = getattr(agent, node_name)
            # Check if the method source contains logging calls
            import inspect
            try:
                source = inspect.getsource(node_method)
                if "self.logging.log_step" in source:
                    coverage_report["nodes_with_logging"] += 1
                    coverage_report["has_logging"].append(node_name)
                else:
                    coverage_report["missing_logging"].append(node_name)
            except:
                coverage_report["missing_logging"].append(f"{node_name} (source unavailable)")
        else:
            coverage_report["missing_logging"].append(f"{node_name} (method not found)")
    
    # Calculate coverage percentage
    coverage_report["coverage_percentage"] = (
        coverage_report["nodes_with_logging"] / coverage_report["total_nodes"] * 100
    )
    
    print(f"📊 LOGGING COVERAGE REPORT")
    print("=" * 30)
    print(f"Total workflow nodes: {coverage_report['total_nodes']}")
    print(f"Nodes with logging: {coverage_report['nodes_with_logging']}")
    print(f"Coverage: {coverage_report['coverage_percentage']:.1f}%")
    
    if coverage_report["missing_logging"]:
        print(f"\n❌ Missing logging:")
        for node in coverage_report["missing_logging"]:
            print(f"   - {node}")
    
    if coverage_report["has_logging"]:
        print(f"\n✅ Has logging:")
        for node in coverage_report["has_logging"]:
            print(f"   - {node}")
    
    return coverage_report

def analyze_agent_performance(agent: GAIAAgent) -> Dict:
    """Analyze agent performance and logging status"""
    
    analysis = {
        "agent_initialized": agent is not None,
        "logging_enabled": agent.logging is not None,
        "specialists_count": len(agent.specialists) if hasattr(agent, 'specialists') else 0,
        "manager_available": hasattr(agent, 'manager') and agent.manager is not None,
        "workflow_built": hasattr(agent, 'workflow') and agent.workflow is not None,
        "config": {
            "model_provider": agent.config.model_provider,
            "model_name": agent.config.model_name,
            "smart_routing": agent.config.enable_smart_routing,
            "csv_logging": agent.config.enable_csv_logging
        }
    }
    
    # Check logging status
    if agent.logging:
        analysis["logging_status"] = {
            "csv_loggers_available": hasattr(agent.logging, 'csv_loggers'),
            "batch_timestamp": getattr(agent.logging, 'batch_timestamp', 'unknown'),
            "current_files": agent.logging.current_log_files
        }
        
        # Check if any manual steps have been logged
        if hasattr(agent.logging, 'manual_steps'):
            analysis["logging_status"]["manual_steps_count"] = len(agent.logging.manual_steps)
    
    print(f"🔍 AGENT PERFORMANCE ANALYSIS")
    print("=" * 35)
    print(f"Agent initialized: {'✅' if analysis['agent_initialized'] else '❌'}")
    print(f"Logging enabled: {'✅' if analysis['logging_enabled'] else '❌'}")
    print(f"Specialists: {analysis['specialists_count']}")
    print(f"Manager available: {'✅' if analysis['manager_available'] else '❌'}")
    print(f"Workflow built: {'✅' if analysis['workflow_built'] else '❌'}")
    
    print(f"\n⚙️  Configuration:")
    for key, value in analysis['config'].items():
        print(f"   {key}: {value}")
    
    if 'logging_status' in analysis:
        print(f"\n📊 Logging Status:")
        for key, value in analysis['logging_status'].items():
            print(f"   {key}: {value}")
    
    return analysis

def quick_test(question: str = "Calculate 15% of 1000", 
               config_overrides: Union[Dict, str] = None) -> Dict:
    """Quick test function with custom config"""
    try:
        agent = create_gaia_agent(config_overrides)
        result = agent.process_question(question)
        
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
            result = run_single_question_with_logging(agent, question)
            results.append(result)
            
            # Show enhanced information
            print(f"✅ Answer: {result.get('final_answer', 'No answer')}")
            print(f"🔄 Strategy: {result.get('routing_path', 'unknown')}")
            print(f"🤖 Agent: {result.get('selected_agent', 'unknown')}")
            print(f"⏱️ Time: {result.get('execution_time', 0):.2f}s")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({"question": question, "error": str(e)})
    
    return results

def test_agent_with_questions(agent_config_name: str = "groq", questions: List[str] = None) -> Dict:
    """Test agent with a set of questions using process_question()"""
    
    if questions is None:
        questions = [
            "What is 2 + 2?",
            "Calculate 15% of 200",
            "What are the primary colors?",
            "What is the capital of France?"
        ]
    
    print(f"🧪 TESTING AGENT WITH {len(questions)} QUESTIONS")
    print(f"   Config: {agent_config_name}")
    print("=" * 50)
    
    # Create agent
    config = get_agent_config(agent_config_name)
    config.enable_csv_logging = True
    
    agent = create_gaia_agent(config)
    
    results = []
    start_time = datetime.datetime.now()
    
    for i, question in enumerate(questions, 1):
        print(f"\n🔄 Question {i}/{len(questions)}: {question}")
        
        # Use the enhanced interface method (which uses process_question internally)
        result = run_single_question_with_logging(agent, question)
        
        results.append({
            "question_number": i,
            "question": question,
            "final_answer": result.get("final_answer", "No answer"),
            "execution_successful": result.get("execution_successful", False),
            "complexity": result.get("complexity", "unknown"),
            "routing_path": result.get("routing_path", "unknown"),
            "manual_steps": result.get("total_manual_steps", 0),
            "execution_time": result.get("execution_time", 0)
        })
        
        status = "✅" if result.get("execution_successful", False) else "❌"
        answer = result.get("final_answer", "No answer")
        print(f"{status} Answer: {answer}")
    
    total_time = (datetime.datetime.now() - start_time).total_seconds()
    
    # Generate summary
    successful_questions = sum(1 for r in results if r["execution_successful"])
    total_steps = sum(r["manual_steps"] for r in results)
    
    summary = {
        "test_config": agent_config_name,
        "total_questions": len(questions),
        "successful_questions": successful_questions,
        "success_rate": successful_questions / len(questions) if questions else 0,
        "total_execution_time": total_time,
        "average_time_per_question": total_time / len(questions) if questions else 0,
        "total_manual_steps": total_steps,
        "average_steps_per_question": total_steps / len(questions) if questions else 0,
        "results": results,
        "logging_analysis": analyze_agent_performance(agent),
        "csv_files": agent.logging.current_log_files if agent.logging else {}
    }
    
    print(f"\n🎯 TEST SUMMARY")
    print("=" * 20)
    print(f"Questions: {successful_questions}/{len(questions)} ({summary['success_rate']:.1%})")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg per question: {summary['average_time_per_question']:.1f}s")
    print(f"Total steps logged: {total_steps}")
    print(f"Avg steps per question: {summary['average_steps_per_question']:.1f}")
    
    if agent.logging and hasattr(agent.logging, 'csv_loggers'):
        print(f"\n📁 CSV Files:")
        for log_type, file_path in agent.logging.csv_loggers.items():
            print(f"   {log_type.replace('_file', '').title()}: {file_path}")
    
    return summary

def quick_agent_test(agent_config_name: str = "groq") -> Dict:
    """Quick test using process_question()"""
    
    print(f"⚡ QUICK AGENT TEST")
    print(f"   Config: {agent_config_name}")
    print("=" * 25)
    
    try:
        # Create agent
        config = get_agent_config(agent_config_name)
        config.enable_csv_logging = True
        
        agent = create_gaia_agent(config)
        print("✅ Agent created successfully")
        
        # Quick performance check
        performance = analyze_agent_performance(agent)
        
        # Test with simple question using process_question
        test_question = "What is 2 + 2?"
        print(f"\n🔄 Testing with: {test_question}")
        
        result = agent.process_question(test_question)  # Direct use of process_question
        
        success = result.get("execution_successful", False)
        answer = result.get("final_answer", "No answer")
        
        # Get additional info from logging
        steps = 0
        time_taken = 0
        if agent.logging:
            steps = len(agent.logging.manual_steps) if hasattr(agent.logging, 'manual_steps') else 0
            if agent.logging.task_start_time:
                time_taken = (datetime.datetime.now() - agent.logging.task_start_time).total_seconds()
        
        print(f"{'✅' if success else '❌'} Result: {answer}")
        print(f"📊 Steps logged: {steps}")
        print(f"⏱️  Time: {time_taken:.1f}s")
        
        # Verify logging
        coverage = verify_logging_coverage(agent)
        
        return {
            "success": success,
            "agent_created": True,
            "answer": answer,
            "steps_logged": steps,
            "execution_time": time_taken,
            "performance_analysis": performance,
            "logging_coverage": coverage,
            "csv_files": agent.logging.current_log_files if agent.logging else {}
        }
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def compare_agent_configs(config_names: List[str], test_question: str = "What is 15% of 200?") -> Dict:
    """Compare different agent configurations using process_question()"""
    
    print(f"⚖️  COMPARING AGENT CONFIGURATIONS")
    print(f"   Configs: {', '.join(config_names)}")
    print(f"   Test question: {test_question}")
    print("=" * 50)
    
    comparison_results = {}
    
    for config_name in config_names:
        print(f"\n🔄 Testing config: {config_name}")
        
        try:
            config = get_agent_config(config_name)
            config.enable_csv_logging = True
            
            agent = create_gaia_agent(config)
            result = agent.process_question(test_question)  # Use process_question
            
            # Get additional metadata from logging
            steps_logged = 0
            execution_time = 0
            complexity = "unknown"
            routing_path = "unknown"
            
            if agent.logging:
                steps_logged = len(agent.logging.manual_steps) if hasattr(agent.logging, 'manual_steps') else 0
                complexity = getattr(agent.logging, 'current_complexity', 'unknown')
                routing_path = getattr(agent.logging, 'current_routing_path', 'unknown')
                if agent.logging.task_start_time:
                    execution_time = (datetime.datetime.now() - agent.logging.task_start_time).total_seconds()
            
            comparison_results[config_name] = {
                "success": result.get("execution_successful", False),
                "answer": result.get("final_answer", "No answer"),
                "complexity": complexity,
                "routing_path": routing_path,
                "steps_logged": steps_logged,
                "execution_time": execution_time,
                "model_name": config.model_name,
                "provider": config.model_provider
            }
            
            status = "✅" if result.get("execution_successful", False) else "❌"
            print(f"{status} {config_name}: {result.get('final_answer', 'No answer')}")
            
        except Exception as e:
            comparison_results[config_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ {config_name}: Failed - {e}")
    
    # Generate comparison summary
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 25)
    
    for config_name, result in comparison_results.items():
        if result.get("success", False):
            print(f"\n✅ {config_name.upper()}:")
            print(f"   Answer: {result['answer']}")
            print(f"   Model: {result['provider']}/{result['model_name']}")
            print(f"   Routing: {result['routing_path']}")
            print(f"   Steps: {result['steps_logged']}")
            print(f"   Time: {result['execution_time']:.1f}s")
        else:
            print(f"\n❌ {config_name.upper()}: {result.get('error', 'Failed')}")
    
    return comparison_results

def test_routing_behavior(agent_config_name: str = "groq") -> Dict:
    """Test smart routing using process_question()"""
    
    routing_test_questions = [
        {"question": "What is 2 + 2?", "expected_routing": "simple", "type": "simple_math"},
        {"question": "Calculate 15% of 200", "expected_routing": "simple", "type": "simple_math"},
        {"question": "What are the primary colors?", "expected_routing": "simple", "type": "simple_fact"},
        {"question": "What is the current population of Tokyo?", "expected_routing": "complex", "type": "web_search"},
        {"question": "Analyze the data in the attached spreadsheet", "expected_routing": "complex", "type": "file_processing"},
        {"question": "Write a Python script to calculate fibonacci numbers", "expected_routing": "complex", "type": "code_generation"}
    ]
    
    print(f"🔀 TESTING SMART ROUTING BEHAVIOR")
    print(f"   Config: {agent_config_name}")
    print(f"   Questions: {len(routing_test_questions)}")
    print("=" * 40)
    
    config = get_agent_config(agent_config_name)
    config.enable_csv_logging = True
    config.enable_smart_routing = True  # Ensure smart routing is enabled
    
    agent = create_gaia_agent(config)
    
    routing_results = []
    
    for i, test_case in enumerate(routing_test_questions, 1):
        question = test_case["question"]
        expected = test_case["expected_routing"]
        q_type = test_case["type"]
        
        print(f"\n🔄 Test {i}: {q_type}")
        print(f"   Question: {question}")
        print(f"   Expected routing: {expected}")
        
        # Use process_question directly
        result = agent.process_question(question)
        
        # Get routing info from logging
        actual_routing = "unknown"
        actual_complexity = "unknown"
        
        if agent.logging:
            actual_complexity = getattr(agent.logging, 'current_complexity', 'unknown')
            actual_routing = getattr(agent.logging, 'current_routing_path', 'unknown')
        
        routing_correct = (
            (expected == "simple" and actual_routing == "one_shot_llm") or
            (expected == "complex" and actual_routing == "manager_coordination")
        )
        
        routing_results.append({
            "question": question,
            "question_type": q_type,
            "expected_routing": expected,
            "actual_routing": actual_routing,
            "actual_complexity": actual_complexity,
            "routing_correct": routing_correct,
            "execution_successful": result.get("execution_successful", False),
            "answer": result.get("final_answer", "No answer"),
            "steps_logged": len(agent.logging.manual_steps) if agent.logging and hasattr(agent.logging, 'manual_steps') else 0
        })
        
        status = "✅" if routing_correct else "❌"
        print(f"   {status} Actual: {actual_complexity} → {actual_routing}")
        print(f"   Answer: {result.get('final_answer', 'No answer')}")
    
    # Generate routing analysis
    correct_routing = sum(1 for r in routing_results if r["routing_correct"])
    total_questions = len(routing_results)
    routing_accuracy = correct_routing / total_questions if total_questions > 0 else 0
    
    # Breakdown by routing path
    routing_breakdown = {}
    for result in routing_results:
        path = result["actual_routing"]
        if path not in routing_breakdown:
            routing_breakdown[path] = {"count": 0, "successful": 0}
        routing_breakdown[path]["count"] += 1
        if result["execution_successful"]:
            routing_breakdown[path]["successful"] += 1
    
    summary = {
        "config": agent_config_name,
        "total_questions": total_questions,
        "correct_routing": correct_routing,
        "routing_accuracy": routing_accuracy,
        "routing_breakdown": routing_breakdown,
        "detailed_results": routing_results,
        "csv_files": agent.logging.current_log_files if agent.logging else {}
    }
    
    print(f"\n🎯 ROUTING ANALYSIS")
    print("=" * 20)
    print(f"Routing accuracy: {correct_routing}/{total_questions} ({routing_accuracy:.1%})")
    
    print(f"\n📊 Routing breakdown:")
    for path, stats in routing_breakdown.items():
        success_rate = stats["successful"] / stats["count"] if stats["count"] > 0 else 0
        print(f"   {path}: {stats['count']} questions ({success_rate:.1%} successful)")
    
    return summary

# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

def get_groq_config(model_name: str = "qwen-qwq-32b") -> GAIAConfig:
    """Get Groq configuration with enhanced defaults"""
    return GAIAConfig(
        model_provider="groq",
        model_name=model_name,
        temperature=0.3,
        max_agent_steps=15,
        enable_smart_routing=True,
        skip_rag_for_simple=True,
        enable_csv_logging=True,
        debug_mode=True
    )

def get_google_config(model_name: str = "gemini-2.5-flash-preview") -> GAIAConfig:
    """Get Google configuration with enhanced defaults"""
    return GAIAConfig(
        model_provider="google",
        model_name=model_name,
        temperature=0.3,
        max_agent_steps=15,
        enable_smart_routing=True,
        skip_rag_for_simple=True,
        enable_csv_logging=True,
        debug_mode=True
    )

def get_openrouter_config(model_name: str = "google/gemini-2.5-flash") -> GAIAConfig:
    """Get OpenRouter configuration with enhanced defaults"""
    return GAIAConfig(
        model_provider="openrouter",
        model_name=model_name,
        temperature=0.1,
        max_agent_steps=10,
        enable_smart_routing=True,
        skip_rag_for_simple=True,
        enable_csv_logging=True,
        debug_mode=True
    )

def get_performance_config() -> GAIAConfig:
    """Get performance-optimized configuration"""
    return GAIAConfig(
        model_provider="groq",
        model_name="qwen-qwq-32b",
        temperature=0.1,  # Lower temperature for consistency
        max_agent_steps=12,  # Fewer steps for speed
        enable_smart_routing=True,
        skip_rag_for_simple=True,
        enable_csv_logging=True,
        debug_mode=False  # Less verbose for performance
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

def get_debug_config() -> GAIAConfig:
    """Get debug configuration with enhanced logging"""
    return GAIAConfig(
        model_provider="groq",
        model_name="qwen-qwq-32b",
        temperature=0.3,
        max_agent_steps=15,
        enable_smart_routing=True,
        skip_rag_for_simple=True,
        enable_csv_logging=True,
        debug_mode=True
    )

def get_accuracy_config() -> GAIAConfig:
    """Get accuracy-optimized configuration"""
    return GAIAConfig(
        model_provider="google",
        model_name="gemini-2.5-flash-preview",
        temperature=0.2,
        max_agent_steps=20,
        enable_smart_routing=True,
        skip_rag_for_simple=False,
        rag_examples_count=5,
        enable_csv_logging=True,
        debug_mode=True
    )

def get_agent_config(config_name: str) -> GAIAConfig:
    """Get agent configuration by name"""
    configs = {
        "groq": get_groq_config,
        "openrouter": get_openrouter_config,
        # Add other configs as needed
    }
    
    if config_name not in configs:
        print(f"⚠️  Unknown config '{config_name}', using groq")
        config_name = "groq"
    
    return configs[config_name]()

def create_gaia_agent(config: GAIAConfig = None) -> GAIAAgent:
    """Create GAIA agent with configuration"""
    if config is None:
        config = get_groq_config()
    
    return GAIAAgent(config)

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

def run_quick_test(config_name: str = "groq"):
    """Run a quick test with minimal output"""
    result = quick_agent_test(config_name)
    return result["success"]

def run_routing_test(config_name: str = "groq"):
    """Run routing test and return accuracy"""
    result = test_routing_behavior(config_name)
    return result["routing_accuracy"]

def run_performance_comparison():
    """Compare performance across different configurations"""
    configs = ["groq", "google", "performance"]
    return compare_agent_configs(configs)

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
                    result = agent.process_question(question)
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
# BATCH TESTING UTILITIES
# ============================================================================

def run_batch_test(config_name: str = "groq", num_questions: int = 10) -> Dict:
    """Run a batch test with multiple questions"""
    
    # Generate test questions of different types
    test_questions = [
        "What is 2 + 2?",
        "Calculate 25% of 80",
        "What are the primary colors?",
        "What is the capital of Japan?",
        "How many days are in a leap year?",
        "What is 144 divided by 12?",
        "What color do you get when you mix red and blue?",
        "How many sides does a triangle have?",
        "What is 7 times 8?",
        "What is the freezing point of water in Celsius?"
    ]
    
    # Use only the requested number of questions
    selected_questions = test_questions[:num_questions]
    
    return test_agent_with_questions(config_name, selected_questions)

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
    print("   - 'openrouter': OpenRouter models")
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