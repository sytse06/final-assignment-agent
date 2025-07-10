# app.py - Enhanced HF Spaces Integration with Testing Framework
import os
import gradio as gr
import requests
import pandas as pd
import json
from datetime import datetime

# Import existing system with better error handling
SYSTEM_AVAILABLE = False
TESTING_AVAILABLE = False
agent_error = None

try:
    from agent_interface import create_gaia_agent, get_openrouter_config, get_performance_config, get_google_config
    SYSTEM_AVAILABLE = True
    print("✅ GAIA agent system loaded successfully")
except ImportError as e:
    agent_error = f"Agent system not available: {e}"
    print(f"⚠️  {agent_error}")
except Exception as e:
    agent_error = f"Agent initialization error: {e}"
    print(f"❌ {agent_error}")

try:
    from agent_testing import (
        run_quick_gaia_test, 
        run_gaia_test, 
        compare_agent_configs,
        run_smart_routing_test,
        analyze_failure_patterns
    )
    TESTING_AVAILABLE = True
    print("✅ Testing framework loaded successfully")
except ImportError as e:
    print(f"⚠️  Testing framework not available: {e}")
except Exception as e:
    print(f"❌ Testing framework error: {e}")

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class GAIAAgent:
    def __init__(self, config_name="groq"):
        self.agent = None
        self.system_available = SYSTEM_AVAILABLE
        self.error_message = agent_error
        self.config_name = config_name
        
        if SYSTEM_AVAILABLE:
            try:
                # Get configuration based on selection
                if config_name == "groq":
                    config = get_groq_config()
                elif config_name == "google":
                    config = get_google_config()
                elif config_name == "performance":
                    config = get_performance_config()
                else:
                    config = get_groq_config()  # Default fallback
                
                # Production settings for HF Spaces
                config.update({
                    "enable_csv_logging": False,
                    "debug_mode": False,
                    "max_agent_steps": 12,
                })
                
                self.agent = create_gaia_agent(config)
                print(f"✅ GAIA Agent initialized with {config_name} config")
            except Exception as e:
                self.error_message = f"Agent creation failed: {e}"
                print(f"❌ {self.error_message}")
        else:
            print("⚠️  Using fallback BasicAgent due to missing dependencies")
    
    def __call__(self, task_id: str, question: str) -> str:
        """Process question with GAIA agent or fallback"""
        if self.agent:
            try:
                result = self.agent.run_single_question(question=question, task_id=task_id)
                final_answer = result.get("final_answer", "No answer generated")
                return final_answer
            except Exception as e:
                return f"Processing error: {e}"
        
        return self._fallback_processing(question)
    
    def _fallback_processing(self, question: str) -> str:
        """Simple fallback responses"""
        q_lower = question.lower()
        
        if "2+2" in question.replace(" ", ""):
            return "4"
        elif "capital" in q_lower and "france" in q_lower:
            return "Paris"
        elif "president" in q_lower and "united states" in q_lower:
            return "Donald Trump"
        elif "largest" in q_lower and "planet" in q_lower:
            return "Jupiter"
        else:
            return "Unable to process question"

def run_quick_test(config_name, max_questions):
    """Run quick test on agent"""
    if not TESTING_AVAILABLE:
        return "❌ Testing framework not available", None
    
    try:
        print(f"🧪 Running quick test with {config_name} config, max {max_questions} questions")
        
        # Run the test
        results = run_quick_gaia_test(config_name, max_questions=max_questions)
        
        # Format results for display
        if results and 'results' in results:
            test_data = []
            for result in results['results'][:10]:  # Show first 10 results
                test_data.append({
                    "Task ID": result.get("task_id", "N/A"),
                    "Question": result.get("question", "N/A")[:100] + "..." if len(result.get("question", "")) > 100 else result.get("question", "N/A"),
                    "Agent Answer": result.get("final_answer", "N/A"),
                    "Expected": result.get("ground_truth", "N/A"),
                    "Correct": "✅" if result.get("is_correct", False) else "❌"
                })
            
            summary = f"""🧪 Quick Test Results ({config_name} config)
📊 Total Questions: {results.get('total_questions', 0)}
✅ Correct Answers: {results.get('correct_answers', 0)}
📈 Accuracy: {results.get('accuracy', 0):.1f}%
⏱️ Average Time: {results.get('average_time', 0):.2f}s per question
🔀 Routing Stats: {results.get('routing_stats', {})}
"""
            
            return summary, pd.DataFrame(test_data)
        else:
            return "❌ Test completed but no results returned", None
            
    except Exception as e:
        return f"❌ Test failed: {e}", None

def run_comprehensive_test(config_name, max_questions):
    """Run comprehensive GAIA test"""
    if not TESTING_AVAILABLE:
        return "❌ Testing framework not available", None
    
    try:
        print(f"🔬 Running comprehensive test with {config_name} config, max {max_questions} questions")
        
        # Run comprehensive test
        results = run_gaia_test(config_name, max_questions=max_questions)
        
        # Format results
        if results and 'results' in results:
            # Detailed results table
            test_data = []
            for result in results['results'][:20]:  # Show first 20 results
                test_data.append({
                    "Task ID": result.get("task_id", "N/A"),
                    "Level": result.get("level", "N/A"),
                    "Question": result.get("question", "N/A")[:80] + "..." if len(result.get("question", "")) > 80 else result.get("question", "N/A"),
                    "Agent Answer": result.get("final_answer", "N/A"),
                    "Expected": result.get("ground_truth", "N/A"),
                    "Correct": "✅" if result.get("is_correct", False) else "❌",
                    "Steps": result.get("total_steps", 0),
                    "Complexity": result.get("complexity", "N/A")
                })
            
            # Comprehensive summary
            level_stats = results.get('level_breakdown', {})
            summary = f"""🔬 Comprehensive Test Results ({config_name} config)

📊 Overall Performance:
• Total Questions: {results.get('total_questions', 0)}
• Correct Answers: {results.get('correct_answers', 0)}
• Overall Accuracy: {results.get('accuracy', 0):.1f}%

📈 Level Breakdown:
• Level 1: {level_stats.get('level_1', {}).get('accuracy', 0):.1f}% ({level_stats.get('level_1', {}).get('correct', 0)}/{level_stats.get('level_1', {}).get('total', 0)})
• Level 2: {level_stats.get('level_2', {}).get('accuracy', 0):.1f}% ({level_stats.get('level_2', {}).get('correct', 0)}/{level_stats.get('level_2', {}).get('total', 0)})
• Level 3: {level_stats.get('level_3', {}).get('accuracy', 0):.1f}% ({level_stats.get('level_3', {}).get('correct', 0)}/{level_stats.get('level_3', {}).get('total', 0)})

🔀 Smart Routing Performance:
• Simple Questions (one-shot): {results.get('routing_stats', {}).get('simple_accuracy', 0):.1f}%
• Complex Questions (manager): {results.get('routing_stats', {}).get('complex_accuracy', 0):.1f}%

⏱️ Performance Metrics:
• Average Time: {results.get('average_time', 0):.2f}s per question
• Total Time: {results.get('total_time', 0):.1f}s
• Average Steps: {results.get('average_steps', 0):.1f} per question

🎯 Target Achievement:
• Target (50%): {"✅ ACHIEVED" if results.get('accuracy', 0) >= 50 else "❌ Below target"}
• Stretch Goal (60%): {"✅ ACHIEVED" if results.get('accuracy', 0) >= 60 else "❌ Below stretch"}
"""
            
            return summary, pd.DataFrame(test_data)
        else:
            return "❌ Test completed but no results returned", None
            
    except Exception as e:
        return f"❌ Comprehensive test failed: {e}", None

def compare_configs():
    """Compare different agent configurations"""
    if not TESTING_AVAILABLE:
        return "❌ Testing framework not available", None
    
    try:
        print("⚖️ Comparing agent configurations...")
        
        # Compare main configurations
        comparison = compare_agent_configs(["groq", "google", "performance"])
        
        if comparison:
            comp_data = []
            for config_name, results in comparison.items():
                comp_data.append({
                    "Configuration": config_name,
                    "Accuracy": f"{results.get('accuracy', 0):.1f}%",
                    "Correct": f"{results.get('correct_answers', 0)}/{results.get('total_questions', 0)}",
                    "Avg Time": f"{results.get('average_time', 0):.2f}s",
                    "Simple Route": f"{results.get('routing_stats', {}).get('simple_percentage', 0):.1f}%",
                    "Complex Route": f"{results.get('routing_stats', {}).get('complex_percentage', 0):.1f}%",
                    "Cost Estimate": f"${results.get('estimated_cost', 0):.3f}"
                })
            
            summary = f"""⚖️ Configuration Comparison Results

📊 Best Performing:
• Highest Accuracy: {max(comparison.keys(), key=lambda k: comparison[k].get('accuracy', 0))}
• Fastest: {min(comparison.keys(), key=lambda k: comparison[k].get('average_time', float('inf')))}
• Most Cost-Effective: {min(comparison.keys(), key=lambda k: comparison[k].get('estimated_cost', float('inf')))}

🎯 Recommendations:
• For accuracy: Use the highest performing configuration
• For speed: Use the fastest configuration  
• For budget: Use the most cost-effective configuration
• For production: Balance accuracy and cost
"""
            
            return summary, pd.DataFrame(comp_data)
        else:
            return "❌ Comparison completed but no results returned", None
            
    except Exception as e:
        return f"❌ Configuration comparison failed: {e}", None

def run_and_submit_all(profile: gr.OAuthProfile | None, config_name):
    """Main evaluation function with configurable agent"""
    space_id = os.getenv("SPACE_ID")
    
    if not profile:
        return "Please login to Hugging Face.", None
    
    username = profile.username
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    
    # System status
    status_lines = [f"🚀 Starting evaluation for user: {username}"]
    status_lines.append(f"🔧 Using configuration: {config_name}")
    
    if SYSTEM_AVAILABLE:
        status_lines.append("✅ GAIA agent system active")
    else:
        status_lines.append(f"⚠️  Fallback mode: {agent_error}")
    
    # Initialize agent with selected config
    try:
        agent = GAIAAgent(config_name)
        if agent.error_message:
            status_lines.append(f"⚠️  Agent warning: {agent.error_message}")
    except Exception as e:
        return f"❌ Error initializing agent: {e}", None
    
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    
    # Fetch and process questions (same as before)
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        
        if not questions_data:
            return "❌ No questions received from server.", None
            
        status_lines.append(f"✅ Received {len(questions_data)} questions")
            
    except Exception as e:
        return f"❌ Error fetching questions: {e}", None
    
    # Process questions
    results_log = []
    answers_payload = []
    
    for i, item in enumerate(questions_data, 1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        
        if not task_id or not question_text:
            continue
        
        try:
            submitted_answer = agent(task_id=task_id, question=question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            
            display_question = question_text[:100] + "..." if len(question_text) > 100 else question_text
            results_log.append({
                "Task ID": task_id,
                "Question": display_question,
                "Submitted Answer": submitted_answer
            })
            
        except Exception as e:
            error_answer = f"ERROR: {str(e)}"
            answers_payload.append({"task_id": task_id, "submitted_answer": error_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text,
                "Submitted Answer": error_answer
            })
    
    if not answers_payload:
        return "❌ No answers generated.", pd.DataFrame(results_log)
    
    # Submit results
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }
    
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        
        final_status = "\n".join(status_lines) + "\n\n" + (
            f"🎉 Submission Successful!\n"
            f"👤 User: {result_data.get('username')}\n"
            f"📊 Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"💬 Message: {result_data.get('message', 'No message')}"
        )
        
        return final_status, pd.DataFrame(results_log)
        
    except Exception as e:
        error_status = "\n".join(status_lines) + f"\n\n❌ Submission failed: {e}"
        return error_status, pd.DataFrame(results_log)

# Enhanced Gradio Interface with Testing
with gr.Blocks(title="GAIA Agent Evaluation & Testing") as demo:
    gr.Markdown("# 🤖 GAIA Agent Evaluation & Testing System")
    
    if SYSTEM_AVAILABLE:
        gr.Markdown("✅ **Status**: Advanced multi-agent GAIA system loaded")
        gr.Markdown("""
        **System Features:**
        - 🧠 Smart routing (simple → direct LLM, complex → multi-agent)
        - 🔧 4 specialized agents (data analyst, web researcher, document processor, general assistant)
        - 📚 RAG-enhanced decision making with 165 GAIA examples
        - 🎯 GAIA-compliant answer formatting
        - 💰 Cost-optimized with automatic fallback
        """)
    else:
        gr.Markdown(f"⚠️ **Status**: Fallback mode active - {agent_error}")
    
    if TESTING_AVAILABLE:
        gr.Markdown("✅ **Testing Framework**: Available")
    else:
        gr.Markdown("⚠️ **Testing Framework**: Not available")
    
    with gr.Tabs():
        # Testing Tab
        with gr.TabItem("🧪 Testing & Validation"):
            gr.Markdown("### Test your agent before final submission")
            
            with gr.Row():
                test_config = gr.Dropdown(
                    choices=["groq", "google", "performance"],
                    value="groq",
                    label="Configuration to Test"
                )
                test_questions = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=5,
                    step=1,
                    label="Number of Questions"
                )
            
            with gr.Row():
                quick_test_btn = gr.Button("🧪 Quick Test", variant="secondary")
                comprehensive_test_btn = gr.Button("🔬 Comprehensive Test", variant="primary")
                compare_btn = gr.Button("⚖️ Compare Configs", variant="secondary")
            
            test_output = gr.Textbox(
                label="📊 Test Results",
                lines=15,
                interactive=False
            )
            
            test_table = gr.DataFrame(
                label="📋 Detailed Results",
                wrap=True
            )
            
            # Connect test buttons
            quick_test_btn.click(
                fn=lambda config, questions: run_quick_test(config, questions),
                inputs=[test_config, test_questions],
                outputs=[test_output, test_table]
            )
            
            comprehensive_test_btn.click(
                fn=lambda config, questions: run_comprehensive_test(config, questions),
                inputs=[test_config, test_questions],
                outputs=[test_output, test_table]
            )
            
            compare_btn.click(
                fn=compare_configs,
                outputs=[test_output, test_table]
            )
        
        # Submission Tab
        with gr.TabItem("🚀 Final Submission"):
            gr.Markdown("### Submit to course evaluation system")
            gr.Markdown("""
            **Instructions:**
            1. **Login** to Hugging Face
            2. **Select** your preferred configuration
            3. **Click 'Run Evaluation'** to process all questions and submit
            """)
            
            gr.LoginButton()
            
            with gr.Row():
                submit_config = gr.Dropdown(
                    choices=["groq", "google", "performance"],
                    value="groq",
                    label="Configuration for Submission"
                )
            
            with gr.Row():
                run_button = gr.Button(
                    "🚀 Run Evaluation & Submit All Answers", 
                    variant="primary",
                    size="lg"
                )
            
            status_output = gr.Textbox(
                label="📊 Submission Status", 
                lines=10, 
                interactive=False
            )
            
            results_table = gr.DataFrame(
                label="📋 Submission Results", 
                wrap=True
            )
            
            # Connect submission button
            run_button.click(
                fn=lambda profile, config: run_and_submit_all(profile, config),
                inputs=[gr.LoginButton.load(), submit_config],
                outputs=[status_output, results_table]
            )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 GAIA Agent Evaluation & Testing System Starting")
    print("="*60)
    
    # Environment check
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")
    
    if space_host:
        print(f"✅ SPACE_HOST: {space_host}")
    if space_id:
        print(f"✅ SPACE_ID: {space_id}")
    
    # System status
    print(f"✅ GAIA agent system: {'LOADED' if SYSTEM_AVAILABLE else 'FALLBACK'}")
    print(f"✅ Testing framework: {'LOADED' if TESTING_AVAILABLE else 'NOT AVAILABLE'}")
    
    print("="*60)
    print("🎯 Launching enhanced Gradio interface...")
    
    demo.launch(debug=True, share=False)