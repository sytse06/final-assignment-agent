# app.py - Enhanced Course Integration Adapter with Quick Test
import os
import gradio as gr
import requests
import pandas as pd

# Import existing system
try:
    from agent_interface import create_gaia_agent, get_groq_config, get_openrouter_config
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Agent system not available: {e}")
    SYSTEM_AVAILABLE = False

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class GAIAAgent:
    def __init__(self):
        self.agent = None
        
        if SYSTEM_AVAILABLE:
            try:
                # Try OpenRouter first (better for free tier), fallback to Groq
                try:
                    config = get_openrouter_config()
                    provider_name = "OpenRouter + Gemini 2.5 Flash"
                except:
                    config = get_groq_config()
                    provider_name = "Groq"
                
                # Handle both dataclass and dict config types
                if hasattr(config, '__dataclass_fields__'):
                    # It's a dataclass - set attributes directly
                    config.enable_csv_logging = False
                    config.debug_mode = False
                    config.max_agent_steps = 10
                else:
                    # It's a dict - use update method
                    config.update({
                        "enable_csv_logging": False, 
                        "debug_mode": False,
                        "max_agent_steps": 10
                    })
                
                self.agent = create_gaia_agent(config)
                print(f"✅ GAIA Agent initialized successfully with {provider_name}")
            except Exception as e:
                print(f"❌ Agent initialization failed: {e}")
                self.agent = None
        else:
            print("⚠️ Using fallback mode - agent system not available")
    
    def __call__(self, task_id: str, question: str) -> str:
        if self.agent:
            try:
                result = self.agent.run_single_question(question=question, task_id=task_id)
                answer = result.get("final_answer", "No answer generated")
                print(f"✅ Processed question {task_id}: {answer[:50]}...")
                return answer
            except Exception as e:
                error_msg = f"Processing error: {str(e)}"
                print(f"❌ {error_msg}")
                return error_msg
        
        # Fallback responses for testing
        question_lower = question.lower()
        if "2+2" in question.replace(" ", ""):
            return "4"
        elif "capital" in question_lower and "france" in question_lower:
            return "Paris" 
        elif "hello" in question_lower:
            return "Hello! I'm a GAIA agent."
        elif "square root" in question_lower and "144" in question:
            return "12"
        else:
            return "Question processed by fallback system"

# Quick test function for individual question testing
def quick_test_question(question: str) -> str:
    """Quick single question test function"""
    if not question.strip():
        return "❓ Please enter a question to test"
    
    if not SYSTEM_AVAILABLE:
        return "❌ System not available - agent import failed"
    
    try:
        # Create a test agent with minimal config
        try:
            config = get_openrouter_config()
            provider = "OpenRouter"
        except:
            config = get_groq_config()
            provider = "Groq"
        
        # Disable logging for testing
        if hasattr(config, '__dataclass_fields__'):
            config.enable_csv_logging = False
            config.debug_mode = False
            config.max_agent_steps = 5
        else:
            config.update({
                "enable_csv_logging": False, 
                "debug_mode": False,
                "max_agent_steps": 5
            })
        
        agent = create_gaia_agent(config)
        result = agent.run_single_question(question)
        
        # Format result
        answer = result.get('final_answer', 'No answer')
        complexity = result.get('complexity', 'unknown')
        strategy = result.get('strategy_used', 'unknown')
        steps = len(result.get('steps', []))
        
        return f"""✅ **Test Successful!**

**Question:** {question}

**Answer:** {answer}

**Details:**
- **Complexity:** {complexity}
- **Strategy:** {strategy}
- **Steps:** {steps}
- **Provider:** {provider}

**Status:** Agent working correctly! 🚀"""
        
    except Exception as e:
        return f"""❌ **Test Failed:**

**Question:** {question}

**Error:** {str(e)}

**Status:** Check configuration and try again"""

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Main function to run evaluation and submit results"""
    space_id = os.getenv("SPACE_ID")
    
    if not profile:
        return "❌ Please login to Hugging Face first.", None
    
    username = profile.username
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    
    # Initialize agent
    try:
        agent = GAIAAgent()
    except Exception as e:
        return f"❌ Error initializing agent: {e}", None
    
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    
    # Fetch questions
    try:
        print("📥 Fetching questions from scoring server...")
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        
        if not questions_data:
            return "❌ No questions received from server.", None
        
        print(f"📋 Received {len(questions_data)} questions")
            
    except Exception as e:
        return f"❌ Error fetching questions: {e}", None
    
    # Process questions
    results_log = []
    answers_payload = []
    
    for i, item in enumerate(questions_data, 1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        
        if not task_id or not question_text:
            print(f"⚠️ Skipping incomplete question {i}")
            continue
        
        print(f"🔄 Processing question {i}/{len(questions_data)}: {task_id}")
        
        try:
            submitted_answer = agent(task_id=task_id, question=question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            
            # Truncate long questions for display
            display_question = question_text[:100] + "..." if len(question_text) > 100 else question_text
            results_log.append({
                "Task ID": task_id,
                "Question": display_question,
                "Submitted Answer": submitted_answer[:100] + "..." if len(submitted_answer) > 100 else submitted_answer
            })
            
        except Exception as e:
            error_answer = f"ERROR: {str(e)}"
            answers_payload.append({"task_id": task_id, "submitted_answer": error_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text,
                "Submitted Answer": error_answer
            })
            print(f"❌ Error processing question {task_id}: {e}")
    
    if not answers_payload:
        return "❌ No answers generated.", pd.DataFrame(results_log)
    
    # Submit results
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }
    
    try:
        print("📤 Submitting answers to scoring server...")
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        
        final_status = (
            f"🎉 Submission Successful!\n"
            f"👤 User: {result_data.get('username')}\n"
            f"📊 Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"💬 Message: {result_data.get('message', 'No message')}\n"
            f"🔗 Agent Code: {agent_code}"
        )
        
        print(f"✅ Submission complete: {result_data.get('score', 'N/A')}% accuracy")
        return final_status, pd.DataFrame(results_log)
        
    except Exception as e:
        error_msg = f"❌ Submission failed: {e}"
        print(error_msg)
        return error_msg, pd.DataFrame(results_log)

# Enhanced Gradio Interface with Quick Test
with gr.Blocks(
    title="GAIA Agent Evaluation",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("""
    # 🧠 GAIA Agent Evaluation
    
    **Multi-agent system for GAIA benchmark evaluation**
    
    This agent uses:
    - 🤖 Smart routing between specialized agents
    - 📁 File processing capabilities  
    - 🔍 RAG-enhanced decision making
    - 🛡️ Production error handling
    
    *Built for HF Agents Course Final Assignment*
    """)
    
    # Quick Test Section
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🧪 Quick Test")
            gr.Markdown("Test your agent with a single question before running the full evaluation.")
            
            test_question = gr.Textbox(
                label="Test Question",
                placeholder="Enter a question to test your agent...",
                lines=2
            )
            
            with gr.Row():
                quick_test_btn = gr.Button("🧪 Test Agent", variant="primary")
                clear_test_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            test_result = gr.Markdown(
                value="*Enter a question above and click 'Test Agent' to verify your system works.*",
                label="Test Result"
            )
            
            # Example questions
            gr.Examples(
                examples=[
                    "What is 2 + 2?",
                    "What is the capital of France?",
                    "Calculate the square root of 144",
                    "What are the main components of photosynthesis?",
                    "How many continents are there?"
                ],
                inputs=test_question,
                label="Quick Test Examples:"
            )
    
    gr.Markdown("---")
    
    # Main Evaluation Section
    gr.Markdown("## 🎯 Full GAIA Evaluation")
    gr.Markdown("**Warning:** This will submit your results to the course evaluation system.")
    
    gr.LoginButton()
    
    with gr.Row():
        run_button = gr.Button("🚀 Run Evaluation & Submit All Answers", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            status_output = gr.Textbox(
                label="📋 Evaluation Status", 
                lines=8, 
                interactive=False,
                placeholder="Click the button above to start evaluation..."
            )
        with gr.Column():
            results_table = gr.DataFrame(
                label="📊 Results Table", 
                wrap=True,
                height=400
            )
    
    # Event handlers - Connect the buttons to functions
    quick_test_btn.click(
        fn=quick_test_question,
        inputs=[test_question],
        outputs=[test_result]
    )
    
    clear_test_btn.click(
        fn=lambda: ("", "*Enter a question above and click 'Test Agent' to verify your system works.*"),
        outputs=[test_question, test_result]
    )
    
    test_question.submit(
        fn=quick_test_question,
        inputs=[test_question],
        outputs=[test_result]
    )
    
    run_button.click(
        fn=run_and_submit_all, 
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=False)
