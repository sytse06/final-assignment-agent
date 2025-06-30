# Add this to the END of your app.py file after the quick_test_question function

def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")
    
    if not profile:
        return "Please login to Hugging Face.", None
    
    username = profile.username
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    
    # Initialize agent
    try:
        agent = GAIAAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None
    
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    
    # Fetch questions
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        
        if not questions_data:
            return "No questions received.", None
            
    except Exception as e:
        return f"Error fetching questions: {e}", None
    
    # Process questions
    results_log = []
    answers_payload = []
    
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        
        if not task_id or not question_text:
            continue
        
        try:
            submitted_answer = agent(task_id=task_id, question=question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text,
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
        return "No answers generated.", pd.DataFrame(results_log)
    
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
        
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message')}"
        )
        
        return final_status, pd.DataFrame(results_log)
        
    except Exception as e:
        return f"Submission failed: {e}", pd.DataFrame(results_log)

# NEW GRADIO INTERFACE - This is what you're missing!
with gr.Blocks(
    title="GAIA Agent Evaluation",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("# GAIA Agent Evaluation")
    gr.Markdown("Multi-agent system for GAIA benchmark evaluation.")
    
    # Quick Test Section - THIS IS THE NEW PART
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
                    "What are the main components of photosynthesis?"
                ],
                inputs=test_question,
                label="Quick Test Examples:"
            )
    
    gr.Markdown("---")
    
    # Main Evaluation Section
    gr.Markdown("## 🎯 Full GAIA Evaluation")
    gr.Markdown("**Warning:** This will submit your results to the course evaluation system.")
    
    gr.LoginButton()
    
    run_button = gr.Button("🚀 Run Evaluation & Submit All Answers", variant="primary", size="lg")
    status_output = gr.Textbox(label="Status", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Results", wrap=True)
    
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
    
    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    demo.launch(debug=True, share=False)