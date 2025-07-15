# test_agent.py
from agent_interface import create_gaia_agent

def test_agent():
    try:
        print("🔄 Creating agent...")
        agent = create_gaia_agent('openrouter')
        print("✅ Agent created")
        
        print("🔄 Testing simple question...")
        result = agent.process_question("What is 2+2?")
        print(f"✅ Answer: {result['final_answer']}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_agent()