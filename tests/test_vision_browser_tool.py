# test_vision_browser_tool.py
import sys
import os

# Add tool class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tools'))

from vision_browser_tool import VisionBrowserTool, setup_agentic_browser

def test_actual_tool():
    print("🔧 Testing actual VisionBrowserTool...")
    
    # Test the tool class directly
    tool = VisionBrowserTool(headless=True)  # Use headless for testing
    
    # Test setup_agentic action
    print("\n📋 Testing setup_agentic...")
    result = tool.forward(
        action="setup_agentic",
        parameters={}
    )
    print("Setup Result:", result)
    
    # Test navigation
    print("\n🌐 Testing navigation...")
    result = tool.forward(
        action="navigate_to",
        parameters={"url": "https://nu.nl"}
    )
    print("Navigation Result:", result)
    
    # Test page info
    print("\n📄 Testing page info...")
    result = tool.forward(
        action="get_page_info",
        parameters={}
    )
    print("Page Info Result:", result)
    
    # Clean up
    tool.cleanup()
    print("\n✅ Test completed!")

def test_setup_function():
    print("\n🔧 Testing setup_agentic_browser function...")
    
    # Mock agent for testing
    class MockAgent:
        def __init__(self):
            self.tools = []
        
        def python_executor(self, code):
            print(f"Executing: {code}")
    
    mock_agent = MockAgent()
    
    # Test setup function
    result = setup_agentic_browser(mock_agent, headless=True)
    print("Setup Function Result:", result)
    print(f"Tools added to agent: {len(mock_agent.tools)}")

if __name__ == "__main__":
    test_actual_tool()
    test_setup_function()