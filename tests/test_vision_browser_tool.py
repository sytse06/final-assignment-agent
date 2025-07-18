# test_vision_browser_tool.py
import sys
import os
# Add tool class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tools'))

from vision_browser_tool import VisionWebBrowserTool

def test_actual_tool():
    print("🔧 Testing actual VisionWebBrowserTool...")
    
    tool = VisionWebBrowserTool()
    
    result = tool.forward(
        action="navigate",
        url="https://linkedin.com",
        wait_seconds=2
    )
    
    print("Result:", result)

if __name__ == "__main__":
    test_actual_tool()