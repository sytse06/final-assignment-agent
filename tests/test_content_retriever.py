# test_content_retriever.py
def test_content_retriever_safe():
    """Test ContentRetrieverTool with error handling"""
    from tools import ContentRetrieverTool
    
    tool = ContentRetrieverTool()
    
    # Test with invalid input (should return error string, not crash)
    result = tool.forward("nonexistent_file.pdf", "test query")
    
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    print(f"Has .strip(): {hasattr(result, 'strip')}")
    
    # Should be: type=str, has .strip()=True, result="Error: File not found..."
    
    return result

if __name__ == "__main__":
    print("🧪 Testing ContentRetrieverTool error handling...")
    print("=" * 50)
    
    try:
        result = test_content_retriever_safe()
        print("\n✅ Test completed successfully!")
        print(f"✅ Returns string: {isinstance(result, str)}")
        print(f"✅ Has .strip() method: {hasattr(result, 'strip')}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()