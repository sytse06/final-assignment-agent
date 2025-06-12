#/tests/debug_metadata.py
"""
Debug script to analyze metadata.json structure

The script will analyze your metadata.json file and show you:

File structure: Whether it's a list, dict, or something else
Content preview: Sample of what's inside
Suggestions: How to fix any issues
Test results: Whether the fixes work

Run it: python debug_metadata.py /path/to/your/gaia_data
poetry run tests/debug_metadata
"""
import json
import os

def debug_metadata_structure(dataset_path: str = "./tests/gaia_data"):
    """Analyze the structure of your metadata.json file"""
    
    metadata_file = os.path.join(dataset_path, "metadata.json")
    
    print(f"🔍 Debugging Metadata Structure")
    print("=" * 40)
    print(f"Looking for: {metadata_file}")
    
    if not os.path.exists(metadata_file):
        print(f"❌ File not found: {metadata_file}")
        
        # Look for alternative files
        if os.path.exists(dataset_path):
            files = os.listdir(dataset_path)
            json_files = [f for f in files if f.endswith('.json')]
            print(f"📁 JSON files found: {json_files}")
        return False
    
    print(f"✅ File found: {metadata_file}")
    
    try:
        with open(metadata_file, 'r') as f:
            raw_data = json.load(f)
        
        print(f"\n📊 Metadata Analysis:")
        print(f"Root type: {type(raw_data)}")
        
        if isinstance(raw_data, dict):
            print(f"Dictionary keys: {list(raw_data.keys())}")
            
            # Check if it's a single question
            if 'task_id' in raw_data:
                print(f"✅ Single question format")
                print(f"Task ID: {raw_data.get('task_id')}")
                print(f"Question: {raw_data.get('Question', '')[:80]}...")
                return raw_data
            
            # Check for nested structure
            for key, value in raw_data.items():
                print(f"  {key}: {type(value)}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"    List length: {len(value)}")
                    print(f"    First item type: {type(value[0])}")
                    if isinstance(value[0], dict):
                        print(f"    First item keys: {list(value[0].keys())}")
        
        elif isinstance(raw_data, list):
            print(f"List length: {len(raw_data)}")
            if len(raw_data) > 0:
                print(f"First item type: {type(raw_data[0])}")
                if isinstance(raw_data[0], dict):
                    print(f"First item keys: {list(raw_data[0].keys())}")
                    
                    # Show sample question
                    sample = raw_data[0]
                    print(f"\n📝 Sample Question:")
                    for key, value in sample.items():
                        if isinstance(value, str) and len(value) > 80:
                            print(f"  {key}: {str(value)[:80]}...")
                        else:
                            print(f"  {key}: {value}")
        
        else:
            print(f"❌ Unexpected root type: {type(raw_data)}")
        
        return raw_data
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"💡 Check that {metadata_file} contains valid JSON")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def suggest_metadata_fix(raw_data):
    """Suggest how to fix metadata structure"""
    print(f"\n💡 Suggestions:")
    
    if isinstance(raw_data, dict) and 'task_id' in raw_data:
        print(f"✅ Your metadata is a single question - that's fine!")
        print(f"The GAIADatasetManager will handle this correctly.")
        
    elif isinstance(raw_data, list):
        print(f"✅ Your metadata is a list of questions - perfect!")
        
    elif isinstance(raw_data, dict):
        # Check for common patterns
        possible_questions_keys = ['questions', 'data', 'items', 'tasks']
        found_questions = False
        
        for key in possible_questions_keys:
            if key in raw_data and isinstance(raw_data[key], list):
                print(f"✅ Found questions under '{key}' - will handle automatically")
                found_questions = True
                break
        
        if not found_questions:
            print(f"⚠️  Structure not recognized. Expected:")
            print(f"   - List of questions: [{{'task_id': ..., 'Question': ...}}, ...]")
            print(f"   - Dict with questions: {{'questions': [...]}} or {{'data': [...]}}")
            print(f"   - Single question: {{'task_id': ..., 'Question': ...}}")

def quick_fix_test():
    """Quick test to see if the fix works"""
    print(f"\n🧪 Testing Fixed Code:")
    print("=" * 30)
    
    # Import the fixed function
    try:
        from gaia_dataset_utils import GAIADatasetManager
        
        # Test the manager
        manager = GAIADatasetManager("./tests/gaia_data")
        
        if manager.metadata:
            print(f"✅ Manager loaded successfully")
            print(f"Questions: {len(manager.metadata)}")
            print(f"File questions: {len(manager.file_questions)}")
            return True
        else:
            print(f"❌ Manager failed to load metadata")
            return False
            
    except Exception as e:
        print(f"❌ Import/test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "./tests/gaia_data"
    
    print(f"🔧 GAIA Metadata Debug Tool")
    print(f"Dataset path: {dataset_path}")
    
    # Debug the metadata structure
    raw_data = debug_metadata_structure(dataset_path)
    
    if raw_data:
        suggest_metadata_fix(raw_data)
        
        # Test the fix
        quick_fix_test()
    
    print(f"\n✅ Debug complete!")