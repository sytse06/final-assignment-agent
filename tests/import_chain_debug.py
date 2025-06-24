#!/usr/bin/env python3
# import_chain_debug.py
# Debug what happens when importing GAIAAgent

import sys
import importlib.util

def trace_imports():
    """Trace what gets imported when we import GAIAAgent"""
    print("🔍 TRACING IMPORT CHAIN")
    print("=" * 25)
    
    # Check what agent_logic imports
    print("📄 agent_logic.py imports:")
    try:
        with open('agent_logic.py', 'r') as f:
            lines = f.readlines()
        
        imports = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('from ') or line.startswith('import '):
                if 'langchain_tools' in line or 'search_wikipedia' in line:
                    print(f"🎯 Line {i}: {line}")
                imports.append((i, line))
        
        print(f"\n📊 Found {len(imports)} import statements")
        
    except Exception as e:
        print(f"❌ Error reading agent_logic.py: {e}")

def check_imported_modules():
    """Check modules that agent_logic imports"""
    print("\n🔍 CHECKING IMPORTED MODULES")
    print("=" * 30)
    
    # Modules that agent_logic imports
    modules_to_check = [
        'agent_context',
        'dev_retriever', 
        'agent_logging',
        'tools',
        'agent_interface'  # If it exists
    ]
    
    for module_name in modules_to_check:
        if module_name == 'tools':
            # Check tools/__init__.py
            init_file = 'tools/__init__.py'
        else:
            init_file = f'{module_name}.py'
        
        print(f"\n📄 Checking {init_file}:")
        
        try:
            with open(init_file, 'r') as f:
                content = f.read()
            
            # Look for problematic patterns
            problem_lines = []
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                if any(pattern in line.lower() for pattern in [
                    'search_wikipedia', 
                    'research_tools',
                    'gaia tools status',
                    'total_research_tools'
                ]):
                    problem_lines.append((i, line.strip()))
            
            if problem_lines:
                print("🎯 Found potential issues:")
                for line_num, line_content in problem_lines:
                    print(f"   Line {line_num}: {line_content}")
            else:
                print("✅ No obvious issues found")
                
        except FileNotFoundError:
            print(f"⚠️  {init_file} not found")
        except Exception as e:
            print(f"❌ Error reading {init_file}: {e}")

def test_individual_imports():
    """Test importing each component individually"""
    print("\n🧪 TESTING INDIVIDUAL IMPORTS")
    print("=" * 32)
    
    test_imports = [
        "from tools.langchain_tools import ALL_LANGCHAIN_TOOLS",
        "from agent_context import ContextVariableFlow",
        "from dev_retriever import load_gaia_retriever", 
        "from agent_logging import AgentLoggingSetup",
        "from tools import GetAttachmentTool, ContentRetrieverTool"
    ]
    
    for import_stmt in test_imports:
        print(f"Testing: {import_stmt}")
        try:
            exec(import_stmt)
            print("✅ Success")
        except Exception as e:
            print(f"❌ Failed: {e}")
            if 'search_wikipedia' in str(e):
                print("🎯 This is likely the problematic import!")
        print()

if __name__ == "__main__":
    trace_imports()
    check_imported_modules()
    test_individual_imports()