# Testing Framework for GAIA Agents - Enhanced

## Overview

The GAIA Agent testing framework provides comprehensive validation across **development** and **production** scenarios, ensuring your agent works correctly for both arbitrary questions and formal GAIA benchmark evaluation.

---

## 🧪 Quick Development Testing (`test_gaia_agent.py`)

### **🎯 Purpose: Rapid Development Validation**

Interactive CLI-based testing for immediate feedback during development. Perfect for:
- **Component validation** - Verify agent creation and tool setup
- **Question routing testing** - Test smart routing with different question types  
- **Development debugging** - Isolate issues before full GAIA testing
- **Model provider comparison** - Quick A/B testing across providers
- **Custom question testing** - Test with your own questions outside GAIA scope

### **🛠️ CLI Interface**

```bash
# Basic usage - test with default question
poetry run python tests/test_gaia_agent.py

# Custom question testing
poetry run python tests/test_gaia_agent.py -q "Is Elon Musk still the CEO of Tesla?"

# Model provider testing
poetry run python tests/test_gaia_agent.py -c groq -q "What is 15% of 200?"
poetry run python tests/test_gaia_agent.py -c google -q "Current population of Netherlands"

# Development debugging
poetry run python tests/test_gaia_agent.py --skip-agent-test -v

# Verbose output for detailed debugging
poetry run python tests/test_gaia_agent.py -q "Your question" -v
```

### **📋 Command Line Arguments**

| Argument | Short | Options | Default | Description |
|----------|-------|---------|---------|-------------|
| `--question` | `-q` | Any string | "What is Mark Rutte doing right now?" | Question to test |
| `--config` | `-c` | `openrouter`, `groq`, `google` | `openrouter` | Model provider |
| `--verbose` | `-v` | Flag | False | Enable detailed output |
| `--skip-agent-test` | | Flag | False | Skip question processing (debug setup) |

### **🔍 Test Components**

1. **Context Bridge Test** - Validates thread-safe context management
2. **Tool Creation Test** - Verifies SmolagAgents Tool compliance
3. **Agent Creation Test** - Confirms multi-agent architecture setup
4. **Manager Delegation Test** - Validates agent coordination format
5. **Question Processing Test** - Full workflow with real question

### **✅ Expected Behaviors**

**Development Questions (No Files)**:
```bash
# These are CORRECT behaviors for arbitrary questions:
✅ GetAttachmentTool reports "no files" (expected - no GAIA files)
✅ Agent uses web research for current information
✅ Simple questions → one-shot answering
✅ Complex questions → manager coordination
✅ Context bridge passes task IDs correctly
```

**Success Indicators**:
- All 5 test components pass
- Agent processes questions without crashing
- Appropriate routing based on question complexity
- Context bridge maintains state across workflow

### **🎯 Development Test Examples**

```bash
# Test simple math (should use one-shot routing)
poetry run python tests/test_gaia_agent.py -q "What is 15% of 200?"

# Test current information (should use manager + web research)
poetry run python tests/test_gaia_agent.py -q "Who is the current president of France?"

# Test complex reasoning (should use manager coordination)
poetry run python tests/test_gaia_agent.py -q "Compare renewable energy adoption in EU vs US"

# Debug agent setup without processing questions
poetry run python tests/test_gaia_agent.py --skip-agent-test -v
```

---

## 📊 Dataset Utils (`gaia_dataset_utils.py`)

### **Functional Goals:**

- **Dataset Discovery & Validation**: Find and verify GAIA dataset integrity
- **Data Access & Organization**: Provide clean interfaces to question data and files
- **Test Batch Creation**: Generate strategically composed question sets for testing
- **Ground Truth Management**: Safely store and provide expected answers for evaluation
- **Data Export & Analysis**: Enable dataset inspection and batch reproducibility

### **🛠️ Core Methods:**

#### **Dataset Management**

```python
GAIADatasetManager(dataset_path)          # Load and validate GAIA dataset
._load_metadata()                         # Parse metadata.json structure  
._find_dataset_path()                     # Auto-discover dataset location
quick_dataset_check(path)                 # Validate dataset without agents
```

#### **Data Access**

```python
.get_question_by_id(task_id)             # Retrieve specific question data
.get_ground_truth(task_id)               # Get expected answer (evaluation only)
.find_file_for_question(task_id)         # Locate associated files
.get_questions_by_criteria(level=N, ...)  # Filter questions by attributes
```

#### **Test Batch Creation**

```python
.create_test_batch(size, strategy)       # Generate strategic question sets
._create_balanced_batch()                # Even distribution across levels/types
._create_small_sample_batch()            # Quick 5-question validation set
._create_large_comprehensive_batch()     # Thorough 25-question evaluation
._create_level_focused_batch()           # Target specific difficulty level
._create_file_diverse_batch()            # Maximize file type coverage
```

#### **Analysis & Validation**

```python
.analyze_dataset_distribution()          # Statistical breakdown of questions
.validate_answer_format(answer)          # Check GAIA formatting compliance
.get_file_types_distribution()           # Available file format analysis
.generate_dataset_report()               # Comprehensive dataset summary
```

#### **Export & Utilities**

```python
.export_test_batch(questions, path, fmt) # Save batches as JSON/CSV/JSONL
generate_test_batches(configs)           # Bulk batch generation
compare_dataset_versions(path1, path2)   # Dataset comparison analysis
```

**`gaia_dataset_utils.py` - Pure Data Layer**

- ✅ Dataset loading, validation, and file operations
- ✅ Test batch creation with multiple strategies
- ✅ Ground truth access and answer validation
- ❌ **NO** agent execution or evaluation logic

---

## 🔧 Debug Metadata (`/tests/debug_metadata.py`)

Debug script to analyze metadata.json structure

The script will analyze your metadata.json file and show you:

- **File structure**: Whether it's a list, dict, or something else
- **Content preview**: Sample of what's inside
- **Suggestions**: How to fix any issues
- **Test results**: Whether the fixes work

### **Sample Results**

```
🔧 GAIA Metadata Debug Tool
Dataset path: ./tests/gaia_data
🔍 Debugging Metadata Structure
========================================
Looking for: ./tests/gaia_data/metadata.json
✅ File found: ./tests/gaia_data/metadata.json
📊 Metadata Analysis:
Root type: <class 'dict'>
Dictionary keys: ['validation', 'test', 'stats']
 validation: <class 'list'>
 List length: 165
 First item type: <class 'dict'>
 First item keys: ['task_id', 'Question', 'Level', 'Final answer', 'file_name', 'file_path', 'Annotator Metadata']
 test: <class 'list'>
 List length: 301
 First item type: <class 'dict'>
 First item keys: ['task_id', 'Question', 'Level', 'Final answer', 'file_name', 'file_path', 'Annotator Metadata']
 stats: <class 'dict'>
```

---

## 🧪 Testing & Evaluation Layer (`agent_testing.py`)

### **🎯 Functional Goals:**

- **Agent Execution**: Run GAIA agents on question batches with comprehensive tracking
- **Proper Evaluation**: Compare agent responses against expected answers without contamination
- **Performance Analysis**: Understand agent behavior and identify improvements
- **Configuration Comparison**: Test different agent setups fairly
- **Result Management**: Generate reports and track progress

### **🔒 Two-Phase Testing System**

To ensure fair evaluation, the testing system separates execution from evaluation:

#### **Phase 1: Blind Execution**
The agent processes questions **without seeing expected answers**. This prevents the agent from accidentally using ground truth information.

```python
class GAIAQuestionExecutor:
    """Execute questions without access to expected answers"""
    
    def execute_questions_batch(self, blind_questions):
        # blind_questions only contain task_id, question, and level
        # NO expected answers included!
        
        for question in blind_questions:
            result = self.agent.process_question(question["Question"])
            # Agent never sees the expected answer
        
        return execution_file  # Saved for later evaluation
```

#### **Phase 2: Evaluation**
After execution is complete, results are compared against expected answers:

```python
class GAIAAnswerEvaluator:
    """Compare results against expected answers after execution"""
    
    def evaluate_execution_results(self, execution_file):
        # Load execution results
        # NOW compare against expected answers
        
        for result in execution_results:
            expected = dataset.get_ground_truth(result.task_id)
            is_correct = self.compare_answers(result.answer, expected)
        
        return evaluation_results
```

### **🔍 Why This Separation Matters**

**Problem**: If the agent can see expected answers during execution, results might be inflated
**Solution**: Keep expected answers completely separate until after execution

**This ensures:**
- ✅ Fair testing - agent can't cheat by seeing answers
- ✅ Honest results - performance reflects real capability
- ✅ Proper benchmarking - matches how GAIA testing should work
- ✅ Debugging clarity - execution issues separate from evaluation issues

### **🛠️ Core Methods:**

#### **Agent Execution**

```python
GAIAQuestionExecutor(agent_config)           # Initialize agent for testing
.execute_questions_batch(questions)          # Run agent on batch of questions
.execute_single_question(question_data)      # Execute individual question with tracking
._determine_strategy_used(result)            # Identify routing decisions made
```

#### **Evaluation & Analysis**

```python
GAIAAnswerEvaluator(dataset_manager)         # Initialize evaluator with ground truth access
.evaluate_execution_results(exec_results)    # Compare agent answers vs expected
.gaia_answer_matching(predicted, expected)   # GAIA-compliant answer comparison
.fuzzy_answer_matching(predicted, expected)  # Similarity-based matching
```

#### **High-Level Testing Workflows**

```python
run_gaia_test(config, dataset_path, ...)     # Complete test: execute→evaluate
run_quick_gaia_test(config, num_questions=5) # Quick 5-question validation
compare_agent_configs(configs)               # Multi-config comparison
run_smart_routing_test(config)               # Test routing effectiveness
```

#### **Result Management**

```python
analyze_failure_patterns(results)            # Identify improvement opportunities
._save_execution_results(results)            # Persist execution data
._save_evaluation_results(results)           # Persist evaluation outcomes
```

#### **Configuration Management**

```python
get_agent_config_by_name(name)               # Map config names to actual settings
GAIATestConfig()                             # Test execution parameters
```

---

## 🌟 Comprehensive Test Suite (`test_comprehensive.py`)

### **🎯 Purpose: Multi-Modal Testing**

Complete testing system that properly distinguishes between development and GAIA scenarios:

```bash
# Development mode - arbitrary questions (no files expected)
poetry run python tests/test_comprehensive.py development -q "Is Elon Musk CEO of Tesla?"

# GAIA benchmark mode - real GAIA tasks with files
poetry run python tests/test_comprehensive.py gaia -n 3

# Manual GAIA mode - specific task ID testing
poetry run python tests/test_comprehensive.py manual -t task_001 -q "Question about attached file"

# All modes - comprehensive validation
poetry run python tests/test_comprehensive.py all -q "Your question" -v
```

### **📋 Test Modes**

| Mode | Purpose | File Access | Expected Behavior |
|------|---------|-------------|-------------------|
| `development` | Arbitrary questions | ❌ No files | GetAttachmentTool "fails" (correct) |
| `gaia` | Real GAIA benchmark | ✅ Associated files | Full file processing workflow |
| `manual` | Specific GAIA task | ✅ If task exists | Tests specific scenarios |
| `all` | Complete validation | ✅ Mixed | Comprehensive testing |

### **🔍 Understanding Results**

**✅ Expected "Failures" (Actually Correct)**:
```bash
# Development mode
❌ GetAttachmentTool could not access files
→ CORRECT! Development questions don't have files

# GAIA mode without dataset
⚠️ GAIA testing infrastructure not available  
→ OK! Need to set up full GAIA dataset
```

**✅ Success Indicators**:
- Agent processes questions without crashing
- Appropriate routing (simple → one-shot, complex → manager)
- Context bridge maintains state
- Tools report expected behaviors for each mode

---

## 🔄 Complete Testing Workflow

### **Step 1: Dataset Preparation**
```python
from gaia_dataset_utils import GAIADatasetManager

# Load dataset and create test questions (without answers)
dataset_manager = GAIADatasetManager("./tests/gaia_data")
test_questions = dataset_manager.create_test_batch(15, "balanced")
# Result: Questions without expected answers
```

### **Step 2: Agent Execution**
```python
from agent_testing import GAIAQuestionExecutor

# Execute questions without seeing expected answers
executor = GAIAQuestionExecutor(get_agent_config_by_name("groq"))
execution_file = executor.execute_questions_batch(test_questions)
# Agent processes questions blindly
```

### **Step 3: Evaluation**
```python
from agent_testing import GAIAAnswerEvaluator

# Compare results against expected answers
evaluator = GAIAAnswerEvaluator(dataset_manager)
evaluation = evaluator.evaluate_execution_results(execution_file)
# Now we can see how well the agent did
```

### **Step 4: Analysis**
```python
from agent_testing import analyze_failure_patterns

# Understand what worked and what didn't
analysis = analyze_failure_patterns(evaluation)
print(f"Accuracy: {evaluation['overall_performance']['accuracy']:.1%}")
```

### **🎯 High-Level Functions**

#### **Complete Test**
```python
def run_gaia_test(agent_config_name="groq", max_questions=20):
    """Run complete execution and evaluation workflow"""
    
    # Step 1: Prepare questions (no answers)
    dataset_manager = GAIADatasetManager("./tests/gaia_data")
    test_questions = dataset_manager.create_test_batch(max_questions, "balanced")
    
    # Step 2: Execute blindly
    executor = GAIAQuestionExecutor(get_agent_config_by_name(agent_config_name))
    execution_file = executor.execute_questions_batch(test_questions)
    
    # Step 3: Evaluate
    evaluator = GAIAAnswerEvaluator(dataset_manager)
    results = evaluator.evaluate_execution_results(execution_file)
    
    return results
```

**Usage:**
```python
# Test your agent properly
results = run_gaia_test("groq", max_questions=20)
print(f"Your agent got {results['overall_performance']['accuracy']:.1%} correct")
```

#### **Quick Validation**
```python
# Quick test with 5 questions
quick_results = run_quick_gaia_test("groq", num_questions=5)
```

#### **Compare Configurations**
```python
# Test different setups fairly
comparison = compare_agent_configs(["groq", "google"], num_questions=10)
for config, perf in comparison.items():
    print(f"{config}: {perf['accuracy']:.1%}")
```

### **📊 Understanding Results**

**Performance Metrics:**
```python
{
    "overall_performance": {
        "total_questions": 20,
        "correct_answers": 12,
        "accuracy": 0.60,  # 60% correct
        "successful_executions": 19  # 19/20 ran without errors
    },
    "level_performance": {
        "1": {"accuracy": 0.75, "correct": 6, "total": 8},  # Easy questions
        "2": {"accuracy": 0.50, "correct": 4, "total": 8},  # Medium questions  
        "3": {"accuracy": 0.25, "correct": 1, "total": 4}   # Hard questions
    },
    "strategy_analysis": {
        "one_shot_llm": {"accuracy": 0.80, "total_questions": 5},      # Direct answers
        "manager_coordination": {"accuracy": 0.53, "total_questions": 15}  # Complex workflow
    }
}
```

**What This Tells You:**
- Your agent answers 60% of questions correctly
- It's better at easy questions (75%) than hard ones (25%)
- Simple direct answers work better than complex coordination
- 19 out of 20 questions executed without crashes

---

## 🎯 **Key Design Principles**

1. **Clear Separation**: Development testing vs GAIA testing
2. **Fair Evaluation**: Agent can't see answers during execution
3. **Provider Flexibility**: Test different models independently
4. **Real File Testing**: Actual GAIA file types (.xlsx, .png, .pdf)
5. **Practical Results**: Clear metrics for improvement
6. **Debugging Support**: Isolate issues at each step
7. **CLI-Driven Development**: Quick iteration with command-line testing
8. **Multi-Modal Validation**: Development vs GAIA vs Custom scenarios

---

## 🚀 **Complete Usage Flow**

### **1. Quick Development Validation**
```bash
# Start here - basic functionality test
poetry run python tests/test_gaia_agent.py -q "What is 2+2?"

# Test your questions
poetry run python tests/test_gaia_agent.py -q "Is climate change real?" -c groq -v
```

### **2. Component Testing**
```bash
# Debug setup issues
poetry run python tests/test_gaia_agent.py --skip-agent-test -v

# Test different providers
poetry run python tests/test_gaia_agent.py -c groq -q "Simple question"
poetry run python tests/test_gaia_agent.py -c google -q "Same question"
```

### **3. Comprehensive Validation**
```bash
# Multi-modal testing
poetry run python tests/test_comprehensive.py development -q "Your question"
poetry run python tests/test_comprehensive.py all -v
```

### **4. GAIA Benchmark Testing**
```bash
# Quick validation
python -c "
from agent_testing import run_quick_gaia_test
results = run_quick_gaia_test('groq', num_questions=5)
print(f'Accuracy: {results[\"overall_performance\"][\"accuracy\"]:.1%}')
"

# Full evaluation
python -c "
from agent_testing import run_gaia_test
results = run_gaia_test('groq', max_questions=20)
print(f'GAIA Score: {results[\"overall_performance\"][\"accuracy\"]:.1%}')
"

# Compare models
python -c "
from agent_testing import compare_agent_configs
comparison = compare_agent_configs(['groq', 'google'], num_questions=10)
for config, perf in comparison.items():
    print(f'{config}: {perf[\"accuracy\"]:.1%}')
"
```

---

## 🔑 **Key Insight: GetAttachmentTool "Errors" Are Expected**

**For Development Questions**: GetAttachmentTool reporting "no files" is **CORRECT BEHAVIOR**

- ✅ Development questions don't have GAIA files
- ✅ Tool correctly reports no associated files  
- ✅ Agent continues with web research/reasoning
- ✅ This validates your system works properly!

**The "error" is actually proof your agent correctly distinguishes between development testing and real GAIA tasks.**

---

## 📚 **Summary: Practical Testing System**

Your GAIA agent testing framework provides:

1. **🧪 Development Testing**: Quick iteration and debugging
2. **🔒 Fair Evaluation**: Proper separation of execution and evaluation
3. **📊 Clear Metrics**: Understand what's working and what isn't
4. **🔄 Easy Comparison**: Test different configurations fairly
5. **📈 Improvement Guidance**: Identify specific areas to work on
6. **🎯 GAIA Compatibility**: Test against the actual benchmark

This gives you a solid foundation for developing and improving your GAIA agent with confidence in your results. 🎯