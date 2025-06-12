# Testing Framework for GAIA Agents

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

**After refactoring:**
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
- **Blind Evaluation**: Compare agent responses against ground truth post-execution
- **Performance Analysis**: Deep-dive analysis of agent behavior and effectiveness
- **Configuration Comparison**: Systematic comparison of different agent setups
- **Result Management**: Generate detailed reports and actionable insights

### **🛠️ Core Methods:**

#### **Agent Execution**

```python
GAIATestExecutor(agent_config)           # Initialize agent for testing
.execute_test_batch(questions)           # Run agent on batch of questions
.execute_single_question(question_data)  # Execute individual question with tracking
._determine_strategy_used(result)        # Identify routing decisions made
```

#### **Evaluation & Analysis**

```python
GAIATestEvaluator(dataset_manager)       # Initialize evaluator with ground truth access
.evaluate_execution_results(exec_results) # Compare agent answers vs expected
.evaluate_single_answer(agent, task_id)  # Detailed single-answer analysis  
.gaia_answer_matching(predicted, expected) # GAIA-compliant answer comparison
.fuzzy_answer_matching(predicted, expected) # Similarity-based matching
```

#### **Performance Analysis**

```python
GAIATestAnalyzer()                       # Initialize analysis engine
.analyze_failure_patterns(results)       # Identify improvement opportunities
.analyze_routing_effectiveness(results)  # Evaluate smart routing decisions
.compare_agent_configurations(configs)   # Cross-configuration performance
._generate_improvement_recommendations() # Actionable optimization suggestions
```

#### **High-Level Testing Workflows**

```python
run_gaia_test(config, dataset_path, ...)  # Complete test: batch→execute→evaluate
run_small_batch_test(config)             # Quick 5-question validation
run_large_batch_test(config)             # Comprehensive 25-question evaluation  
run_agent_comparison_study(configs)      # Multi-config head-to-head comparison
```

#### **Result Management**

```python
generate_test_report(results, config)    # Comprehensive performance report
analyze_test_results(evaluation)         # Extract insights and patterns
._save_execution_results(results)        # Persist execution data
._save_evaluation_results(results)       # Persist evaluation outcomes
```

#### **Configuration Management**

```python
get_agent_config_by_name(name)          # Map config names to actual settings
GAIATestConfig()                         # Test execution parameters
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

## 🔄 Functional Interaction Pattern

### **Data Preparation (gaia_dataset_utils)**

```python
# 1. Load and validate dataset
manager = GAIADatasetManager("./tests/gaia_data")

# 2. Create strategic test batch 
test_batch = manager.create_test_batch(15, "balanced")
# → Returns blind questions (no ground truth)
```

### **🔒 Blind Testing Strategy**

The blind testing strategy ensures rigorous evaluation:

#### **🎯 What Blind Testing Means**

1. **During execution**: The agent never sees expected answers
2. **During evaluation**: Results compared against ground truth AFTER execution
3. **No contamination**: Ground truth doesn't influence agent responses

#### **🔍 How Architecture Preserves Blind Testing**

**1. Stricter Separation**
```python
# Data Layer - NEVER exposes ground truth to execution
def create_test_batch(self, size, strategy):
    blind_questions = [{
        'task_id': q['task_id'],
        'Question': q['Question'],
        'Level': q['Level']
        # 'Final answer' NEVER included
    } for q in selected]
```

**2. Clear Data Flow**
```
📊 Dataset Manager → 🔒 Blind Batch → 🤖 Agent Execution → 📝 Results
                                                              ↓
📊 Dataset Manager ← 🎯 Evaluation ← 🔓 Ground Truth Added ←──┘
```

**3. Interface-Level Protection**
```python
# Execution Phase - CAN'T access ground truth
executor = GAIATestExecutor("groq")
execution_results = executor.execute_test_batch(blind_questions)

# Evaluation Phase - Gets ground truth ONLY after execution
evaluator = GAIATestEvaluator(dataset_manager)
evaluation = evaluator.evaluate_execution_results(execution_results)
```

---

## 📓 **Agent Builder Notebook** - Development Focus

✅ **Dependency checking** - Reveals all import requirements and system setup
✅ **Component isolation** - Tests each piece independently
✅ **Model initialization** - Validates all provider configurations
✅ **Workflow validation** - Tests LangGraph routing and complexity detection
✅ **GAIA file type testing** - Real file processing validation (.xlsx, .png, .pdf, etc.)
✅ **Mock data testing** - Synthetic validation before real GAIA data
✅ **Development readiness score** - Clear go/no-go for production testing

---

## 📊 **Production Validator Notebook** - Performance Focus

✅ **Testing framework exploration** - Learn the comprehensive testing system
✅ **Performance baselines** - Establish metrics across question types
✅ **Smart routing analysis** - Deep dive into routing effectiveness
✅ **Provider-specific testing** - Groq, OpenRouter, Ollama independent evaluation
✅ **Interactive visualization** - One example with charts, then direct data access
✅ **Failure pattern analysis** - Systematic improvement insights
✅ **Production readiness assessment** - Data-driven deployment recommendation
✅ **Comprehensive reporting** - Professional documentation and data export

---

## 🎯 **Key Design Principles Achieved**

1. **Clear Separation**: Building vs Production testing
2. **Dependency Transparency**: Full import and system requirements revealed
3. **Provider Flexibility**: Independent testing of Groq, OpenRouter, Ollama
4. **Real File Testing**: Actual GAIA file types (.xlsx, .png, .pdf)
5. **Data Access**: Direct access to source data after visualization example
6. **Manual Configuration**: No automatic tuning - you maintain control
7. **Comprehensive Coverage**: From component testing to production readiness
8. **CLI-Driven Development**: Quick iteration with command-line testing
9. **Multi-Modal Validation**: Development vs GAIA vs Custom scenarios

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

### **4. Production Testing**
- **Use Agent Builder Notebook** - Validate all components work
- **Fix any issues** found in development testing
- **Move to Production Validator** - Real GAIA performance analysis
- **Use failure analysis** to iterate and improve
- **Generate final report** for production deployment decision

### **5. GAIA Benchmark Testing**
```bash
# If you have GAIA dataset
poetry run python tests/test_comprehensive.py gaia -n 5 -c groq
```

---

## 🔑 **Key Insight: GetAttachmentTool "Errors" Are Expected**

**For Development Questions**: GetAttachmentTool reporting "no files" is **CORRECT BEHAVIOR**

- ✅ Development questions don't have GAIA files
- ✅ Tool correctly reports no associated files  
- ✅ Agent continues with web research/reasoning
- ✅ This validates your system works properly!

**The "error" is actually proof your agent correctly distinguishes between development testing and real GAIA tasks.**