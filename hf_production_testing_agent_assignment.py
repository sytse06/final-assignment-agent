# hf_production_testing_agent_assignment.py
# Comprehensive testing framework tailored to ready GAIA agent system for deployment

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import time
import traceback
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Import your production system
from gaia_agent_system import (
    create_gaia_agent, 
    create_production_gaia_agent,
    GAIAConfig,
    ModelConfigs,
    run_gaia_benchmark
)

# ============================================================================
# CONFIGURATION FOR TESTING
# ============================================================================

@dataclass
class TestConfig:
    """Configuration for comprehensive GAIA testing"""
    # Test execution settings
    max_questions_per_config: int = 50
    max_total_budget: float = 10.0
    estimated_cost_per_question: float = 0.05
    timeout_per_question: int = 180
    
    # Test strategies
    enable_model_comparison: bool = True
    enable_level_analysis: bool = True
    enable_error_analysis: bool = True
    enable_performance_tracking: bool = True
    
    # Output settings
    results_dir: str = "test_results"
    save_detailed_logs: bool = True
    generate_visualizations: bool = True
    
    # Safety settings
    enable_budget_protection: bool = True
    max_errors_before_abort: int = 10
    enable_graceful_degradation: bool = True

# ============================================================================
# COMPREHENSIVE GAIA TESTING FRAMEWORK
# ============================================================================

class GAIAProductionTestFramework:
    """Production-grade testing framework for your GAIA agent system"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.test_results = []
        self.budget_tracker = BudgetTracker(self.config.max_total_budget)
        self.error_tracker = ErrorTracker()
        self.performance_metrics = PerformanceMetrics()
        
        # Test session metadata
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_start = datetime.now()
        
        print(f"🧪 GAIA Production Testing Framework Initialized")
        print(f"📊 Session ID: {self.session_id}")
        print(f"💰 Budget: ${self.config.max_total_budget}")
        print(f"📁 Results: {self.results_dir}")
    
    def run_comprehensive_evaluation(self, 
                                   configs_to_test: List[str] = None,
                                   sample_size: int = None) -> Dict:
        """Run comprehensive evaluation across multiple configurations"""
        
        if configs_to_test is None:
            # Select representative configs from each provider
            configs_to_test = [
                "qwen2.5_coder",      # OpenRouter - High performance
                "qwen_qwq_groq",      # Groq - Fast execution  
                "gemini_flash_04",    # Google - Balanced
                "deepseek"            # OpenRouter - Alternative
            ]
        
        if sample_size is None:
            # Calculate optimal sample size based on budget
            estimated_cost = len(configs_to_test) * self.config.estimated_cost_per_question
            sample_size = min(
                self.config.max_questions_per_config,
                int(self.config.max_total_budget / estimated_cost)
            )
        
        print(f"🚀 COMPREHENSIVE GAIA EVALUATION")
        print("=" * 60)
        print(f"Configurations: {len(configs_to_test)}")
        print(f"Sample size per config: {sample_size}")
        print(f"Estimated total cost: ${len(configs_to_test) * sample_size * self.config.estimated_cost_per_question:.2f}")
        print(f"Budget protection: {'✅ Enabled' if self.config.enable_budget_protection else '❌ Disabled'}")
        
        evaluation_results = {}
        
        for i, config_name in enumerate(configs_to_test):
            if not self.budget_tracker.can_continue():
                print(f"⚠️  Budget exhausted, stopping at config {i}/{len(configs_to_test)}")
                break
                
            print(f"\n🔄 Testing configuration {i+1}/{len(configs_to_test)}: {config_name}")
            print("-" * 40)
            
            try:
                config_result = self.test_single_configuration(
                    config_name=config_name,
                    sample_size=sample_size
                )
                evaluation_results[config_name] = config_result
                
                # Print immediate results
                accuracy = config_result.get('accuracy', 0)
                avg_time = config_result.get('avg_execution_time', 0)
                cost = config_result.get('total_cost', 0)
                
                print(f"✅ {config_name} completed:")
                print(f"  ├── Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                print(f"  ├── Avg time: {avg_time:.2f}s")
                print(f"  ├── Cost: ${cost:.3f}")
                print(f"  └── Budget remaining: ${self.budget_tracker.remaining:.3f}")
                
            except Exception as e:
                print(f"❌ Configuration {config_name} failed: {e}")
                self.error_tracker.record_error(config_name, str(e))
                evaluation_results[config_name] = {"error": str(e)}
                
                if self.error_tracker.should_abort():
                    print(f"⚠️  Too many errors ({self.error_tracker.error_count}), aborting evaluation")
                    break
        
        # Generate comprehensive analysis
        analysis = self.generate_comprehensive_analysis(evaluation_results)
        
        # Save results
        self.save_evaluation_results(evaluation_results, analysis)
        
        return {
            "evaluation_results": evaluation_results,
            "analysis": analysis,
            "session_metadata": self.get_session_metadata()
        }
    
    def test_single_configuration(self, config_name: str, sample_size: int) -> Dict:
        """Test single model configuration with detailed metrics"""
        
        start_time = datetime.now()
        
        try:
            # Create agent with specific configuration
            agent = create_production_gaia_agent(
                model_config=config_name,
                enable_logging=True,
                performance_tracking=True,
                max_retries=2
            )
            
            # Get test sample with level distribution
            test_sample = self.get_balanced_test_sample(sample_size)
            
            results = []
            successful_questions = 0
            total_cost = 0.0
            
            print(f"  📝 Processing {len(test_sample)} questions...")
            
            for i, question_data in enumerate(test_sample):
                if not self.budget_tracker.can_continue():
                    print(f"    💰 Budget limit reached at question {i+1}")
                    break
                
                question_start = time.time()
                
                try:
                    # Execute single question
                    result = agent.run_single_question(
                        question=question_data.get("Question", ""),
                        task_id=question_data.get("task_id", f"{config_name}_{i}"),
                        ground_truth=question_data.get("Final answer", ""),
                        level=question_data.get("Level", 1)
                    )
                    
                    question_time = time.time() - question_start
                    question_cost = self.config.estimated_cost_per_question
                    
                    # Track budget
                    self.budget_tracker.spend(question_cost)
                    total_cost += question_cost
                    
                    # Process result
                    is_correct = self.evaluate_answer_correctness(
                        result.get("final_answer", ""),
                        question_data.get("Final answer", "")
                    )
                    
                    if is_correct:
                        successful_questions += 1
                    
                    # Store detailed result
                    detailed_result = {
                        "config_name": config_name,
                        "question_id": i,
                        "task_id": question_data.get("task_id", f"{config_name}_{i}"),
                        "question": question_data.get("Question", ""),
                        "ground_truth": question_data.get("Final answer", ""),
                        "level": question_data.get("Level", 1),
                        "predicted_answer": result.get("final_answer", ""),
                        "is_correct": is_correct,
                        "execution_time": question_time,
                        "cost": question_cost,
                        "strategy_used": result.get("selected_strategy", ""),
                        "agent_used": result.get("selected_agent", ""),
                        "confidence_score": result.get("confidence_score", 0.0),
                        "errors": result.get("errors", []),
                        "fallback_used": result.get("fallback_used", False),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    results.append(detailed_result)
                    self.test_results.append(detailed_result)
                    
                    # Progress indicator
                    if (i + 1) % 10 == 0:
                        current_accuracy = successful_questions / (i + 1)
                        print(f"    📈 Progress: {i+1}/{len(test_sample)} | Accuracy: {current_accuracy:.3f} | Budget: ${self.budget_tracker.remaining:.2f}")
                
                except Exception as e:
                    error_msg = f"Question {i} failed: {str(e)}"
                    print(f"    ❌ {error_msg}")
                    
                    self.error_tracker.record_error(config_name, error_msg)
                    
                    # Record failed result
                    failed_result = {
                        "config_name": config_name,
                        "question_id": i,
                        "question": question_data.get("Question", ""),
                        "error": error_msg,
                        "is_correct": False,
                        "execution_time": time.time() - question_start,
                        "cost": 0.0,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    results.append(failed_result)
                    self.test_results.append(failed_result)
            
            # Clean up agent
            agent.close()
            
            # Calculate metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            
            config_metrics = {
                "config_name": config_name,
                "total_questions": len(results),
                "successful_questions": successful_questions,
                "accuracy": successful_questions / len(results) if results else 0,
                "total_cost": total_cost,
                "avg_cost_per_question": total_cost / len(results) if results else 0,
                "total_execution_time": execution_time,
                "avg_execution_time": np.mean([r.get("execution_time", 0) for r in results]) if results else 0,
                "error_count": len([r for r in results if r.get("error")]),
                "fallback_usage": len([r for r in results if r.get("fallback_used", False)]) / len(results) if results else 0,
                "level_breakdown": self.analyze_level_performance(results),
                "strategy_breakdown": self.analyze_strategy_performance(results),
                "detailed_results": results
            }
            
            return config_metrics
            
        except Exception as e:
            error_msg = f"Configuration {config_name} failed: {str(e)}"
            print(f"    ❌ {error_msg}")
            self.error_tracker.record_error(config_name, error_msg)
            
            return {
                "config_name": config_name,
                "error": error_msg,
                "total_questions": 0,
                "accuracy": 0.0,
                "total_cost": 0.0
            }
    
    def get_balanced_test_sample(self, sample_size: int) -> List[Dict]:
        """Get balanced test sample maintaining GAIA level distribution"""
        
        # Load metadata
        try:
            with open("metadata.jsonl", 'r', encoding='utf-8') as f:
                all_metadata = [json.loads(line) for line in f]
        except FileNotFoundError:
            print("⚠️  metadata.jsonl not found, using simulated data")
            return self.generate_simulated_test_data(sample_size)
        
        # Group by level
        level_groups = defaultdict(list)
        for item in all_metadata:
            level = item.get('Level', 1)
            level_groups[level].append(item)
        
        # Maintain proportional distribution
        sample = []
        total_items = len(all_metadata)
        
        for level in sorted(level_groups.keys()):
            level_items = level_groups[level]
            level_proportion = len(level_items) / total_items
            level_sample_size = max(1, int(sample_size * level_proportion))
            
            # Take sample (or all if smaller)
            level_sample = level_items[:min(level_sample_size, len(level_items))]
            sample.extend(level_sample)
            
            if len(sample) >= sample_size:
                break
        
        return sample[:sample_size]
    
    def generate_simulated_test_data(self, sample_size: int) -> List[Dict]:
        """Generate simulated test data if metadata not available"""
        
        simulated_data = []
        
        # Realistic GAIA question patterns
        question_templates = {
            1: [
                "Calculate {num1}% of {num2}",
                "What is {num1} + {num2}?",
                "Convert {num1} {unit1} to {unit2}",
                "Find the area of a circle with radius {num1}m"
            ],
            2: [
                "According to the document, what is the population of {city}?",
                "Calculate the compound interest on ${num1} at {num2}% for {num3} years",
                "What percentage of {category} are represented in the data?",
                "Compare the values in column A and B of the spreadsheet"
            ],
            3: [
                "Analyze the correlation between variables X and Y in the dataset",
                "What conclusions can be drawn from the statistical analysis?",
                "Integrate multiple data sources to determine the final result",
                "Perform multi-step analysis involving calculations and research"
            ]
        }
        
        for i in range(sample_size):
            # Distribute across levels (60% L1, 30% L2, 10% L3)
            if i < sample_size * 0.6:
                level = 1
            elif i < sample_size * 0.9:
                level = 2
            else:
                level = 3
            
            # Generate realistic question
            template = np.random.choice(question_templates[level])
            question = template.format(
                num1=np.random.randint(10, 1000),
                num2=np.random.randint(10, 1000),
                num3=np.random.randint(1, 10),
                city=np.random.choice(["Tokyo", "London", "New York", "Paris"]),
                unit1=np.random.choice(["meters", "feet", "kilometers"]),
                unit2=np.random.choice(["feet", "meters", "miles"]),
                category=np.random.choice(["companies", "people", "countries"])
            )
            
            # Generate realistic answer
            if level == 1:
                answer = str(np.random.randint(1, 1000))
            elif level == 2:
                answer = np.random.choice(["42", "156", "23.5", "New York", "15%"])
            else:
                answer = np.random.choice(["positive correlation", "no significant difference", "inconclusive"])
            
            simulated_data.append({
                "task_id": f"sim_{i}",
                "Question": question,
                "Final answer": answer,
                "Level": level
            })
        
        print(f"🔧 Generated {len(simulated_data)} simulated test questions")
        return simulated_data
    
    def evaluate_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Evaluate answer correctness with GAIA-style matching"""
        
        if not predicted or not ground_truth:
            return False
        
        # Normalize answers
        pred_clean = str(predicted).lower().strip()
        truth_clean = str(ground_truth).lower().strip()
        
        # Remove common artifacts
        for artifact in ['.', ',', '!', '?', '"', "'", 'the ', 'a ', 'an ']:
            pred_clean = pred_clean.replace(artifact, '')
            truth_clean = truth_clean.replace(artifact, '')
        
        # Exact match
        if pred_clean == truth_clean:
            return True
        
        # Numeric comparison
        try:
            pred_num = float(pred_clean.replace(',', ''))
            truth_num = float(truth_clean.replace(',', ''))
            return abs(pred_num - truth_num) < 1e-6
        except ValueError:
            pass
        
        # Fuzzy string matching for partial credit
        if len(truth_clean) > 3:
            # Check if prediction contains the key part of the truth
            truth_words = truth_clean.split()
            pred_words = pred_clean.split()
            
            # If main words match
            common_words = set(truth_words) & set(pred_words)
            if len(common_words) >= max(1, len(truth_words) * 0.7):
                return True
        
        return False
    
    def analyze_level_performance(self, results: List[Dict]) -> Dict:
        """Analyze performance by GAIA level"""
        
        level_stats = defaultdict(lambda: {"total": 0, "correct": 0, "avg_time": 0.0})
        
        for result in results:
            level = result.get("level", 1)
            level_stats[level]["total"] += 1
            
            if result.get("is_correct", False):
                level_stats[level]["correct"] += 1
            
            exec_time = result.get("execution_time", 0)
            if exec_time > 0:
                level_stats[level]["avg_time"] = (
                    level_stats[level]["avg_time"] * (level_stats[level]["total"] - 1) + exec_time
                ) / level_stats[level]["total"]
        
        # Calculate accuracy rates
        performance = {}
        for level, stats in level_stats.items():
            performance[f"level_{level}"] = {
                "total_questions": stats["total"],
                "correct_answers": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "avg_execution_time": stats["avg_time"]
            }
        
        return performance
    
    def analyze_strategy_performance(self, results: List[Dict]) -> Dict:
        """Analyze performance by strategy used"""
        
        strategy_stats = defaultdict(lambda: {"total": 0, "correct": 0, "avg_time": 0.0})
        
        for result in results:
            strategy = result.get("strategy_used", "unknown")
            strategy_stats[strategy]["total"] += 1
            
            if result.get("is_correct", False):
                strategy_stats[strategy]["correct"] += 1
            
            exec_time = result.get("execution_time", 0)
            if exec_time > 0:
                strategy_stats[strategy]["avg_time"] = (
                    strategy_stats[strategy]["avg_time"] * (strategy_stats[strategy]["total"] - 1) + exec_time
                ) / strategy_stats[strategy]["total"]
        
        # Calculate performance metrics
        performance = {}
        for strategy, stats in strategy_stats.items():
            performance[strategy] = {
                "total_questions": stats["total"],
                "correct_answers": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "avg_execution_time": stats["avg_time"]
            }
        
        return performance
    
    def generate_comprehensive_analysis(self, evaluation_results: Dict) -> Dict:
        """Generate comprehensive analysis across all tested configurations"""
        
        analysis = {
            "summary": {},
            "model_comparison": {},
            "performance_insights": {},
            "recommendations": [],
            "gaia_compliance": {},
            "budget_analysis": {}
        }
        
        # Summary statistics
        all_results = []
        for config_name, config_result in evaluation_results.items():
            if "error" not in config_result:
                all_results.extend(config_result.get("detailed_results", []))
        
        if all_results:
            analysis["summary"] = {
                "total_questions_tested": len(all_results),
                "overall_accuracy": np.mean([r.get("is_correct", False) for r in all_results]),
                "average_execution_time": np.mean([r.get("execution_time", 0) for r in all_results]),
                "total_cost": sum([r.get("cost", 0) for r in all_results]),
                "error_rate": len([r for r in all_results if r.get("error")]) / len(all_results),
                "fallback_usage_rate": len([r for r in all_results if r.get("fallback_used", False)]) / len(all_results)
            }
        
        # Model comparison
        model_performance = {}
        for config_name, config_result in evaluation_results.items():
            if "error" not in config_result:
                model_performance[config_name] = {
                    "accuracy": config_result.get("accuracy", 0),
                    "avg_execution_time": config_result.get("avg_execution_time", 0),
                    "cost_efficiency": config_result.get("accuracy", 0) / config_result.get("avg_cost_per_question", 1),
                    "error_rate": config_result.get("error_count", 0) / config_result.get("total_questions", 1),
                    "fallback_rate": config_result.get("fallback_usage", 0)
                }
        
        analysis["model_comparison"] = model_performance
        
        # Performance insights
        if model_performance:
            best_accuracy = max(model_performance.items(), key=lambda x: x[1]["accuracy"])
            fastest_model = min(model_performance.items(), key=lambda x: x[1]["avg_execution_time"])
            most_cost_efficient = max(model_performance.items(), key=lambda x: x[1]["cost_efficiency"])
            
            analysis["performance_insights"] = {
                "best_accuracy": {
                    "model": best_accuracy[0],
                    "accuracy": best_accuracy[1]["accuracy"]
                },
                "fastest_model": {
                    "model": fastest_model[0],
                    "avg_time": fastest_model[1]["avg_execution_time"]
                },
                "most_cost_efficient": {
                    "model": most_cost_efficient[0],
                    "efficiency_score": most_cost_efficient[1]["cost_efficiency"]
                }
            }
        
        # Generate recommendations
        recommendations = []
        
        if analysis["summary"].get("overall_accuracy", 0) > 0.45:
            recommendations.append("✅ Overall accuracy meets GAIA target (>45%)")
        else:
            recommendations.append("⚠️  Consider improving accuracy through better prompt engineering or model selection")
        
        if analysis["summary"].get("average_execution_time", 0) < 30:
            recommendations.append("✅ Response times are acceptable (<30s average)")
        else:
            recommendations.append("⚠️  Consider optimizing for faster response times")
        
        if analysis["summary"].get("total_cost", 0) < self.config.max_total_budget:
            recommendations.append("✅ Testing completed within budget constraints")
        else:
            recommendations.append("⚠️  Budget exceeded - consider cost optimization")
        
        analysis["recommendations"] = recommendations
        
        # GAIA compliance analysis
        analysis["gaia_compliance"] = {
            "target_accuracy": 0.45,
            "achieved_accuracy": analysis["summary"].get("overall_accuracy", 0),
            "meets_target": analysis["summary"].get("overall_accuracy", 0) >= 0.45,
            "level_performance_distribution": self.analyze_overall_level_performance(all_results)
        }
        
        # Budget analysis
        analysis["budget_analysis"] = {
            "budget_allocated": self.config.max_total_budget,
            "budget_used": self.budget_tracker.spent,
            "budget_remaining": self.budget_tracker.remaining,
            "cost_per_question": self.budget_tracker.spent / len(all_results) if all_results else 0,
            "efficiency_score": analysis["summary"].get("overall_accuracy", 0) / (self.budget_tracker.spent / len(all_results)) if all_results and self.budget_tracker.spent > 0 else 0
        }
        
        return analysis
    
    def analyze_overall_level_performance(self, all_results: List[Dict]) -> Dict:
        """Analyze performance across all GAIA levels"""
        
        level_performance = defaultdict(lambda: {"total": 0, "correct": 0})
        
        for result in all_results:
            level = result.get("level", 1)
            level_performance[level]["total"] += 1
            if result.get("is_correct", False):
                level_performance[level]["correct"] += 1
        
        performance = {}
        for level, stats in level_performance.items():
            performance[f"level_{level}"] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "total_questions": stats["total"]
            }
        
        return performance
    
    def save_evaluation_results(self, evaluation_results: Dict, analysis: Dict):
        """Save comprehensive evaluation results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "evaluation_results": evaluation_results,
                "analysis": analysis,
                "session_metadata": self.get_session_metadata()
            }, f, indent=2, default=str)
        
        # Save detailed CSV for analysis
        if self.test_results:
            df = pd.DataFrame(self.test_results)
            csv_file = self.results_dir / f"detailed_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
        
        # Generate visualizations
        if self.config.generate_visualizations and self.test_results:
            self.generate_visualizations(timestamp)
        
        print(f"\n💾 Results saved:")
        print(f"  ├── Main results: {results_file}")
        if self.test_results:
            print(f"  ├── Detailed CSV: {csv_file}")
        if self.config.generate_visualizations:
            print(f"  └── Visualizations: {self.results_dir}/plots_{timestamp}/")
    
    def generate_visualizations(self, timestamp: str):
        """Generate comprehensive visualizations"""
        
        plots_dir = self.results_dir / f"plots_{timestamp}"
        plots_dir.mkdir(exist_ok=True)
        
        df = pd.DataFrame(self.test_results)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Accuracy by Configuration
        fig, ax = plt.subplots(figsize=(12, 6))
        config_accuracy = df.groupby('config_name')['is_correct'].mean().sort_values(ascending=False)
        bars = ax.bar(range(len(config_accuracy)), config_accuracy.values)
        ax.set_xticks(range(len(config_accuracy)))
        ax.set_xticklabels(config_accuracy.index, rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Model Configuration')
        ax.axhline(y=0.45, color='red', linestyle='--', label='GAIA Target (45%)')
        ax.legend()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'accuracy_by_config.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance by GAIA Level
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Level accuracy
        level_accuracy = df.groupby('level')['is_correct'].mean()
        ax1.bar(level_accuracy.index, level_accuracy.values)
        ax1.set_xlabel('GAIA Level')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy by GAIA Level')
        for i, v in enumerate(level_accuracy.values):
            ax1.text(level_accuracy.index[i], v + 0.01, f'{v:.3f}', ha='center')
        
        # Level distribution
        level_counts = df['level'].value_counts().sort_index()
        ax2.pie(level_counts.values, labels=[f'Level {i}' for i in level_counts.index], 
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribution of Test Questions by Level')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'level_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Execution Time Analysis
        fig, ax = plt.subplots(figsize=(12, 6))
        config_times = df.groupby('config_name')['execution_time'].mean().sort_values()
        bars = ax.bar(range(len(config_times)), config_times.values)
        ax.set_xticks(range(len(config_times)))
        ax.set_xticklabels(config_times.index, rotation=45, ha='right')
        ax.set_ylabel('Average Execution Time (seconds)')
        ax.set_title('Average Execution Time by Configuration')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'execution_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Strategy Performance Analysis
        if 'strategy_used' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            strategy_performance = df.groupby('strategy_used')['is_correct'].agg(['mean', 'count'])
            
            bars = ax.bar(range(len(strategy_performance)), strategy_performance['mean'])
            ax.set_xticks(range(len(strategy_performance)))
            ax.set_xticklabels(strategy_performance.index, rotation=45, ha='right')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy by Execution Strategy')
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, strategy_performance['count'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}\n(n={count})', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'strategy_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Cost vs Performance Analysis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot of cost vs accuracy for each config
        config_stats = df.groupby('config_name').agg({
            'is_correct': 'mean',
            'cost': 'sum',
            'execution_time': 'mean'
        })
        
        scatter = ax.scatter(config_stats['cost'], config_stats['is_correct'], 
                           s=config_stats['execution_time']*10, alpha=0.6)
        
        # Add labels for each point
        for config, row in config_stats.iterrows():
            ax.annotate(config, (row['cost'], row['is_correct']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Total Cost ($)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Cost vs Performance Analysis\n(Bubble size = Average Execution Time)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'cost_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Error Analysis
        if 'errors' in df.columns:
            error_counts = df['errors'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            config_errors = df.groupby('config_name').apply(
                lambda x: (x['errors'].apply(lambda y: len(y) if isinstance(y, list) else 0) > 0).sum()
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(config_errors)), config_errors.values)
            ax.set_xticks(range(len(config_errors)))
            ax.set_xticklabels(config_errors.index, rotation=45, ha='right')
            ax.set_ylabel('Number of Questions with Errors')
            ax.set_title('Error Count by Configuration')
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Generated {len([f for f in plots_dir.iterdir() if f.suffix == '.png'])} visualization plots")
    
    def get_session_metadata(self) -> Dict:
        """Get comprehensive session metadata"""
        
        session_duration = datetime.now() - self.session_start
        
        return {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_duration_seconds": session_duration.total_seconds(),
            "total_questions_tested": len(self.test_results),
            "total_budget_allocated": self.config.max_total_budget,
            "total_budget_spent": self.budget_tracker.spent,
            "total_errors_encountered": self.error_tracker.error_count,
            "test_configuration": {
                "max_questions_per_config": self.config.max_questions_per_config,
                "estimated_cost_per_question": self.config.estimated_cost_per_question,
                "timeout_per_question": self.config.timeout_per_question,
                "budget_protection_enabled": self.config.enable_budget_protection
            },
            "system_info": {
                "framework_version": "production_v1.0",
                "gaia_target_accuracy": 0.45,
                "test_environment": "development"
            }
        }
    
    def generate_executive_summary(self, analysis: Dict) -> str:
        """Generate executive summary for stakeholders"""
        
        summary = analysis.get("summary", {})
        insights = analysis.get("performance_insights", {})
        compliance = analysis.get("gaia_compliance", {})
        budget = analysis.get("budget_analysis", {})
        
        report = f"""
🎯 GAIA AGENT EVALUATION - EXECUTIVE SUMMARY
{'='*60}

📊 OVERALL PERFORMANCE
├── Questions Tested: {summary.get('total_questions_tested', 0)}
├── Overall Accuracy: {summary.get('overall_accuracy', 0):.1%}
├── GAIA Target (45%): {'✅ ACHIEVED' if compliance.get('meets_target', False) else '❌ NOT MET'}
├── Average Response Time: {summary.get('average_execution_time', 0):.2f}s
└── Error Rate: {summary.get('error_rate', 0):.1%}

💰 BUDGET PERFORMANCE
├── Budget Allocated: ${budget.get('budget_allocated', 0):.2f}
├── Budget Used: ${budget.get('budget_used', 0):.2f}
├── Budget Remaining: ${budget.get('budget_remaining', 0):.2f}
├── Cost per Question: ${budget.get('cost_per_question', 0):.3f}
└── Efficiency Score: {budget.get('efficiency_score', 0):.3f}

🏆 TOP PERFORMERS
├── Best Accuracy: {insights.get('best_accuracy', {}).get('model', 'N/A')} ({insights.get('best_accuracy', {}).get('accuracy', 0):.1%})
├── Fastest Model: {insights.get('fastest_model', {}).get('model', 'N/A')} ({insights.get('fastest_model', {}).get('avg_time', 0):.2f}s)
└── Most Cost-Efficient: {insights.get('most_cost_efficient', {}).get('model', 'N/A')}

📈 GAIA LEVEL PERFORMANCE
"""
        
        level_perf = compliance.get('level_performance_distribution', {})
        for level_key, perf in level_perf.items():
            level_num = level_key.replace('level_', '')
            accuracy = perf.get('accuracy', 0)
            total = perf.get('total_questions', 0)
            report += f"├── Level {level_num}: {accuracy:.1%} accuracy ({total} questions)\n"
        
        report += f"\n🎯 RECOMMENDATIONS\n"
        recommendations = analysis.get('recommendations', [])
        for i, rec in enumerate(recommendations):
            prefix = "├──" if i < len(recommendations) - 1 else "└──"
            report += f"{prefix} {rec}\n"
        
        return report
    
    def run_quick_validation(self, config_name: str = "qwen2.5_coder", 
                           num_questions: int = 10) -> Dict:
        """Run quick validation test for development"""
        
        print(f"🚀 Quick Validation Test")
        print(f"Config: {config_name} | Questions: {num_questions}")
        print("-" * 40)
        
        try:
            # Create agent
            agent = create_production_gaia_agent(config_name)
            
            # Get small test sample
            test_sample = self.get_balanced_test_sample(num_questions)
            
            results = []
            start_time = time.time()
            
            for i, question_data in enumerate(test_sample):
                print(f"Question {i+1}/{num_questions}: {question_data.get('Question', '')[:50]}...")
                
                try:
                    result = agent.run_single_question(
                        question=question_data.get("Question", ""),
                        ground_truth=question_data.get("Final answer", ""),
                        level=question_data.get("Level", 1)
                    )
                    
                    is_correct = self.evaluate_answer_correctness(
                        result.get("final_answer", ""),
                        question_data.get("Final answer", "")
                    )
                    
                    print(f"  ├── Answer: {result.get('final_answer', 'No answer')}")
                    print(f"  ├── Expected: {question_data.get('Final answer', 'N/A')}")
                    print(f"  ├── Correct: {'✅' if is_correct else '❌'}")
                    print(f"  └── Strategy: {result.get('selected_strategy', 'Unknown')}")
                    
                    results.append({
                        "question": question_data.get("Question", ""),
                        "predicted": result.get("final_answer", ""),
                        "expected": question_data.get("Final answer", ""),
                        "is_correct": is_correct,
                        "strategy": result.get("selected_strategy", ""),
                        "level": question_data.get("Level", 1)
                    })
                    
                except Exception as e:
                    print(f"  └── Error: {e}")
                    results.append({
                        "question": question_data.get("Question", ""),
                        "error": str(e),
                        "is_correct": False
                    })
            
            agent.close()
            
            # Calculate metrics
            total_time = time.time() - start_time
            correct_count = sum(1 for r in results if r.get("is_correct", False))
            accuracy = correct_count / len(results) if results else 0
            
            validation_result = {
                "config_tested": config_name,
                "total_questions": len(results),
                "correct_answers": correct_count,
                "accuracy": accuracy,
                "total_time": total_time,
                "avg_time_per_question": total_time / len(results) if results else 0,
                "meets_gaia_target": accuracy >= 0.45,
                "detailed_results": results
            }
            
            print(f"\n🎯 VALIDATION RESULTS")
            print(f"├── Accuracy: {accuracy:.1%}")
            print(f"├── Total Time: {total_time:.2f}s")
            print(f"├── Avg Time/Question: {total_time/len(results):.2f}s")
            print(f"└── GAIA Target: {'✅ Met' if accuracy >= 0.45 else '❌ Not Met'}")
            
            return validation_result
            
        except Exception as e:
            error_result = {
                "config_tested": config_name,
                "error": str(e),
                "accuracy": 0.0
            }
            print(f"❌ Validation failed: {e}")
            return error_result

# ============================================================================
# SUPPORTING CLASSES
# ============================================================================

class BudgetTracker:
    """Track budget usage during testing"""
    
    def __init__(self, max_budget: float):
        self.max_budget = max_budget
        self.spent = 0.0
        self.transactions = []
    
    @property
    def remaining(self) -> float:
        return max(0.0, self.max_budget - self.spent)
    
    def can_continue(self, estimated_cost: float = 0.05) -> bool:
        return (self.spent + estimated_cost) <= self.max_budget
    
    def spend(self, amount: float):
        self.spent += amount
        self.transactions.append({
            "amount": amount,
            "timestamp": datetime.now(),
            "remaining": self.remaining
        })
    
    def get_summary(self) -> Dict:
        return {
            "max_budget": self.max_budget,
            "spent": self.spent,
            "remaining": self.remaining,
            "utilization": self.spent / self.max_budget if self.max_budget > 0 else 0,
            "transactions": len(self.transactions)
        }

class ErrorTracker:
    """Track errors during testing"""
    
    def __init__(self, max_errors: int = 10):
        self.max_errors = max_errors
        self.errors = []
        self.error_count = 0
    
    def record_error(self, source: str, error_msg: str):
        self.error_count += 1
        self.errors.append({
            "source": source,
            "error": error_msg,
            "timestamp": datetime.now()
        })
    
    def should_abort(self) -> bool:
        return self.error_count >= self.max_errors
    
    def get_error_summary(self) -> Dict:
        error_sources = Counter([e["source"] for e in self.errors])
        return {
            "total_errors": self.error_count,
            "error_by_source": dict(error_sources),
            "recent_errors": self.errors[-5:] if self.errors else []
        }

class PerformanceMetrics:
    """Track performance metrics during testing"""
    
    def __init__(self):
        self.metrics = {
            "accuracy_history": [],
            "timing_history": [],
            "memory_usage": [],
            "api_calls": 0
        }
    
    def record_result(self, accuracy: float, timing: float):
        self.metrics["accuracy_history"].append(accuracy)
        self.metrics["timing_history"].append(timing)
        self.metrics["api_calls"] += 1
    
    def get_trends(self) -> Dict:
        if not self.metrics["accuracy_history"]:
            return {}
        
        return {
            "accuracy_trend": np.polyfit(range(len(self.metrics["accuracy_history"])), 
                                       self.metrics["accuracy_history"], 1)[0],
            "timing_trend": np.polyfit(range(len(self.metrics["timing_history"])), 
                                     self.metrics["timing_history"], 1)[0],
            "current_accuracy": self.metrics["accuracy_history"][-1],
            "average_timing": np.mean(self.metrics["timing_history"])
        }

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_production_test(configs: List[str] = None, sample_size: int = 30) -> Dict:
    """Run production test with default settings"""
    
    test_config = TestConfig(
        max_questions_per_config=sample_size,
        max_total_budget=8.0,  # Leave budget for final submission
        enable_model_comparison=True,
        enable_performance_tracking=True,
        generate_visualizations=True
    )
    
    framework = GAIAProductionTestFramework(test_config)
    
    if configs is None:
        configs = ["qwen2.5_coder", "qwen_qwq_groq", "gemini_flash_04"]
    
    return framework.run_comprehensive_evaluation(configs, sample_size)

def quick_agent_validation(config_name: str = "qwen2.5_coder") -> Dict:
    """Quick validation for development"""
    
    framework = GAIAProductionTestFramework()
    return framework.run_quick_validation(config_name, num_questions=5)

def compare_agent_configs(configs: List[str], sample_size: int = 15) -> pd.DataFrame:
    """Compare multiple agent configurations"""
    
    framework = GAIAProductionTestFramework()
    
    comparison_results = []
    
    for config in configs:
        print(f"\n🔄 Testing {config}...")
        result = framework.test_single_configuration(config, sample_size)
        
        comparison_results.append({
            "config": config,
            "accuracy": result.get("accuracy", 0),
            "avg_time": result.get("avg_execution_time", 0),
            "total_cost": result.get("total_cost", 0),
            "error_rate": result.get("error_count", 0) / result.get("total_questions", 1),
            "fallback_rate": result.get("fallback_usage", 0)
        })
    
    df = pd.DataFrame(comparison_results)
    
    print(f"\n📊 CONFIGURATION COMPARISON")
    print("=" * 50)
    print(df.to_string(index=False, float_format='%.3f'))
    
    return df

def benchmark_best_config() -> Dict:
    """Benchmark the best performing configuration"""
    
    # First, find the best config with small sample
    print("🔍 Finding best configuration...")
    quick_comparison = compare_agent_configs(
        ["qwen2.5_coder", "qwen_qwq_groq", "gemini_flash_04", "deepseek"],
        sample_size=10
    )
    
    best_config = quick_comparison.loc[quick_comparison['accuracy'].idxmax(), 'config']
    print(f"🏆 Best config identified: {best_config}")
    
    # Run comprehensive benchmark on best config
    print(f"\n🚀 Running comprehensive benchmark on {best_config}...")
    return run_production_test([best_config], sample_size=50)

# ============================================================================
# MAIN EXECUTION FOR TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 GAIA Production Testing Framework")
    print("=" * 50)
    
    # Configuration options
    test_options = {
        "quick_validation": "Quick 5-question validation test",
        "config_comparison": "Compare multiple configurations (15 questions each)",
        "production_test": "Full production test (30 questions per config)",
        "benchmark_best": "Find and benchmark best configuration",
        "custom_test": "Custom test with your parameters"
    }
    
    print("\n📋 Available Test Options:")
    for key, description in test_options.items():
        print(f"  ├── {key}: {description}")
    
    print(f"\n💡 Usage Examples:")
    print(f"  ├── quick_agent_validation('qwen2.5_coder')")
    print(f"  ├── compare_agent_configs(['qwen2.5_coder', 'qwen_qwq_groq'])")
    print(f"  ├── run_production_test(['qwen2.5_coder'], sample_size=50)")
    print(f"  └── benchmark_best_config()")
    
    # Run quick demonstration
    print(f"\n🚀 Running quick demonstration...")
    try:
        demo_result = quick_agent_validation("qwen2.5_coder")
        
        if "error" not in demo_result:
            print(f"✅ Demo completed successfully!")
            print(f"   └── Achieved {demo_result['accuracy']:.1%} accuracy in demo")
        else:
            print(f"❌ Demo failed: {demo_result['error']}")
            print(f"💡 Check your setup and API keys")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print(f"💡 Ensure gaia_agent_system.py and dependencies are available")
    
    print(f"\n🎓 Ready for comprehensive GAIA testing!")
    print(f"Use the functions above to run your tests.")