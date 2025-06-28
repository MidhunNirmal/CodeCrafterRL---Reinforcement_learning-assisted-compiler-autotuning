import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import logging
import json
import os
from typing import Dict, List, Tuple, Optional
import torch
from collections import defaultdict
import time

from data_preprocess import DataPreprocessor
from comp_env import CompilerEnvironment
from dqn_agent import DQNAgent
from trainer import CompilerOptimizationTrainer

logger = logging.getLogger(__name__)

class ModelValidator:
    """Comprehensive validation system for compiler optimization models"""
    
    def __init__(self, trainer: CompilerOptimizationTrainer):
        self.trainer = trainer
        self.benchmark_data = trainer.benchmark_data
        self.agents = trainer.agents
        self.environments = trainer.environments
        self.validation_results = {}
        
    def validate_against_dataset(self, benchmark_name: str, sample_size: Optional[int] = None):
        """Validate model predictions against actual dataset values"""
        
        if benchmark_name not in self.benchmark_data:
            logger.error(f"Benchmark {benchmark_name} not found")
            return None
            
        benchmark_data = self.benchmark_data[benchmark_name]
        agent = self.agents[benchmark_name]
        env = self.environments[benchmark_name]
        
        # Get test data
        X_test = benchmark_data['X_test']
        y_test = benchmark_data['y_test']
        X_raw = benchmark_data['X_raw']
        y_raw = benchmark_data['y_raw']
        feature_names = benchmark_data['feature_names']
        scaler = benchmark_data['scaler']
        
        # Sample data if specified
        if sample_size and sample_size < len(X_raw):
            indices = np.random.choice(len(X_raw), sample_size, replace=False)
            X_sample = X_raw[indices]
            y_actual = y_raw[indices]
        else:
            X_sample = X_raw
            y_actual = y_raw
        
        logger.info(f"Validating {benchmark_name} on {len(X_sample)} samples...")
        
        # Get model predictions
        predictions = []
        model_configs = []
        
        for i, config_raw in enumerate(X_sample):
            # Extract compiler flags (first 7 features)
            compiler_flags = config_raw[:7].astype(int)
            
            # Create state for the model (similar to environment observation)
            # We need to simulate the environment state
            current_time = y_actual[i]  # Use actual time as current performance
            
            # Normalize performance metrics (similar to environment)
            normalized_current = (current_time - env.best_time) / (env.worst_time - env.best_time)
            normalized_current = np.clip(normalized_current, 0, 1)
            
            # Create observation
            improvement_from_baseline = (env.baseline_time - current_time) / env.baseline_time
            steps_remaining = 1.0  # Assume full episode remaining
            
            state = np.concatenate([
                compiler_flags.astype(np.float32),
                [normalized_current],
                [steps_remaining], 
                [improvement_from_baseline]
            ])
            
            # Get model's predicted action
            predicted_action = agent.act(state, training=False)
            model_configs.append(predicted_action)
            
            # Get predicted execution time for this configuration
            predicted_time = env._get_execution_time(predicted_action)
            predictions.append(predicted_time)
        
        predictions = np.array(predictions)
        actual_configs = X_sample[:, :7].astype(int)  # Actual compiler flags
        
        # Calculate metrics
        mse = mean_squared_error(y_actual, predictions)
        mae = mean_absolute_error(y_actual, predictions)
        rmse = np.sqrt(mse)
        
        # R² score (coefficient of determination)
        r2 = r2_score(y_actual, predictions)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_actual - predictions) / y_actual)) * 100
        
        # Configuration accuracy (how often model suggests better configs)
        config_improvements = 0
        for i, (actual_config, model_config) in enumerate(zip(actual_configs, model_configs)):
            actual_time = y_actual[i]
            model_time = env._get_execution_time(model_config)
            if model_time < actual_time:
                config_improvements += 1
        
        config_accuracy = config_improvements / len(actual_configs)
        
        # Performance improvement analysis
        baseline_time = env.baseline_time
        actual_improvements = (baseline_time - y_actual) / baseline_time
        predicted_improvements = (baseline_time - predictions) / baseline_time
        
        results = {
            'benchmark': benchmark_name,
            'sample_size': len(X_sample),
            'metrics': {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'mape': mape,
                'config_accuracy': config_accuracy
            },
            'performance_analysis': {
                'baseline_time': baseline_time,
                'mean_actual_improvement': np.mean(actual_improvements),
                'mean_predicted_improvement': np.mean(predicted_improvements),
                'actual_best_time': np.min(y_actual),
                'predicted_best_time': np.min(predictions),
                'correlation_coefficient': np.corrcoef(y_actual, predictions)[0, 1]
            },
            'predictions': predictions.tolist(),
            'actual_values': y_actual.tolist(),
            'actual_configs': actual_configs.tolist(),
            'model_configs': [config.tolist() for config in model_configs]
        }
        
        logger.info(f"Validation results for {benchmark_name}:")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  Config Accuracy: {config_accuracy:.4f}")
        
        return results
    
    def cross_validate_benchmark(self, benchmark_name: str, cv_folds: int = 5):
        """Perform cross-validation on benchmark data"""
        
        if benchmark_name not in self.benchmark_data:
            logger.error(f"Benchmark {benchmark_name} not found")
            return None
        
        benchmark_data = self.benchmark_data[benchmark_name]
        X_raw = benchmark_data['X_raw']
        y_raw = benchmark_data['y_raw']
        
        logger.info(f"Cross-validating {benchmark_name} with {cv_folds} folds...")
        
        # Use only compiler flags for baseline comparison
        X_flags = X_raw[:, :7]  # First 7 features are compiler flags
        
        # Compare against baseline models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        cv_results = {}
        
        for model_name, model in models.items():
            # Perform cross-validation
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Calculate cross-validation scores
            mse_scores = -cross_val_score(model, X_flags, y_raw, cv=kfold, scoring='neg_mean_squared_error')
            mae_scores = -cross_val_score(model, X_flags, y_raw, cv=kfold, scoring='neg_mean_absolute_error')
            r2_scores = cross_val_score(model, X_flags, y_raw, cv=kfold, scoring='r2')
            
            cv_results[model_name] = {
                'mse_mean': np.mean(mse_scores),
                'mse_std': np.std(mse_scores),
                'mae_mean': np.mean(mae_scores),
                'mae_std': np.std(mae_scores),
                'r2_mean': np.mean(r2_scores),
                'r2_std': np.std(r2_scores),
                'rmse_mean': np.sqrt(np.mean(mse_scores))
            }
            
            logger.info(f"{model_name} CV Results:")
            logger.info(f"  RMSE: {cv_results[model_name]['rmse_mean']:.6f} ± {np.sqrt(np.std(mse_scores)):.6f}")
            logger.info(f"  R² Score: {cv_results[model_name]['r2_mean']:.4f} ± {cv_results[model_name]['r2_std']:.4f}")
        
        return cv_results
    
    def test_specific_configurations(self, benchmark_name: str, test_configs: List[List[int]]):
        """Test model on specific compiler flag configurations"""
        
        if benchmark_name not in self.agents:
            logger.error(f"No trained agent found for benchmark: {benchmark_name}")
            return None
        
        agent = self.agents[benchmark_name]
        env = self.environments[benchmark_name]
        
        logger.info(f"Testing {len(test_configs)} specific configurations on {benchmark_name}")
        
        results = []
        
        for i, config in enumerate(test_configs):
            config = np.array(config[:7])  # Ensure 7 flags
            
            # Get actual execution time from environment
            actual_time = env._get_execution_time(config)
            
            # Create state observation
            normalized_current = (actual_time - env.best_time) / (env.worst_time - env.best_time)
            normalized_current = np.clip(normalized_current, 0, 1)
            improvement_from_baseline = (env.baseline_time - actual_time) / env.baseline_time
            
            state = np.concatenate([
                config.astype(np.float32),
                [normalized_current],
                [1.0],  # steps_remaining
                [improvement_from_baseline]
            ])
            
            # Get model's suggested action
            suggested_config = agent.act(state, training=False)
            suggested_time = env._get_execution_time(suggested_config)
            
            # Calculate improvements
            actual_improvement = (env.baseline_time - actual_time) / env.baseline_time
            suggested_improvement = (env.baseline_time - suggested_time) / env.baseline_time
            
            result = {
                'config_id': i,
                'input_config': config.tolist(),
                'suggested_config': suggested_config.tolist(),
                'actual_time': actual_time,
                'suggested_time': suggested_time,
                'actual_improvement': actual_improvement,
                'suggested_improvement': suggested_improvement,
                'model_improvement': suggested_improvement - actual_improvement,
                'config_changed': not np.array_equal(config, suggested_config)
            }
            
            results.append(result)
            
            logger.info(f"Config {i}: Input improvement: {actual_improvement:.4f}, "
                       f"Suggested improvement: {suggested_improvement:.4f}")
        
        # Summary statistics
        improvements = [r['model_improvement'] for r in results]
        positive_improvements = [imp for imp in improvements if imp > 0]
        
        summary = {
            'total_configs_tested': len(test_configs),
            'configs_improved': len(positive_improvements),
            'improvement_rate': len(positive_improvements) / len(test_configs),
            'mean_improvement': np.mean(improvements),
            'mean_positive_improvement': np.mean(positive_improvements) if positive_improvements else 0,
            'max_improvement': max(improvements),
            'min_improvement': min(improvements)
        }
        
        return {
            'benchmark': benchmark_name,
            'results': results,
            'summary': summary
        }
    
    def generate_test_configurations(self, num_configs: int = 50):
        """Generate test configurations for validation"""
        
        # Generate diverse test configurations
        test_configs = []
        
        # 1. Random configurations
        for _ in range(num_configs // 5):
            config = np.random.randint(0, 2, 7).tolist()
            test_configs.append(config)
        
        # 2. All flags off
        test_configs.append([0, 0, 0, 0, 0, 0, 0])
        
        # 3. All flags on
        test_configs.append([1, 1, 1, 1, 1, 1, 1])
        
        # 4. Only O2 optimization
        test_configs.append([0, 0, 0, 0, 0, 0, 1])
        
        # 5. Common optimization patterns
        test_configs.extend([
            [1, 0, 0, 0, 0, 1, 1],  # Math optimizations + unroll + O2
            [0, 0, 0, 0, 1, 0, 1],  # No inline + O2
            [1, 0, 0, 0, 0, 0, 0],  # Only unsafe math
            [0, 1, 1, 1, 1, 0, 0],  # All disable flags
        ])
        
        # 6. Fill remaining with systematic combinations
        remaining = num_configs - len(test_configs)
        for _ in range(remaining):
            # Generate configurations with 2-4 flags enabled
            num_flags = np.random.randint(2, 5)
            config = [0] * 7
            positions = np.random.choice(7, num_flags, replace=False)
            for pos in positions:
                config[pos] = 1
            test_configs.append(config)
        
        return test_configs[:num_configs]
    
    def benchmark_comparison(self, benchmark_name: str):
        """Compare model performance against baseline algorithms"""
        
        if benchmark_name not in self.benchmark_data:
            logger.error(f"Benchmark {benchmark_name} not found")
            return None
        
        benchmark_data = self.benchmark_data[benchmark_name]
        X_test = benchmark_data['X_test']
        y_test = benchmark_data['y_test']
        X_train = benchmark_data['X_train']
        y_train = benchmark_data['y_train']
        X_raw = benchmark_data['X_raw']
        y_raw = benchmark_data['y_raw']
        
        logger.info(f"Comparing models for {benchmark_name}...")
        
        # Train baseline models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        comparison_results = {}
        
        # Test baseline models
        for model_name, model in models.items():
            # Train on compiler flags only
            X_flags_train = X_train[:, :7] if X_train.shape[1] > 7 else X_train
            X_flags_test = X_test[:, :7] if X_test.shape[1] > 7 else X_test
            
            model.fit(X_flags_train, y_train)
            predictions = model.predict(X_flags_test)
            
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            comparison_results[model_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'r2_score': r2
            }
        
        # Test our RL model
        rl_results = self.validate_against_dataset(benchmark_name, sample_size=len(X_test))
        if rl_results:
            comparison_results['RL Model'] = rl_results['metrics']
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results).T
        
        logger.info("Model Comparison Results:")
        logger.info(comparison_df.round(6))
        
        return comparison_results
    
    def visualize_validation_results(self, benchmark_name: str, validation_results: Dict):
        """Create visualizations for validation results"""
        
        if not validation_results:
            logger.warning("No validation results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Validation Results: {benchmark_name}', fontsize=16)
        
        predictions = np.array(validation_results['predictions'])
        actual_values = np.array(validation_results['actual_values'])
        
        # Plot 1: Predicted vs Actual
        axes[0, 0].scatter(actual_values, predictions, alpha=0.6)
        min_val = min(actual_values.min(), predictions.min())
        max_val = max(actual_values.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Actual Execution Time')
        axes[0, 0].set_ylabel('Predicted Execution Time')
        axes[0, 0].set_title('Predicted vs Actual')
        axes[0, 0].grid(True)
        
        # Add R² score to plot
        r2 = validation_results['metrics']['r2_score']
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Residuals
        residuals = predictions - actual_values
        axes[0, 1].scatter(predictions, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('Predicted Execution Time')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True)
        
        # Plot 3: Error distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].grid(True)
        
        # Plot 4: Performance improvements
        baseline_time = validation_results['performance_analysis']['baseline_time']
        actual_improvements = (baseline_time - actual_values) / baseline_time * 100
        predicted_improvements = (baseline_time - predictions) / baseline_time * 100
        
        axes[1, 1].scatter(actual_improvements, predicted_improvements, alpha=0.6)
        min_imp = min(actual_improvements.min(), predicted_improvements.min())
        max_imp = max(actual_improvements.max(), predicted_improvements.max())
        axes[1, 1].plot([min_imp, max_imp], [min_imp, max_imp], 'r--', alpha=0.8)
        axes[1, 1].set_xlabel('Actual Improvement (%)')
        axes[1, 1].set_ylabel('Predicted Improvement (%)')
        axes[1, 1].set_title('Performance Improvement Prediction')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs("validation_plots", exist_ok=True)
        plt.savefig(f"validation_plots/{benchmark_name}_validation.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_validation(self, benchmark_name: str = None, save_results: bool = True):
        """Run comprehensive validation on benchmark(s)"""
        
        benchmarks_to_test = [benchmark_name] if benchmark_name else list(self.benchmark_data.keys())
        
        all_results = {}
        
        for benchmark in benchmarks_to_test:
            logger.info(f"\n{'='*60}")
            logger.info(f"Comprehensive Validation: {benchmark}")
            logger.info(f"{'='*60}")
            
            try:
                # 1. Dataset validation
                logger.info("1. Running dataset validation...")
                dataset_results = self.validate_against_dataset(benchmark)
                
                # 2. Cross-validation
                logger.info("2. Running cross-validation...")
                cv_results = self.cross_validate_benchmark(benchmark)
                
                # 3. Model comparison
                logger.info("3. Running model comparison...")
                comparison_results = self.benchmark_comparison(benchmark)
                
                # 4. Specific configuration testing
                logger.info("4. Testing specific configurations...")
                test_configs = self.generate_test_configurations(50)
                config_results = self.test_specific_configurations(benchmark, test_configs)
                
                # 5. Generate visualizations
                if dataset_results:
                    logger.info("5. Generating visualizations...")
                    self.visualize_validation_results(benchmark, dataset_results)
                
                # Compile results
                all_results[benchmark] = {
                    'dataset_validation': dataset_results,
                    'cross_validation': cv_results,
                    'model_comparison': comparison_results,
                    'configuration_testing': config_results,
                    'validation_timestamp': time.time()
                }
                
            except Exception as e:
                logger.error(f"Error validating {benchmark}: {e}")
                continue
        
        # Save results
        if save_results:
            os.makedirs("validation_results", exist_ok=True)
            with open("validation_results/comprehensive_validation.json", 'w') as f:
                # Convert numpy types for JSON serialization
                json_results = self._convert_numpy_types(all_results)
                json.dump(json_results, f, indent=2)
            
            logger.info("Validation results saved to validation_results/comprehensive_validation.json")
        
        return all_results
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
