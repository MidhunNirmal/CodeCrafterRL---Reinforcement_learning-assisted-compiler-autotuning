# ==========================================
# File: test_specific_inputs.py
# ==========================================

"""
Test the model on specific compiler flag configurations
"""

import numpy as np
import logging
from trainer import CompilerOptimizationTrainer
from model_validator import ModelValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_configurations():
    """Test model on predefined compiler configurations"""
    
    # Initialize trainer
    trainer = CompilerOptimizationTrainer("data/exec_times.csv")
    trainer.setup()
    
    # Load trained models if available
    import os
    models_dir = "models"
    if os.path.exists(models_dir):
        for benchmark in trainer.benchmark_data.keys():
            model_path = f"{models_dir}/{benchmark}_final.pth"
            if os.path.exists(model_path):
                trainer.agents[benchmark].load(model_path)
                logger.info(f"Loaded model for {benchmark}")
    
    # Create validator
    validator = ModelValidator(trainer)
    
    # Define test configurations
    test_configs = [
        # Config 1: No optimizations
        [0, 0, 0, 0, 0, 0, 0],
        
        # Config 2: Only O2
        [0, 0, 0, 0, 0, 0, 1],
        
        # Config 3: Aggressive optimizations
        [1, 0, 0, 0, 0, 1, 1],
        
        # Config 4: Conservative optimizations
        [0, 0, 0, 0, 1, 0, 1],
        
        # Config 5: All flags enabled
        [1, 1, 1, 1, 1, 1, 1],
        
        # Config 6: Mixed configuration
        [1, 0, 1, 0, 1, 0, 1],
        
        # Config 7: Only unsafe math
        [1, 0, 0, 0, 0, 0, 0],
        
        # Config 8: Disable optimizations
        [0, 1, 1, 1, 1, 0, 0]
    ]
    
    config_names = [
        "No optimizations",
        "Only O2",
        "Aggressive opts",
        "Conservative opts", 
        "All flags enabled",
        "Mixed config",
        "Only unsafe math",
        "Disable opts"
    ]
    
    # Test on first available benchmark
    benchmark_names = list(trainer.benchmark_data.keys())
    
    for benchmark in benchmark_names[:2]:  # Test first 2 benchmarks
        print(f"\n{'='*60}")
        print(f"Testing configurations on benchmark: {benchmark}")
        print(f"{'='*60}")
        
        # Test specific configurations
        results = validator.test_specific_configurations(benchmark, test_configs)
        
        if results:
            print(f"\nResults for {benchmark}:")
            print("-" * 50)
            
            for i, (config_name, result) in enumerate(zip(config_names, results['results'])):
                input_config = result['input_config']
                suggested_config = result['suggested_config']
                actual_improvement = result['actual_improvement'] * 100
                suggested_improvement = result['suggested_improvement'] * 100
                model_improvement = result['model_improvement'] * 100
                
                print(f"\n{config_name}:")
                print(f"  Input config:     {input_config}")
                print(f"  Suggested config: {suggested_config}")
                print(f"  Input improvement:     {actual_improvement:6.2f}%")
                print(f"  Suggested improvement: {suggested_improvement:6.2f}%")
                print(f"  Model improvement:     {model_improvement:+6.2f}%")
                
                if result['config_changed']:
                    print(f"  ✓ Model suggested different configuration")
                else:
                    print(f"  - Model kept same configuration")
            
            # Print summary
            summary = results['summary']
            print(f"\nSummary for {benchmark}:")
            print(f"  Total configurations tested: {summary['total_configs_tested']}")
            print(f"  Configurations improved: {summary['configs_improved']}")
            print(f"  Improvement rate: {summary['improvement_rate']:.4f}")
            print(f"  Mean improvement: {summary['mean_improvement']*100:.2f}%")
            print(f"  Mean positive improvement: {summary['mean_positive_improvement']*100:.2f}%")
            print(f"  Maximum improvement: {summary['max_improvement']*100:.2f}%")
            print(f"  Minimum improvement: {summary['min_improvement']*100:.2f}%")

if __name__ == "__main__":
    
    test_specific_configurations()