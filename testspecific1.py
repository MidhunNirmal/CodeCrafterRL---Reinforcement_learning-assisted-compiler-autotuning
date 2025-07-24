# import numpy as np
# import logging
# import matplotlib.pyplot as plt
# import os
# import json
# from trainer import CompilerOptimizationTrainer
# from model_validator import ModelValidator

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def test_specific_configurations():
#     """Test model on predefined compiler configurations for all benchmarks, visualize results, and save to files"""
    
#     # Initialize trainer
#     trainer = CompilerOptimizationTrainer("data/exec_times.csv")
#     trainer.setup()
    
#     # Load trained models if available
#     models_dir = "models"
#     if os.path.exists(models_dir):
#         for benchmark in trainer.benchmark_data.keys():
#             model_path = f"{models_dir}/{benchmark}_final.pth"
#             if os.path.exists(model_path):
#                 trainer.agents[benchmark].load(model_path)
#                 logger.info(f"Loaded model for {benchmark}")
    
#     # Create validator
#     validator = ModelValidator(trainer)
    
#     # Define test configurations
#     test_configs = [
#         [0, 0, 0, 0, 0, 0, 0],  # No optimizations
#         [0, 0, 0, 0, 0, 0, 1],  # Only O2
#         [1, 0, 0, 0, 0, 1, 1],  # Aggressive opts
#         [0, 0, 0, 0, 1, 0, 1],  # Conservative opts
#         [1, 1, 1, 1, 1, 1, 1],  # All flags enabled
#         [1, 0, 1, 0, 1, 0, 1],  # Mixed config
#         [1, 0, 0, 0, 0, 0, 0],  # Only unsafe math
#         [0, 1, 1, 1, 1, 0, 0]   # Disable opts
#     ]
    
#     config_names = [
#         "No opts", "Only O2", "Aggressive", "Conservative", 
#         "All flags", "Mixed", "Unsafe math", "Disable opts"
#     ]
    
#     # Initialize dictionaries and lists for saving results
#     all_results_json = {}
#     text_output = []
    
#     # Test on all available benchmarks
#     benchmark_names = list(trainer.benchmark_data.keys())
    
#     for benchmark in benchmark_names:
#         print(f"\n{'='*60}")
#         print(f"Testing configurations on benchmark: {benchmark}")
#         print(f"{'='*60}")
#         text_output.append(f"\n{'='*60}")
#         text_output.append(f"Testing configurations on benchmark: {benchmark}")
#         text_output.append(f"{'='*60}")
        
#         # Test specific configurations
#         results = validator.test_specific_configurations(benchmark, test_configs)
        
#         if results:
#             print(f"\nResults for {benchmark}:")
#             print("-" * 50)
#             text_output.append(f"\nResults for {benchmark}:")
#             text_output.append("-" * 50)
            
#             # Collect data for visualization and JSON
#             input_improvements = []
#             suggested_improvements = []
#             benchmark_results_json = {
#                 'benchmark': benchmark,
#                 'results': [],
#                 'summary': results['summary']
#             }
            
#             for i, (config_name, result) in enumerate(zip(config_names, results['results'])):
#                 input_config = result['input_config']
#                 suggested_config = result['suggested_config']
#                 actual_improvement = result['actual_improvement'] * 100
#                 suggested_improvement = result['suggested_improvement'] * 100
#                 model_improvement = result['model_improvement'] * 100
                
#                 print(f"\n{config_name}:")
#                 print(f"  Input config:     {input_config}")
#                 print(f"  Suggested config: {suggested_config}")
#                 print(f"  Input improvement:     {actual_improvement:6.2f}%")
#                 print(f"  Suggested improvement: {suggested_improvement:6.2f}%")
#                 print(f"  Model improvement:     {model_improvement:+6.2f}%")
#                 print(f"  {'✓ Model suggested different configuration' if result['config_changed'] else '- Model kept same configuration'}")
                
#                 text_output.append(f"\n{config_name}:")
#                 text_output.append(f"  Input config:     {input_config}")
#                 text_output.append(f"  Suggested config: {suggested_config}")
#                 text_output.append(f"  Input improvement:     {actual_improvement:6.2f}%")
#                 text_output.append(f"  Suggested improvement: {suggested_improvement:6.2f}%")
#                 text_output.append(f"  Model improvement:     {model_improvement:+6.2f}%")
#                 text_output.append(f"  {'✓ Model suggested different configuration' if result['config_changed'] else '- Model kept same configuration'}")
                
#                 input_improvements.append(actual_improvement)
#                 suggested_improvements.append(suggested_improvement)
                
#                 # Add to JSON results
#                 benchmark_results_json['results'].append({
#                     'config_name': config_name,
#                     'input_config': input_config,
#                     'suggested_config': suggested_config,
#                     'input_improvement': actual_improvement,
#                     'suggested_improvement': suggested_improvement,
#                     'model_improvement': model_improvement,
#                     'config_changed': result['config_changed']
#                 })
            
#             # Print summary
#             summary = results['summary']
#             print(f"\nSummary for {benchmark}:")
#             print(f"  Total configurations tested: {summary['total_configs_tested']}")
#             print(f"  Configurations improved: {summary['configs_improved']}")
#             print(f"  Improvement rate: {summary['improvement_rate']:.4f}")
#             print(f"  Mean improvement: {summary['mean_improvement']*100:.2f}%")
#             print(f"  Mean positive improvement: {summary['mean_positive_improvement']*100:.2f}%")
#             print(f"  Maximum improvement: {summary['max_improvement']*100:.2f}%")
#             print(f"  Minimum improvement: {summary['min_improvement']*100:.2f}%")
            
#             text_output.append(f"\nSummary for {benchmark}:")
#             text_output.append(f"  Total configurations tested: {summary['total_configs_tested']}")
#             text_output.append(f"  Configurations improved: {summary['configs_improved']}")
#             text_output.append(f"  Improvement rate: {summary['improvement_rate']:.4f}")
#             text_output.append(f"  Mean improvement: {summary['mean_improvement']*100:.2f}%")
#             text_output.append(f"  Mean positive improvement: {summary['mean_positive_improvement']*100:.2f}%")
#             text_output.append(f"  Maximum improvement: {summary['max_improvement']*100:.2f}%")
#             text_output.append(f"  Minimum improvement: {summary['min_improvement']*100:.2f}%")
            
#             # Store in JSON dictionary
#             all_results_json[benchmark] = benchmark_results_json
            
#             # Visualize improvements
#             fig, ax = plt.subplots(figsize=(10, 6))
#             x = np.arange(len(config_names))
#             width = 0.35
            
#             ax.bar(x - width/2, input_improvements, width, label='Input Improvement', alpha=0.6)
#             ax.bar(x + width/2, suggested_improvements, width, label='Suggested Improvement', alpha=0.6)
            
#             ax.set_xlabel('Configuration')
#             ax.set_ylabel('Improvement (%)')
#             ax.set_title(f'Improvement Comparison for {benchmark}')
#             ax.set_xticks(x)
#             ax.set_xticklabels(config_names, rotation=45, ha='right')
#             ax.legend()
#             ax.grid(True, alpha=0.3)
            
#             # Add value labels on top of bars
#             for i in range(len(config_names)):
#                 ax.text(i - width/2, input_improvements[i] + 0.1, f'{input_improvements[i]:.2f}%', 
#                         ha='center', va='bottom' if input_improvements[i] >= 0 else 'top')
#                 ax.text(i + width/2, suggested_improvements[i] + 0.1, f'{suggested_improvements[i]:.2f}%', 
#                         ha='center', va='bottom' if suggested_improvements[i] >= 0 else 'top')
            
#             plt.tight_layout()
#             os.makedirs("plots", exist_ok=True)
#             safe_benchmark_name = benchmark.replace("/", "_")
#             plt.savefig(f"plots/{safe_benchmark_name}_config_improvements.png", dpi=300, bbox_inches='tight')
#             plt.show()
    
#     # Save results to JSON file
#     os.makedirs("results", exist_ok=True)
#     json_output_path = "results/all_benchmarks_test_results.json"
#     with open(json_output_path, 'w') as f:
#         json.dump(all_results_json, f, indent=2)
#     logger.info(f"Results saved to {json_output_path}")
    
#     # Save results to text file
#     text_output_path = "results/all_benchmarks_test_results.txt"
#     with open(text_output_path, 'w') as f:
#         f.write("\n".join(text_output))
#     logger.info(f"Results saved to {text_output_path}")

# if __name__ == "__main__":
#     test_specific_configurations()


import numpy as np
import logging
import matplotlib.pyplot as plt
import os
import json
from trainer import CompilerOptimizationTrainer
from model_validator import ModelValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_configurations():
    """Test model on predefined compiler configurations for all benchmarks, visualize results, and save only summary to files"""
    
    # Initialize trainer
    trainer = CompilerOptimizationTrainer("data/exec_times.csv")
    trainer.setup()
    
    # Load trained models if available
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
        [0, 0, 0, 0, 0, 0, 0],  # No optimizations
        [0, 0, 0, 0, 0, 0, 1],  # Only O2
        [1, 0, 0, 0, 0, 1, 1],  # Aggressive opts
        [0, 0, 0, 0, 1, 0, 1],  # Conservative opts
        [1, 1, 1, 1, 1, 1, 1],  # All flags enabled
        [1, 0, 1, 0, 1, 0, 1],  # Mixed config
        [1, 0, 0, 0, 0, 0, 0],  # Only unsafe math
        [0, 1, 1, 1, 1, 0, 0]   # Disable opts
    ]
    
    config_names = [
        "No opts", "Only O2", "Aggressive", "Conservative", 
        "All flags", "Mixed", "Unsafe math", "Disable opts"
    ]
    
    # Initialize dictionaries and lists for saving summary results
    all_results_json = {}
    text_output = []
    
    # Test on all available benchmarks
    benchmark_names = list(trainer.benchmark_data.keys())
    
    for benchmark in benchmark_names:
        print(f"\n{'='*60}")
        print(f"Testing configurations on benchmark: {benchmark}")
        print(f"{'='*60}")
        text_output.append(f"\n{'='*60}")
        text_output.append(f"Testing configurations on benchmark: {benchmark}")
        text_output.append(f"{'='*60}")
        
        # Test specific configurations
        results = validator.test_specific_configurations(benchmark, test_configs)
        
        if results:
            # Collect data for visualization
            input_improvements = []
            suggested_improvements = []
            benchmark_results_json = {
                'benchmark': benchmark,
                'summary': results['summary']
            }
            
            for i, result in enumerate(results['results']):
                input_improvements.append(result['actual_improvement'] * 100)
                suggested_improvements.append(result['suggested_improvement'] * 100)
            
            # Print and save only summary
            summary = results['summary']
            print(f"\nSummary for {benchmark}:")
            print(f"  Total configurations tested: {summary['total_configs_tested']}")
            print(f"  Configurations improved: {summary['configs_improved']}")
            print(f"  Improvement rate: {summary['improvement_rate']:.4f}")
            print(f"  Mean improvement: {summary['mean_improvement']*100:.2f}%")
            print(f"  Mean positive improvement: {summary['mean_positive_improvement']*100:.2f}%")
            print(f"  Maximum improvement: {summary['max_improvement']*100:.2f}%")
            print(f"  Minimum improvement: {summary['min_improvement']*100:.2f}%")
            
            text_output.append(f"\nSummary for {benchmark}:")
            text_output.append(f"  Total configurations tested: {summary['total_configs_tested']}")
            text_output.append(f"  Configurations improved: {summary['configs_improved']}")
            text_output.append(f"  Improvement rate: {summary['improvement_rate']:.4f}")
            text_output.append(f"  Mean improvement: {summary['mean_improvement']*100:.2f}%")
            text_output.append(f"  Mean positive improvement: {summary['mean_positive_improvement']*100:.2f}%")
            text_output.append(f"  Maximum improvement: {summary['max_improvement']*100:.2f}%")
            text_output.append(f"  Minimum improvement: {summary['min_improvement']*100:.2f}%")
            
            # Store in JSON dictionary
            all_results_json[benchmark] = benchmark_results_json
            
            # Visualize improvements
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(config_names))
            width = 0.35
            
            ax.bar(x - width/2, input_improvements, width, label='Input Improvement', alpha=0.6)
            ax.bar(x + width/2, suggested_improvements, width, label='Suggested Improvement', alpha=0.6)
            
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Improvement (%)')
            ax.set_title(f'Improvement Comparison for {benchmark}')
            ax.set_xticks(x)
            ax.set_xticklabels(config_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on top of bars
            for i in range(len(config_names)):
                ax.text(i - width/2, input_improvements[i] + 0.1, f'{input_improvements[i]:.2f}%', 
                        ha='center', va='bottom' if input_improvements[i] >= 0 else 'top')
                ax.text(i + width/2, suggested_improvements[i] + 0.1, f'{suggested_improvements[i]:.2f}%', 
                        ha='center', va='bottom' if suggested_improvements[i] >= 0 else 'top')
            
            plt.tight_layout()
            os.makedirs("plots", exist_ok=True)
            safe_benchmark_name = benchmark.replace("/", "_")
            plt.savefig(f"plots/{safe_benchmark_name}_config_improvements.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    # Save summary results to JSON file
    os.makedirs("results", exist_ok=True)
    json_output_path = "results/all_benchmarks_summary.json"
    with open(json_output_path, 'w') as f:
        json.dump(all_results_json, f, indent=2)
    logger.info(f"Summary results saved to {json_output_path}")
    
    # Save summary results to text file
    text_output_path = "results/all_benchmarks_summary.txt"
    with open(text_output_path, 'w') as f:
        f.write("\n".join(text_output))
    logger.info(f"Summary results saved to {text_output_path}")

if __name__ == "__main__":
    test_specific_configurations()