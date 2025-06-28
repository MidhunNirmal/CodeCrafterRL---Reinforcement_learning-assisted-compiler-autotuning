import logging
import os
from trainer import CompilerOptimizationTrainer
from model_validator import ModelValidator

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_validation():
    """Run comprehensive model validation"""
    
    # Check if data and models exist
    data_path = "data/exec_times.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    logger.info("Loading trained models for validation...")
    
    # Initialize trainer and load models
    trainer = CompilerOptimizationTrainer(data_path)
    trainer.setup()
    
    # Load any existing results
    if os.path.exists("results/training_results.json"):
        trainer.load_results("results/training_results.json")
    
    # Check if we have trained models
    models_dir = "models"
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        logger.warning("No trained models found. Training models first...")
        
        # Quick training for validation
        trainer.train_all_benchmarks(episodes=200)
    else:
        # Load existing models
        for benchmark in trainer.benchmark_data.keys():
            model_path = f"{models_dir}/{benchmark}_final.pth"
            if os.path.exists(model_path):
                try:
                    trainer.agents[benchmark].load(model_path)
                    logger.info(f"Loaded model for {benchmark}")
                except Exception as e:
                    logger.warning(f"Could not load model for {benchmark}: {e}")
    
    # Create validator
    validator = ModelValidator(trainer)
    
    # Run comprehensive validation
    logger.info("Starting comprehensive validation...")
    
    # Test first few benchmarks for demonstration
    benchmark_names = list(trainer.benchmark_data.keys())[:3]  
    
    for benchmark in benchmark_names:
        logger.info(f"\nValidating benchmark: {benchmark}")
        
        try:
            # Run all validation tests
            results = validator.run_comprehensive_validation(benchmark)
            
            # Print summary
            if benchmark in results and results[benchmark]['dataset_validation']:
                metrics = results[benchmark]['dataset_validation']['metrics']
                print(f"\nValidation Summary for {benchmark}:")
                print(f"  RMSE: {metrics['rmse']:.6f}")
                print(f"  R² Score: {metrics['r2_score']:.4f}")
                print(f"  Configuration Accuracy: {metrics['config_accuracy']:.4f}")
                
                if 'configuration_testing' in results[benchmark]:
                    config_summary = results[benchmark]['configuration_testing']['summary']
                    print(f"  Configs Improved: {config_summary['configs_improved']}/{config_summary['total_configs_tested']}")
                    print(f"  Improvement Rate: {config_summary['improvement_rate']:.4f}")
        
        except Exception as e:
            logger.error(f"Error validating {benchmark}: {e}")
    
    logger.info("Validation completed!")

if __name__ == "__main__":
    run_validation()
