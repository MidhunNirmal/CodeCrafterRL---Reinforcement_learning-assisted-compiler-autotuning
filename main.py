# ==========================================
# File: main.py
# ==========================================

import logging
from trainer import CompilerOptimizationTrainer

# Optional: configure logging for debug
logging.basicConfig(level=logging.INFO)

def main():
    # Step 1: Initialize the trainer
    trainer = CompilerOptimizationTrainer(data_path="exec_times.csv")
    
    # Step 2: Setup everything (preprocessing, agents, envs)
    trainer.setup()
    
    # Step 3: Train all benchmarks (or pick one manually)
    
    ## Option A: Train all
    trainer.train_all_benchmarks(episodes=500)  # You can change to 1000+ for full runs
    
    ## Option B: Train a specific benchmark (e.g., 'gemm')
    # trainer.train_single_benchmark("gemm", episodes=1000)

if __name__ == "__main__":
    main()
