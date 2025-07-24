import pandas as pd
import numpy as np
import os
from typing import Dict
from sklearn.preprocessing import StandardScaler

class LOOCVDataPreprocessor:
    """LOOCV preprocessor that excludes 'security_sha' from training data"""
    def __init__(self, data_path: str = "../alternatedata/raw_data/data.csv", exclude_benchmark: str = "security_sha"):
        self.data_path = data_path
        self.exclude_benchmark = exclude_benchmark
        self.raw_data = None
        self.processed_data = None
        self.flag_columns = None
        self.static_feature_columns = None
        self.exec_time_columns = None
        self.benchmark_column = "APP_NAME"

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")
        self.raw_data = pd.read_csv(self.data_path)
        return self.raw_data

    def preprocess(self) -> Dict[str, pd.DataFrame]:
        if self.raw_data is None:
            self.load_data()
        data = self.raw_data.copy()
        
        # Define the 7 compiler flags (columns 2-8)
        self.flag_columns = [
            'funsafe_math_optimizations',
            'fno_guess_branch_probability', 
            'fno_ivopts',
            'fno_tree_loop_optimize',
            'fno_inline_functions',
            'funroll_all_loops',
            'o2'
        ]
        # Last 5 columns are execution times
        self.exec_time_columns = [
            'execution_time_1',
            'execution_time_2',
            'execution_time_3', 
            'execution_time_4',
            'execution_time_5'
        ]
        # All static features: columns 9–287 (code_size, noBasicBlock, ..., mem_write_global_stride_4096_5)
        all_columns = list(data.columns)
        static_start = all_columns.index('code_size')
        static_end = all_columns.index('mem_write_global_stride_4096_5') + 1
        self.static_feature_columns = all_columns[static_start:static_end]
        
        print(f"Flag columns (7): {self.flag_columns}")
        print(f"Static feature columns ({len(self.static_feature_columns)}): {self.static_feature_columns[:5]} ... {self.static_feature_columns[-5:]}")
        print(f"Execution time columns: {self.exec_time_columns}")
        
        # Convert the 7 flag columns to binary (1 for X, 0 otherwise)
        for flag in self.flag_columns:
            data[flag] = (data[flag] == 'X').astype(int) if data[flag].dtype == object else data[flag]
        
        # Normalize static features
        scaler = StandardScaler()
        data[self.static_feature_columns] = scaler.fit_transform(data[self.static_feature_columns])
        
        # Compute mean execution time
        data['mean_exec_time'] = data[self.exec_time_columns].mean(axis=1)
        
        # Split by benchmark
        benchmarks = data[self.benchmark_column].unique()
        benchmark_data = {}
        
        for bench in benchmarks:
            bench_df = data[data[self.benchmark_column] == bench].copy()
            # Only keep the 7 flag columns, all static features, and mean_exec_time
            output_columns = self.flag_columns + self.static_feature_columns + ['mean_exec_time']
            bench_df_out = bench_df[output_columns].copy()
            benchmark_data[bench] = bench_df_out.reset_index(drop=True)
        
        self.processed_data = benchmark_data
        return benchmark_data

    def get_training_data(self) -> Dict[str, pd.DataFrame]:
        """Get training data excluding the specified benchmark"""
        if self.processed_data is None:
            self.preprocess()
        
        training_data = {}
        for benchmark_name, data in self.processed_data.items():
            if benchmark_name != self.exclude_benchmark:
                training_data[benchmark_name] = data
        
        print(f"LOOCV Training: Excluded '{self.exclude_benchmark}' from training data")
        print(f"Training benchmarks: {len(training_data)}")
        print(f"Available training benchmarks: {list(training_data.keys())}")
        
        return training_data

    def get_test_data(self) -> Dict[str, pd.DataFrame]:
        """Get test data for the excluded benchmark"""
        if self.processed_data is None:
            self.preprocess()
        
        if self.exclude_benchmark not in self.processed_data:
            raise ValueError(f"Benchmark '{self.exclude_benchmark}' not found in data")
        
        test_data = {self.exclude_benchmark: self.processed_data[self.exclude_benchmark]}
        print(f"LOOCV Testing: Using '{self.exclude_benchmark}' as test data")
        print(f"Test samples: {len(test_data[self.exclude_benchmark])}")
        
        return test_data

if __name__ == "__main__":
    pre = LOOCVDataPreprocessor()
    pre.load_data()
    bench_data = pre.preprocess()
    print(f"\nTotal benchmarks: {len(bench_data)}")
    for bench, df in bench_data.items():
        print(f"Benchmark: {bench:30s} | Samples: {len(df)} | Columns: {list(df.columns)[:10]} ... +{len(df.columns)-10} more")
    
    # Test LOOCV split
    training_data = pre.get_training_data()
    test_data = pre.get_test_data() 