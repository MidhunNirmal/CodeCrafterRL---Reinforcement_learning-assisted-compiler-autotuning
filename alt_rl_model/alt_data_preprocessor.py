import pandas as pd
import numpy as np
import os
from typing import Dict

class AltDataPreprocessor:
    """Preprocessor for alternatedata/raw_data/data.csv for RL model training"""
    def __init__(self, data_path: str = "alternatedata/raw_data/data.csv"):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.flag_columns = None
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
        # Drop code_size if present
        if 'code_size' in data.columns:
            data = data.drop('code_size', axis=1)
        # Identify columns
        all_columns = list(data.columns)
        self.exec_time_columns = all_columns[-5:]
        self.flag_columns = [col for col in all_columns if col not in [self.benchmark_column] + self.exec_time_columns]
        # Convert flag columns to binary (1 for X, 0 otherwise)
        for flag in self.flag_columns:
            data[flag] = (data[flag] == 'X').astype(int)
        # Compute mean execution time
        data['mean_exec_time'] = data[self.exec_time_columns].mean(axis=1)
        # Split by benchmark
        benchmarks = data[self.benchmark_column].unique()
        benchmark_data = {}
        for bench in benchmarks:
            bench_df = data[data[self.benchmark_column] == bench].copy()
            # Features: flag columns; Target: mean_exec_time
            features = bench_df[self.flag_columns]
            target = bench_df['mean_exec_time']
            bench_df_out = features.copy()
            bench_df_out['mean_exec_time'] = target
            benchmark_data[bench] = bench_df_out.reset_index(drop=True)
        self.processed_data = benchmark_data
        return benchmark_data

if __name__ == "__main__":
    pre = AltDataPreprocessor()
    pre.load_data()
    bench_data = pre.preprocess()
    print(f"Total benchmarks: {len(bench_data)}")
    for bench, df in bench_data.items():
        print(f"Benchmark: {bench:30s} | Samples: {len(df)} | Columns: {list(df.columns)}") 