import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles loading and preprocessing of PolyBench execution data"""
    
    def __init__(self, data_path: str = "data/exec_times.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.raw_data = None
        self.processed_data = None
        
        # Define column mappings based on actual dataset structure
        self.compiler_flags = [
            '-funsafe-math-optimizations ',
            '-fno-guess-branch-probability ',
            '-fno-ivopts ',
            '-fno-tree-loop-optimize ',
            '-fno-inline-functions ',
            '-funroll-all-loops ',
            '-O2 '
        ]
        
        self.execution_time_cols = [
            'execution_time_1',
            'execution_time_2', 
            'execution_time_3',
            'execution_time_4',
            'execution_time_5'
        ]
        
    def load_and_inspect_data(self) -> pd.DataFrame:
        """Load data and inspect its structure"""
        try:
            self.raw_data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
            logger.info(f"Columns: {list(self.raw_data.columns)}")
            logger.info(f"Unique benchmarks: {self.raw_data['APP_NAME'].unique()}")
            logger.info(f"Number of unique benchmarks: {self.raw_data['APP_NAME'].nunique()}")
            
            # Display sample configurations
            logger.info("Sample flag configurations:")
            for i, row in self.raw_data.head(3).iterrows():
                config = [row[flag] for flag in self.compiler_flags]
                logger.info(f"  Row {i}: {config}")
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self) -> Dict:
        """Complete data preprocessing pipeline"""
        if self.raw_data is None:
            self.load_and_inspect_data()
        
        data = self.raw_data.copy()
        
        # Clean and process the data
        data = self._clean_data(data)
        
        # Process compiler flags (convert X to 1, anything else to 0)
        data = self._process_compiler_flags(data)
        
        # Process execution times (take average and handle scientific notation)
        data = self._process_execution_times(data)
        
        # Split by benchmark
        benchmark_data = self._split_by_benchmark(data)
        
        self.processed_data = benchmark_data
        logger.info("Data preprocessing completed successfully")
        
        return benchmark_data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        initial_shape = data.shape
        logger.info(f"Initial data shape: {initial_shape}")
        
        # Remove rows with missing APP_NAME
        data = data.dropna(subset=['APP_NAME'])
        
        # Remove rows with missing values in execution times
        for col in self.execution_time_cols:
            if col in data.columns:
                data = data.dropna(subset=[col])
        
        logger.info(f"Shape after removing missing values: {data.shape}")
        logger.info(f"Removed {initial_shape[0] - data.shape[0]} rows with missing values")
        
        return data
    
    def _process_compiler_flags(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert compiler flags to binary representation"""
        logger.info("Processing compiler flags...")
        
        for flag in self.compiler_flags:
            if flag in data.columns:
                # Convert 'X' to 1, everything else to 0
                data[flag + '_binary'] = data[flag].apply(
                    lambda x: 1 if str(x).strip().upper() == 'X' else 0
                )
                
                # Log flag distribution
                flag_dist = data[flag + '_binary'].value_counts()
                logger.info(f"Flag {flag.strip()}: {flag_dist.to_dict()}")
            else:
                logger.warning(f"Flag column {flag} not found in data")
                data[flag + '_binary'] = 0  # Default to disabled
        
        return data
    
    def _process_execution_times(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process execution times - handle scientific notation and take statistics"""
        logger.info("Processing execution times...")
        
        # Convert execution time columns to numeric, handling scientific notation
        for col in self.execution_time_cols:
            if col in data.columns:
                # Convert to numeric, handling scientific notation like '2E-05'
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Log some statistics
                logger.info(f"{col} - Mean: {data[col].mean():.6f}, Std: {data[col].std():.6f}")
        
        # Calculate mean execution time across all measurements
        available_time_cols = [col for col in self.execution_time_cols if col in data.columns]
        
        if available_time_cols:
            data['mean_execution_time'] = data[available_time_cols].mean(axis=1)
            data['execution_time_std'] = data[available_time_cols].std(axis=1)
            data['min_execution_time'] = data[available_time_cols].min(axis=1)
            data['max_execution_time'] = data[available_time_cols].max(axis=1)
            
            logger.info(f"Overall execution time statistics:")
            logger.info(f"  Mean: {data['mean_execution_time'].mean():.6f}")
            logger.info(f"  Std: {data['mean_execution_time'].std():.6f}")
            logger.info(f"  Min: {data['mean_execution_time'].min():.6f}")
            logger.info(f"  Max: {data['mean_execution_time'].max():.6f}")
        else:
            logger.error("No execution time columns found!")
            raise ValueError("No execution time columns found in data")
        
        return data
    
    def _remove_outliers_per_benchmark(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers in execution times per benchmark"""
        logger.info("Removing outliers per benchmark...")
        
        cleaned_data = []
        for benchmark in data['APP_NAME'].unique():
            benchmark_data = data[data['APP_NAME'] == benchmark].copy()
            initial_size = len(benchmark_data)
            
            # Remove outliers (> 3 standard deviations from mean)
            mean_time = benchmark_data['mean_execution_time'].mean()
            std_time = benchmark_data['mean_execution_time'].std()
            
            if std_time > 0:  # Avoid division by zero
                lower_bound = mean_time - 3 * std_time
                upper_bound = mean_time + 3 * std_time
                
                benchmark_data = benchmark_data[
                    (benchmark_data['mean_execution_time'] >= lower_bound) &
                    (benchmark_data['mean_execution_time'] <= upper_bound)
                ]
            
            removed = initial_size - len(benchmark_data)
            if removed > 0:
                logger.info(f"Benchmark {benchmark}: Removed {removed} outliers out of {initial_size}")
            
            cleaned_data.append(benchmark_data)
        
        return pd.concat(cleaned_data, ignore_index=True)
    
    def _split_by_benchmark(self, data: pd.DataFrame) -> Dict:
        """Split data by benchmark for individual training"""
        logger.info("Splitting data by benchmark...")
        
        # Remove outliers first
        data = self._remove_outliers_per_benchmark(data)
        
        benchmark_data = {}
        
        for benchmark in data['APP_NAME'].unique():
            benchmark_mask = data['APP_NAME'] == benchmark
            benchmark_df = data[benchmark_mask].copy()
            
            if len(benchmark_df) < 10:  # Skip benchmarks with too few samples
                logger.warning(f"Skipping benchmark {benchmark}: only {len(benchmark_df)} samples")
                continue
            
            # Create feature matrix (compiler flags + code size)
            feature_cols = [flag + '_binary' for flag in self.compiler_flags] + ['code_size']
            
            # Handle missing code_size
            if 'code_size' not in benchmark_df.columns:
                logger.warning(f"code_size not found for {benchmark}, using default value")
                benchmark_df['code_size'] = benchmark_df['mean_execution_time'].median()
            
            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in benchmark_df.columns:
                    benchmark_df[col] = 0
            
            X = benchmark_df[feature_cols].values
            y = benchmark_df['mean_execution_time'].values
            
            # Normalize features
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
            
            # Train-test split
            if len(benchmark_df) >= 20:
                test_size = 0.2
            else:
                test_size = 0.1  # Smaller test set for small datasets
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_normalized, y, test_size=test_size, random_state=42, stratify=None
            )
            
            benchmark_data[benchmark] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X_raw': X,  # Keep raw features for lookup
                'y_raw': y,
                'feature_names': feature_cols,
                'scaler': scaler,
                'data_size': len(benchmark_df),
                'baseline_time': y.mean(),
                'best_time': y.min(),
                'worst_time': y.max()
            }
            
            logger.info(f"Benchmark {benchmark}: {len(benchmark_df)} samples, "
                       f"baseline: {y.mean():.6f}, best: {y.min():.6f}")
        
        logger.info(f"Processed {len(benchmark_data)} benchmarks successfully")
        return benchmark_data