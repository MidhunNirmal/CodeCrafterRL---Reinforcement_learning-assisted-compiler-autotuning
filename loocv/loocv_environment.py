import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class LOOCVCompilerEnvironment(gym.Env):
    """LOOCV RL environment for compiler optimization with separate training and test data"""
    def __init__(self, training_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame] = None, benchmark_encoder: LabelEncoder = None):
        super(LOOCVCompilerEnvironment, self).__init__()
        
        self.training_data = training_data
        self.test_data = test_data
        self.training_benchmarks = list(training_data.keys())
        self.test_benchmarks = list(test_data.keys()) if test_data else []
        
        # Create benchmark encoder if not provided
        if benchmark_encoder is None:
            self.benchmark_encoder = LabelEncoder()
            self.benchmark_encoder.fit(self.training_benchmarks)
        else:
            self.benchmark_encoder = benchmark_encoder
        
        # Action space: 7 compiler flags (binary)
        self.action_space = spaces.MultiBinary(7)
        
        # Get feature dimensions from the first training benchmark
        first_benchmark = list(training_data.keys())[0]
        first_df = training_data[first_benchmark]
        # Count features: flags + static features (excluding mean_exec_time)
        flag_cols = [col for col in first_df.columns if col in [
            'funsafe_math_optimizations', 'fno_guess_branch_probability', 'fno_ivopts',
            'fno_tree_loop_optimize', 'fno_inline_functions', 'funroll_all_loops', 'o2'
        ]]
        static_feature_cols = [col for col in first_df.columns if col not in flag_cols + ['mean_exec_time']]
        
        num_flags = len(flag_cols)
        num_static_features = len(static_feature_cols)
        benchmark_encoding_size = len(self.training_benchmarks)
        
        # Observation space: 
        # - 7 compiler flags
        # - all static features (normalized)
        # - benchmark encoding (one-hot)
        # - 3 context features (performance, steps, improvement)
        total_obs_size = num_flags + num_static_features + benchmark_encoding_size + 3
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )
        
        # Store feature information
        self.flag_columns = flag_cols
        self.static_feature_columns = static_feature_cols
        self.num_flags = num_flags
        self.num_static_features = num_static_features
        
        # Environment state
        self.current_config = None
        self.current_benchmark = None
        self.current_benchmark_data = None
        self.step_count = 0
        self.max_steps = 50
        
        # Performance tracking
        self.baseline_time = None
        self.best_time = None
        self.worst_time = None
        self.current_time = None
        
        # Create performance models for training benchmarks
        self.performance_models = {}
        self._create_performance_models()
        
        logger.info(f"LOOCV environment initialized for {len(self.training_benchmarks)} training benchmarks")
        logger.info(f"State size: {total_obs_size} (7 flags + {num_static_features} static features + {benchmark_encoding_size} benchmarks + 3 context)")
        logger.info(f"Action size: 7 (7 binary flags)")
        
    def _create_performance_models(self):
        """Create performance lookup models for training benchmarks"""
        for benchmark_name in self.training_benchmarks:
            df = self.training_data[benchmark_name]
            
            # Only use the 7 flag columns for performance prediction
            flag_cols = [col for col in df.columns if col in [
                'funsafe_math_optimizations', 'fno_guess_branch_probability', 'fno_ivopts',
                'fno_tree_loop_optimize', 'fno_inline_functions', 'funroll_all_loops', 'o2'
            ]]
            
            X_raw = df[flag_cols].values
            y_raw = df['mean_exec_time'].values
            
            # Create exact lookup
            exact_lookup = {}
            for config, time in zip(X_raw, y_raw):
                config_tuple = tuple(config.astype(int))
                if config_tuple not in exact_lookup:
                    exact_lookup[config_tuple] = []
                exact_lookup[config_tuple].append(time)
            
            # Average multiple measurements
            for config in exact_lookup:
                exact_lookup[config] = np.mean(exact_lookup[config])
            
            # Create nearest neighbor model (only for flags)
            if len(X_raw) > 5:
                nn_model = NearestNeighbors(n_neighbors=3, metric='hamming')
                nn_model.fit(X_raw)
                nn_data = {'X': X_raw, 'y': y_raw, 'model': nn_model}
            else:
                nn_data = None
            
            self.performance_models[benchmark_name] = {
                'exact_lookup': exact_lookup,
                'nn_model': nn_data,
                'baseline_time': np.mean(y_raw),
                'best_time': np.min(y_raw),
                'worst_time': np.max(y_raw)
            }
    
    def reset(self, benchmark_name: str = None):
        """Reset environment to initial state"""
        # Select benchmark if not specified (only from training benchmarks)
        if benchmark_name is None:
            benchmark_name = np.random.choice(self.training_benchmarks)
        elif benchmark_name not in self.training_benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not in training data")
        
        self.current_benchmark = benchmark_name
        self.current_benchmark_data = self.training_data[benchmark_name]
        
        # Start with a random configuration
        self.current_config = np.random.randint(0, 2, size=7)
        self.step_count = 0
        
        # Set performance metrics for this benchmark
        perf_model = self.performance_models[benchmark_name]
        self.baseline_time = perf_model['baseline_time']
        self.best_time = perf_model['best_time']
        self.worst_time = perf_model['worst_time']
        self.current_time = self._get_execution_time(self.current_config)
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        self.step_count += 1
        
        # Ensure action is the right shape and type
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        action = action.astype(int).flatten()[:7]  # Ensure 7 flags
        
        # Update configuration
        self.current_config = action.copy()
        
        # Get execution time for this configuration
        execution_time = self._get_execution_time(self.current_config)
        self.current_time = execution_time
        
        # Calculate reward
        reward = self._calculate_reward(execution_time)
        
        # Check if done
        done = self.step_count >= self.max_steps
        
        # Calculate improvement percentage
        improvement = (self.baseline_time - execution_time) / self.baseline_time
        
        info = {
            'execution_time': execution_time,
            'best_time': self.best_time,
            'baseline_time': self.baseline_time,
            'improvement': improvement,
            'config': self.current_config.copy(),
            'benchmark': self.current_benchmark,
            'step': self.step_count
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current observation (state) with benchmark context"""
        # Normalize performance metrics
        normalized_current = (self.current_time - self.best_time) / (self.worst_time - self.best_time + 1e-8)
        normalized_current = np.clip(normalized_current, 0, 1)
        
        steps_remaining = (self.max_steps - self.step_count) / self.max_steps
        
        # Benchmark encoding (one-hot)
        benchmark_encoding = np.zeros(len(self.training_benchmarks))
        benchmark_idx = self.benchmark_encoder.transform([self.current_benchmark])[0]
        benchmark_encoding[benchmark_idx] = 1.0
        
        # Improvement from baseline (can be negative)
        improvement_from_baseline = (self.baseline_time - self.current_time) / self.baseline_time
        
        # Get static features from current benchmark data (use mean across all samples)
        static_features = self.current_benchmark_data[self.static_feature_columns].mean().values
        
        # Combine all features
        observation = np.concatenate([
            self.current_config.astype(np.float32),  # 7 flags
            static_features.astype(np.float32),      # static features
            benchmark_encoding.astype(np.float32),   # benchmark encoding
            np.array([normalized_current, steps_remaining, improvement_from_baseline], dtype=np.float32)  # context
        ])
        
        return observation
    
    def _get_execution_time(self, config):
        """Get execution time for a configuration using exact lookup or nearest neighbor"""
        perf_model = self.performance_models[self.current_benchmark]
        exact_lookup = perf_model['exact_lookup']
        
        # Try exact lookup first
        config_tuple = tuple(config.astype(int))
        if config_tuple in exact_lookup:
            return exact_lookup[config_tuple]
        
        # Fall back to nearest neighbor
        nn_data = perf_model['nn_model']
        if nn_data is not None:
            distances, indices = nn_data['model'].kneighbors([config])
            weights = 1.0 / (distances[0] + 1e-8)
            weighted_time = np.average(nn_data['y'][indices[0]], weights=weights)
            return weighted_time
        
        # Fall back to baseline
        return perf_model['baseline_time']
    
    def _calculate_reward(self, execution_time):
        """Calculate reward based on execution time improvement"""
        improvement = (self.baseline_time - execution_time) / self.baseline_time
        
        # Reward function: positive for improvement, negative for degradation
        if improvement > 0:
            reward = improvement * 10  # Scale up positive rewards
        else:
            reward = improvement * 5   # Scale down negative rewards
        
        return reward
    
    def get_benchmark_encoder(self):
        """Get the benchmark encoder for use in testing"""
        return self.benchmark_encoder
    
    def evaluate_on_test_benchmark(self, agent, test_benchmark_name: str, num_episodes: int = 5):
        """Evaluate the trained agent on a test benchmark"""
        if test_benchmark_name not in self.test_data:
            raise ValueError(f"Test benchmark '{test_benchmark_name}' not found in test data")
        
        test_df = self.test_data[test_benchmark_name]
        
        # Create performance model for test benchmark
        flag_cols = [col for col in test_df.columns if col in [
            'funsafe_math_optimizations', 'fno_guess_branch_probability', 'fno_ivopts',
            'fno_tree_loop_optimize', 'fno_inline_functions', 'funroll_all_loops', 'o2'
        ]]
        
        X_raw = test_df[flag_cols].values
        y_raw = test_df['mean_exec_time'].values
        
        # Create exact lookup for test benchmark
        exact_lookup = {}
        for config, time in zip(X_raw, y_raw):
            config_tuple = tuple(config.astype(int))
            if config_tuple not in exact_lookup:
                exact_lookup[config_tuple] = []
            exact_lookup[config_tuple].append(time)
        
        for config in exact_lookup:
            exact_lookup[config] = np.mean(exact_lookup[config])
        
        baseline_time = np.mean(y_raw)
        best_time = np.min(y_raw)
        worst_time = np.max(y_raw)
        
        # Evaluate agent
        episode_rewards = []
        episode_improvements = []
        
        for episode in range(num_episodes):
            # Start with random configuration
            current_config = np.random.randint(0, 2, size=7)
            total_reward = 0
            best_improvement = -np.inf
            
            for step in range(self.max_steps):
                # Create observation for current state
                normalized_current = (self._get_test_execution_time(current_config, exact_lookup) - best_time) / (worst_time - best_time + 1e-8)
                normalized_current = np.clip(normalized_current, 0, 1)
                
                steps_remaining = (self.max_steps - step) / self.max_steps
                
                # Use first training benchmark encoding (since test benchmark not in training)
                benchmark_encoding = np.zeros(len(self.training_benchmarks))
                benchmark_encoding[0] = 1.0
                
                improvement_from_baseline = (baseline_time - self._get_test_execution_time(current_config, exact_lookup)) / baseline_time
                
                # Get static features from test benchmark data
                static_features = test_df[self.static_feature_columns].mean().values
                
                observation = np.concatenate([
                    current_config.astype(np.float32),
                    static_features.astype(np.float32),
                    benchmark_encoding.astype(np.float32),
                    np.array([normalized_current, steps_remaining, improvement_from_baseline], dtype=np.float32)
                ])
                
                # Get action from agent
                action = agent.act(observation, training=False)
                # Print model input and output
                print(f"Episode {episode+1}, Step {step+1} | Model input (observation): {observation}")
                print(f"Episode {episode+1}, Step {step+1} | Model output (predicted action): {action}")
                current_config = action.copy()
                
                # Calculate reward
                execution_time = self._get_test_execution_time(current_config, exact_lookup)
                improvement = (baseline_time - execution_time) / baseline_time
                
                if improvement > 0:
                    reward = improvement * 10
                else:
                    reward = improvement * 5
                
                total_reward += reward
                best_improvement = max(best_improvement, improvement)
            
            episode_rewards.append(total_reward)
            episode_improvements.append(best_improvement)
        
        return {
            'avg_reward': np.mean(episode_rewards),
            'avg_improvement': np.mean(episode_improvements),
            'best_improvement': np.max(episode_improvements),
            'episode_rewards': episode_rewards,
            'episode_improvements': episode_improvements
        }
    
    def _get_test_execution_time(self, config, exact_lookup):
        """Get execution time for test benchmark configuration"""
        config_tuple = tuple(config.astype(int))
        if config_tuple in exact_lookup:
            return exact_lookup[config_tuple]
        return np.mean(list(exact_lookup.values()))  # Return average if not found 