import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
import json


import os
import time
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from data_preprocess import DataPreprocessor
from comp_emv import CompilerEnvironment
from dqn_agent import DQNAgent

logger = logging.getLogger(__name__)

class CompilerOptimizationTrainer:
    """Main trainer class for RL compiler optimization"""
    
    def __init__(self, data_path: str = "exec_times.csv"):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor(data_path)
        self.benchmark_data = None
        self.agents = {}
        self.environments = {}
        self.training_stats = defaultdict(list)
        
    def setup(self):
        """Setup data and create agents/environments"""
        logger.info("Setting up trainer...")
        
        # Load and preprocess data
        self.benchmark_data = self.preprocessor.preprocess_data()
        
        if not self.benchmark_data:
            raise ValueError("No benchmark data available after preprocessing!")
        
        # Create agents and environments for each benchmark
        for benchmark_name in self.benchmark_data.keys():
            logger.info(f"Setting up agent for benchmark: {benchmark_name}")
            
            # Create environment
            env = CompilerEnvironment(self.benchmark_data, benchmark_name)
            self.environments[benchmark_name] = env
            
            # Create agent
            state_size = env.observation_space.shape[0]
            
            agent = DQNAgent(
                state_size=state_size,
                action_size=7,  # 7 compiler flags
                lr=0.001,
                gamma=0.95,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                memory_size=10000,
                batch_size=32,
                target_update=100
            )
            
            self.agents[benchmark_name] = agent
            
        logger.info(f"Setup complete for {len(self.benchmark_data)} benchmarks")

    def train_single_benchmark(
        self, 
        benchmark_name: str, 
        episodes: int = 1000,
        save_interval: int = 200,
        log_interval: int = 50,
        save_path: str = "checkpoints"
    ):
        """Train DQN agent on a single benchmark"""
        logger.info(f"Starting training for benchmark: {benchmark_name}")
        
        env = self.environments[benchmark_name]
        agent = self.agents[benchmark_name]

        os.makedirs(save_path, exist_ok=True)

        episode_rewards = []

        for episode in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay()
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            self.training_stats[benchmark_name].append({
                'episode': episode,
                'reward': total_reward,
                'best_time': info['best_time'],
                'final_time': info['execution_time'],
                'config': info['config'].tolist()
            })

            if episode % log_interval == 0:
                logger.info(
                    f"[{benchmark_name}] Episode {episode}/{episodes} - Reward: {total_reward:.2f}, "
                    f"Epsilon: {agent.epsilon:.4f}, Last Exec Time: {info['execution_time']:.6f}"
                )

            if episode % save_interval == 0:
                benchmark_dir = os.path.join(save_path, os.path.dirname(benchmark_name))
                os.makedirs(benchmark_dir, exist_ok=True)
                model_path = os.path.join(save_path, f"{benchmark_name}_ep{episode}.pth")
                # agent.save(os.path.join(benchmark_dir, f"{benchmark_name}_ep{episode}.pth"))
                agent.save(model_path)

        logger.info(f"Finished training for {benchmark_name}")
        self._plot_rewards(benchmark_name, episode_rewards)

    def _plot_rewards(self, benchmark_name: str, rewards: list):
        """Plot training rewards"""
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=rewards)
        plt.title(f"Reward Curve - {benchmark_name}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join("reward_curves", f"{benchmark_name}_reward_curve.png")
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(output_path)
        plt.close()
    
    def export_training_log(self, path="training_log.json"):
        """Save training stats to JSON"""
        logger.info(f"Saving training stats to {path}")
        with open(path, "w") as f:
            json.dump(self.training_stats, f, indent=4)

    def train_all_benchmarks(self, episodes=1000):
        """Train all benchmarks sequentially"""
        logger.info("Training all benchmarks...")
        for benchmark in self.benchmark_data.keys():
            self.train_single_benchmark(benchmark, episodes=episodes)
        self.export_training_log()