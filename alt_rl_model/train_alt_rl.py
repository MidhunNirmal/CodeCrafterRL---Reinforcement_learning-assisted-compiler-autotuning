import os
import numpy as np
import torch
from alt_data_preprocessor import AltDataPreprocessor
from alt_comp_env import AltCompilerEnvironment
from dqn_agent import DQNAgent

EPISODES = 500
SAVE_PATH = "dqn_agent_{}.pth"

def train_on_benchmark(benchmark_name, bench_data):
    env = AltCompilerEnvironment(bench_data, benchmark_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.nvec[0]
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    best_improvement = -np.inf
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            if done:
                if info['improvement'] > best_improvement:
                    best_improvement = info['improvement']
                break
        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1}/{EPISODES} | Last reward: {total_reward:.4f} | Best improvement: {best_improvement:.4f}")
    # Save model
    agent.save(os.path.join(os.path.dirname(__file__), SAVE_PATH.format(benchmark_name)))
    print(f"Training complete for {benchmark_name}. Best improvement: {best_improvement:.4f}")

def main():
    pre = AltDataPreprocessor()
    pre.load_data()
    bench_data = pre.preprocess()
    # Pick the first benchmark for demonstration
    benchmark_name = list(bench_data.keys())[0]
    print(f"Training RL agent on benchmark: {benchmark_name}")
    train_on_benchmark(benchmark_name, bench_data)

if __name__ == "__main__":
    main() 