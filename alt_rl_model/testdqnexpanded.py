import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alt_data_preprocessor_correct import AltDataPreprocessorCorrect
from alt_comp_env_unified import AltUnifiedCompilerEnvironment
from dqn_agent import DQNAgent

SAVE_PATH = "alt_rl_model/expanded_features_dqn_agent1.pth"
RESULTS_CSV = "expanded_features_eval_results.csv"
PLOT_PNG = "expanded_features_improvement_barplot.png"
NUM_EPISODES = 5

def evaluate_agent(agent, env, benchmark_names, num_episodes=5):
    improvements = {}
    total_rewards = {}

    for name in benchmark_names:
        bench_rewards = []
        bench_improvements = []
        for ep in range(num_episodes):
            state = env.reset(name)
            total_reward = 0
            done = False
            while not done:
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                state = next_state
                total_reward += reward
            bench_rewards.append(total_reward)
            bench_improvements.append(info.get("improvement", 0))
        improvements[name] = np.mean(bench_improvements)
        total_rewards[name] = np.mean(bench_rewards)
        print(f"[{name}] -> Avg Reward: {np.mean(bench_rewards):.2f}, Avg Improvement: {np.mean(bench_improvements):.4f}")
    return improvements, total_rewards

def main():
    print("Loading and preprocessing test data...")
    pre = AltDataPreprocessorCorrect()
    pre.load_data()
    benchmark_data = pre.preprocess()
    benchmark_names = list(benchmark_data.keys())

    print("Setting up environment...")
    env = AltUnifiedCompilerEnvironment(benchmark_data)
    state_size = env.observation_space.shape[0]
    action_size = 7  # Same as training

    # Input size check
    try:
        checkpoint = torch.load(SAVE_PATH, map_location='cpu')
        model_state_dict = checkpoint['q_network_state_dict']
        fc1_weight = model_state_dict['fc1.weight']
        trained_input_size = fc1_weight.shape[1]
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return
    if trained_input_size != state_size:
        print(f"ERROR: Model input size ({trained_input_size}) does not match environment state size ({state_size}).")
        print("You must retrain the model with the current features.")
        return
    print("Input size matches. Proceeding with evaluation...")

    agent = DQNAgent(state_size, action_size)
    agent.load(SAVE_PATH)
    print(f"Model loaded from {SAVE_PATH}")

    print(f"\nEvaluating agent on all benchmarks for {NUM_EPISODES} episodes each...\n")
    improvements, rewards = evaluate_agent(agent, env, benchmark_names, num_episodes=NUM_EPISODES)

    # Save results to CSV
    df = pd.DataFrame({
        'benchmark': list(improvements.keys()),
        'avg_improvement': list(improvements.values()),
        'avg_reward': [rewards[k] for k in improvements.keys()]
    })
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved to {RESULTS_CSV}")

    # Plot improvements
    plt.figure(figsize=(12, 6))
    sorted_df = df.sort_values('avg_improvement', ascending=False)
    bars = plt.bar(sorted_df['benchmark'], sorted_df['avg_improvement'], color='skyblue', alpha=0.8)
    plt.xlabel('Benchmark')
    plt.ylabel('Average Improvement')
    plt.title('Average Improvement per Benchmark (Expanded Features DQN)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=200)
    print(f"Improvement bar plot saved to {PLOT_PNG}")

    # Print detailed summary
    print("\n=== Evaluation Summary ===")
    print(f"Average reward: {np.mean(list(rewards.values())):.4f}")
    print(f"Average improvement: {np.mean(list(improvements.values())):.4f}")
    success_count = sum(1 for val in improvements.values() if val > 0)
    print(f"Successful optimizations: {success_count}/{len(benchmark_names)}")
    print("\nTop 5 Benchmarks by Improvement:")
    for name, imp in sorted(improvements.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {imp:.4f}")

if __name__ == "__main__":
    main()
