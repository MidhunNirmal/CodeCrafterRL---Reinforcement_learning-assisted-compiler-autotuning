import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loocv_data_preprocessor import LOOCVDataPreprocessor
from loocv_environment import LOOCVCompilerEnvironment
import sys
sys.path.append('../alt_rl_model')
from dqn_agent import DQNAgent

MODEL_PATH = "loocv_expanded_features_dqn_agent.pth"
TEST_RESULTS_CSV = "loocv_detailed_test_results.csv"
TEST_PLOT_PNG = "loocv_test_results.png"
NUM_EPISODES = 20

def test_loocv_model(exclude_benchmark: str = "security_sha"):
    """Test the LOOCV trained model on the excluded benchmark"""
    
    print(f"Testing LOOCV model on excluded benchmark: {exclude_benchmark}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    pre = LOOCVDataPreprocessor(exclude_benchmark=exclude_benchmark)
    pre.load_data()
    training_data = pre.get_training_data()
    test_data = pre.get_test_data()
    
    # Setup environment
    print("Setting up LOOCV environment...")
    env = LOOCVCompilerEnvironment(training_data, test_data)
    state_size = env.observation_space.shape[0]
    action_size = 7
    
    # Load trained model
    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run the training script first.")
        return
    
    print(f"Loading model from {model_path}...")
    
    # Check model compatibility
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
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
    
    # Create agent and load model
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    print(f"Model loaded successfully")
    
    # Evaluate on test benchmark
    print(f"\nEvaluating agent on '{exclude_benchmark}' for {NUM_EPISODES} episodes...")
    
    test_results = env.evaluate_on_test_benchmark(
        agent, 
        exclude_benchmark, 
        num_episodes=NUM_EPISODES
    )
    
    # Print detailed results
    print(f"\n{'='*60}")
    print(f"LOOCV TEST RESULTS FOR '{exclude_benchmark.upper()}'")
    print(f"{'='*60}")
    print(f"Average reward: {test_results['avg_reward']:.4f}")
    print(f"Average improvement: {test_results['avg_improvement']:.4f}")
    print(f"Best improvement: {test_results['best_improvement']:.4f}")
    print(f"Number of episodes: {NUM_EPISODES}")
    
    # Analyze episode results
    episode_improvements = test_results['episode_improvements']
    episode_rewards = test_results['episode_rewards']
    
    positive_improvements = [imp for imp in episode_improvements if imp > 0]
    negative_improvements = [imp for imp in episode_improvements if imp <= 0]
    
    print(f"\nDetailed Analysis:")
    print(f"Episodes with positive improvement: {len(positive_improvements)}/{NUM_EPISODES} ({len(positive_improvements)/NUM_EPISODES*100:.1f}%)")
    print(f"Episodes with negative improvement: {len(negative_improvements)}/{NUM_EPISODES} ({len(negative_improvements)/NUM_EPISODES*100:.1f}%)")
    
    if positive_improvements:
        print(f"Average positive improvement: {np.mean(positive_improvements):.4f}")
        print(f"Best positive improvement: {np.max(positive_improvements):.4f}")
    
    if negative_improvements:
        print(f"Average negative improvement: {np.mean(negative_improvements):.4f}")
        print(f"Worst negative improvement: {np.min(negative_improvements):.4f}")
    
    # Save detailed results
    detailed_results = []
    for i, (reward, improvement) in enumerate(zip(episode_rewards, episode_improvements)):
        detailed_results.append({
            'episode': i + 1,
            'reward': reward,
            'improvement': improvement,
            'positive_improvement': improvement > 0
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_csv = os.path.join(os.path.dirname(__file__), TEST_RESULTS_CSV)
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"\nDetailed results saved to: {detailed_csv}")
    
    # Create visualization
    create_test_visualization(detailed_results, exclude_benchmark)
    
    return test_results

def create_test_visualization(detailed_results, exclude_benchmark):
    """Create visualization of test results"""
    df = pd.DataFrame(detailed_results)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Improvement by episode
    plt.subplot(2, 3, 1)
    colors = ['green' if imp > 0 else 'red' for imp in df['improvement']]
    plt.bar(df['episode'], df['improvement'], color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Improvement')
    plt.title(f'Improvement by Episode\n({exclude_benchmark})')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Reward by episode
    plt.subplot(2, 3, 2)
    plt.plot(df['episode'], df['reward'], 'b-', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Reward by Episode\n({exclude_benchmark})')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Improvement distribution
    plt.subplot(2, 3, 3)
    plt.hist(df['improvement'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No improvement')
    plt.xlabel('Improvement')
    plt.ylabel('Frequency')
    plt.title(f'Improvement Distribution\n({exclude_benchmark})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Positive vs Negative improvements
    plt.subplot(2, 3, 4)
    positive_count = len(df[df['improvement'] > 0])
    negative_count = len(df[df['improvement'] <= 0])
    plt.pie([positive_count, negative_count], 
            labels=['Positive', 'Negative'], 
            colors=['lightgreen', 'lightcoral'],
            autopct='%1.1f%%',
            startangle=90)
    plt.title(f'Improvement Success Rate\n({exclude_benchmark})')
    
    # Plot 5: Cumulative improvement
    plt.subplot(2, 3, 5)
    cumulative_improvement = df['improvement'].cumsum()
    plt.plot(df['episode'], cumulative_improvement, 'g-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Improvement')
    plt.title(f'Cumulative Improvement\n({exclude_benchmark})')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    stats_text = f"""
    Summary Statistics for {exclude_benchmark}:
    
    Total Episodes: {len(df)}
    Average Improvement: {df['improvement'].mean():.4f}
    Best Improvement: {df['improvement'].max():.4f}
    Worst Improvement: {df['improvement'].min():.4f}
    
    Positive Episodes: {positive_count} ({positive_count/len(df)*100:.1f}%)
    Negative Episodes: {negative_count} ({negative_count/len(df)*100:.1f}%)
    
    Average Reward: {df['reward'].mean():.4f}
    Total Reward: {df['reward'].sum():.4f}
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), TEST_PLOT_PNG)
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Test visualization saved to: {plot_path}")
    plt.close()

def main():
    print("LOOCV Model Tester")
    print("="*50)
    
    # Test the model
    test_results = test_loocv_model("security_sha")
    
    if test_results:
        print(f"\n{'='*50}")
        print("TESTING COMPLETED SUCCESSFULLY!")
        print(f"{'='*50}")
        print(f"Model tested on: security_sha")
        print(f"Results saved to: {TEST_RESULTS_CSV}")
        print(f"Visualization saved to: {TEST_PLOT_PNG}")

if __name__ == "__main__":
    main() 