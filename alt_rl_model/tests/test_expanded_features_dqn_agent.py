import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from alt_data_preprocessor_correct import AltDataPreprocessorCorrect
from alt_comp_env_unified import AltUnifiedCompilerEnvironment
from test_model import AltRLModelTester
from visualize_results import AltRLVisualizer

def main():
    # Use the same preprocessing as training
    pre = AltDataPreprocessorCorrect()
    pre.load_data()
    benchmark_data = pre.preprocess()
    unified_env = AltUnifiedCompilerEnvironment(benchmark_data)

    # Print feature info safely
    if pre.flag_columns is not None:
        print(f"Flag columns ({len(pre.flag_columns)}): {pre.flag_columns}")
    else:
        print("Flag columns: None")
    if pre.static_feature_columns is not None:
        print(f"Static feature columns ({len(pre.static_feature_columns)}): {pre.static_feature_columns[:5]} ... {pre.static_feature_columns[-5:]}")
    else:
        print("Static feature columns: None")
    print(f"State size: {unified_env.observation_space.shape[0]}")

    expected_state_size = unified_env.observation_space.shape[0]
    model_state_dict = None
    try:
        import torch
        checkpoint = torch.load(os.path.join(os.path.dirname(__file__), '../expanded_features_dqn_agent1.pth'), map_location='cpu')
        model_state_dict = checkpoint['q_network_state_dict']
        fc1_weight = model_state_dict['fc1.weight']
        trained_input_size = fc1_weight.shape[1]
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return
    if trained_input_size != expected_state_size:
        print(f"ERROR: Model input size ({trained_input_size}) does not match environment state size ({expected_state_size}).")
        print("You must retrain the model with the current features.")
        return
    print("Input size matches. Proceeding with evaluation...")

    # Use AltRLModelTester for evaluation
    tester = AltRLModelTester(model_path="../expanded_features_dqn_agent.pth")
    all_results = tester.test_all_benchmarks(num_episodes=5)
    summary_df = tester.analyze_results(all_results)
    if summary_df is None:
        print("No summary results to save or visualize.")
        return
    print("\nSummary:")
    print(summary_df)
    summary_csv = os.path.join(os.path.dirname(__file__), 'expanded_features_test_results.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to {summary_csv}")
    # Visualization
    visualizer = AltRLVisualizer()
    visualizer.create_training_visualizations(
        episode_rewards=summary_df['avg_reward'].tolist(),
        best_improvements=dict(zip(summary_df['benchmark'], summary_df['best_improvement'])),
        benchmark_names=summary_df['benchmark'].tolist()
    )

if __name__ == "__main__":
    main() 