import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_model import AltRLModelTester

def main():
    tester = AltRLModelTester(model_path="../unified_alt_dqn_agent.pth")
    # Check input size
    expected_state_size = tester.unified_env.observation_space.shape[0]
    model_state_dict = None
    try:
        import torch
        checkpoint = torch.load(os.path.join(os.path.dirname(__file__), '../unified_alt_dqn_agent.pth'), map_location='cpu')
        model_state_dict = checkpoint['q_network_state_dict']
        # Get the input size from the saved model's first layer
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
    all_results = tester.test_all_benchmarks(num_episodes=5)
    summary_df = tester.analyze_results(all_results)
    print("\nSummary:")
    print(summary_df)
    summary_csv = os.path.join(os.path.dirname(__file__), 'unified_alt_test_results.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to {summary_csv}")

if __name__ == "__main__":
    main() 