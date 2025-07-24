import os
from alt_data_preprocessor import AltDataPreprocessor

def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'processed')
    os.makedirs(out_dir, exist_ok=True)
    pre = AltDataPreprocessor()
    pre.load_data()
    bench_data = pre.preprocess()
    for bench, df in bench_data.items():
        out_path = os.path.join(out_dir, f"{bench}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} | Shape: {df.shape}")
    print(f"Total benchmarks processed: {len(bench_data)}")

if __name__ == "__main__":
    main() 