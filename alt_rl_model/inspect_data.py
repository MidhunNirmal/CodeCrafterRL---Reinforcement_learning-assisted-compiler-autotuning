import pandas as pd
import os

def inspect_data():
    """Inspect the data.csv file to understand its structure"""
    data_path = "alternatedata/raw_data/data.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return
    
    # Load the data
    print("="*80)
    print("DATA INSPECTION")
    print("="*80)
    
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    print(f"\n" + "="*80)
    print("ALL COLUMN NAMES")
    print("="*80)
    
    for i, col in enumerate(df.columns):
        print(f"{i+1:3d}. {col}")
    
    print(f"\n" + "="*80)
    print("COLUMN TYPES")
    print("="*80)
    
    print(df.dtypes.value_counts())
    
    print(f"\n" + "="*80)
    print("FIRST 10 COLUMNS DETAILED")
    print("="*80)
    
    for i, col in enumerate(df.columns[:10]):
        print(f"\n{i+1}. {col}")
        print(f"   Type: {df[col].dtype}")
        print(f"   Unique values: {df[col].nunique()}")
        print(f"   Sample values: {list(df[col].unique()[:5])}")
    
    print(f"\n" + "="*80)
    print("LAST 10 COLUMNS DETAILED")
    print("="*80)
    
    for i, col in enumerate(df.columns[-10:]):
        print(f"\n{len(df.columns)-9+i}. {col}")
        print(f"   Type: {df[col].dtype}")
        print(f"   Unique values: {df[col].nunique()}")
        if df[col].dtype in ['int64', 'float64']:
            print(f"   Min: {df[col].min()}")
            print(f"   Max: {df[col].max()}")
            print(f"   Mean: {df[col].mean():.2f}")
    
    print(f"\n" + "="*80)
    print("BENCHMARK ANALYSIS")
    print("="*80)
    
    if 'APP_NAME' in df.columns:
        benchmarks = df['APP_NAME'].value_counts()
        print(f"Total benchmarks: {len(benchmarks)}")
        print(f"Benchmarks:")
        for bench, count in benchmarks.items():
            print(f"  {bench}: {count} samples")
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Total columns: {len(df.columns)}")
    print(f"✅ Total samples: {len(df)}")
    print(f"✅ Column types: {dict(df.dtypes.value_counts())}")
    print(f"✅ Ready for analysis!")

if __name__ == "__main__":
    inspect_data() 