"""
Script to download IMDB dataset and save train/test subsets as parquet files
"""
import os
import pandas as pd
from datasets import load_dataset

def main():
    # Create data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    print("Loading IMDB dataset...")
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Sample subset for quicker development
    train_df = train_df.sample(n=5000, random_state=42)
    # test_df = test_df.sample(n=10, random_state=42)
    test_df = test_df.sample(n=1000, random_state=42) # Increased test size for better evaluation
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Save as parquet files
    print("Saving datasets to parquet files...")
    train_path = "data/raw/train.parquet"
    test_path = "data/raw/test.parquet"
    
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    
    print(f"Train data saved to: {os.path.abspath(train_path)}")
    print(f"Test data saved to: {os.path.abspath(test_path)}")

if __name__ == "__main__":
    main()
