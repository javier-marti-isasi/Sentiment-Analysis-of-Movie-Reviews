"""
Script to view examples of movie reviews from the IMDB dataset
"""
import os
import pandas as pd
import random
from datasets import load_dataset

def load_data():
    """
    Load data from parquet files if they exist, otherwise download from the dataset
    """
    train_path = "data/raw/train.parquet"
    test_path = "data/raw/test.parquet"
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading data from parquet files...")
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
    else:
        print("Parquet files not found. Loading from IMDB dataset...")
        dataset = load_dataset("imdb")
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
    
    return train_df, test_df

def display_review(review, index, label):
    """
    Format and display a single review
    """
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"\n{'='*80}")
    print(f"Review #{index+1} - {sentiment}")
    print(f"{'-'*80}")
    
    # Display first 500 characters with ellipsis if longer
    review_text = review[:500]
    if len(review) > 500:
        review_text += "..."
    
    print(review_text)
    print(f"{'='*80}")

def main():
    # Load data
    train_df, test_df = load_data()
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Display options
    while True:
        print("\nOptions:")
        print("1. View random positive reviews")
        print("2. View random negative reviews")
        print("3. View random mixed reviews")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            # Filter positive reviews
            positive_reviews = train_df[train_df['label'] == 1]
            # Select 3 random positive reviews
            sample_reviews = positive_reviews.sample(min(3, len(positive_reviews)))
            
            for i, (_, row) in enumerate(sample_reviews.iterrows()):
                display_review(row['text'], i, row['label'])
                
        elif choice == '2':
            # Filter negative reviews
            negative_reviews = train_df[train_df['label'] == 0]
            # Select 3 random negative reviews
            sample_reviews = negative_reviews.sample(min(3, len(negative_reviews)))
            
            for i, (_, row) in enumerate(sample_reviews.iterrows()):
                display_review(row['text'], i, row['label'])
                
        elif choice == '3':
            # Select 6 random reviews (3 positive, 3 negative if possible)
            positive_reviews = train_df[train_df['label'] == 1].sample(min(3, len(train_df[train_df['label'] == 1])))
            negative_reviews = train_df[train_df['label'] == 0].sample(min(3, len(train_df[train_df['label'] == 0])))
            
            # Combine and shuffle
            sample_reviews = pd.concat([positive_reviews, negative_reviews]).sample(frac=1)
            
            for i, (_, row) in enumerate(sample_reviews.iterrows()):
                display_review(row['text'], i, row['label'])
                
        elif choice == '4':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
