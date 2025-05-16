"""
Script to test the DistilBERT model on movie review examples
"""
import os
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def load_data():
    """
    Load data from parquet files if they exist, otherwise download from the dataset
    """
    test_path = "data/raw/test.parquet"
    
    if os.path.exists(test_path) and os.path.exists(test_path):
        print("Loading data from parquet files...")
        test_df = pd.read_parquet(test_path)
    else:
        print("Parquet files not found. Loading from IMDB dataset...")
        dataset = load_dataset("imdb")
        test_df = pd.DataFrame(dataset['test'])
    
    return test_df

def initialize_model():
    """
    Initialize the sentiment analysis model from local directory if available
    """
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model_dir = os.path.join("model", "classification_model", model_name)
    
    print("Initializing sentiment analysis model...")
    
    # Check if model exists locally
    if os.path.exists(model_dir):
        print(f"Loading model from local directory: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    else:
        print(f"Local model not found at {model_dir}")
        print("Loading model from Hugging Face (run src/download_model.py to save locally)")
        sentiment_model = pipeline(
            "sentiment-analysis", 
            model=model_name
        )
    
    return sentiment_model

def analyze_sentiment(model, text):
    """
    Analyze sentiment for a single text and measure inference time
    """
    start_time = time.time()
    result = model(text, truncation=True, max_length=512) # TODO: handle long texts in a better way
    end_time = time.time()
    
    inference_time = end_time - start_time
    
    # Format the result
    sentiment = result[0]['label']
    score = result[0]['score']
    
    return {
        'sentiment': sentiment,
        'score': score,
        'inference_time': inference_time
    }

def test_sample_reviews(model, df, num_samples=5):
    """
    Test the model on a sample of reviews from the dataset
    """
    # Get balanced samples (equal number of positive and negative)
    positive_samples = df[df['label'] == 1].sample(num_samples // 2)
    negative_samples = df[df['label'] == 0].sample(num_samples // 2)
    
    samples = pd.concat([positive_samples, negative_samples]).sample(frac=1)
    
    correct_predictions = 0
    total_inference_time = 0
    
    print(f"\nTesting model on {len(samples)} sample reviews...")
    
    for i, (_, row) in enumerate(samples.iterrows()):
        text = row['text']
        actual_label = "POSITIVE" if row['label'] == 1 else "NEGATIVE"
        
        # Truncate text for display
        display_text = text[:100] + "..." if len(text) > 100 else text
        
        print(f"\n{'-'*80}")
        print(f"Review #{i+1}:")
        print(f"Text: {display_text}")
        print(f"Actual sentiment: {actual_label}")
        
        # Analyze sentiment
        result = analyze_sentiment(model, text)
        predicted_sentiment = result['sentiment']
        confidence = result['score']
        inference_time = result['inference_time']
        
        print(f"Predicted sentiment: {predicted_sentiment} (confidence: {confidence:.4f})")
        print(f"Inference time: {inference_time:.4f} seconds")
        
        # Check if prediction is correct
        is_correct = (predicted_sentiment == actual_label)
        if is_correct:
            correct_predictions += 1
            print("✓ Correct prediction")
        else:
            print("✗ Incorrect prediction")
        
        total_inference_time += inference_time
    
    # Calculate accuracy and average inference time
    accuracy = correct_predictions / len(samples)
    avg_inference_time = total_inference_time / len(samples)
    
    print(f"\n{'-'*80}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Average inference time: {avg_inference_time:.4f} seconds per review")
    
    return accuracy, avg_inference_time

def test_custom_reviews(model):
    """
    Test the model on custom reviews entered by the user
    """
    print("\nEnter your own movie reviews to test (type 'exit' to quit):")
    
    while True:
        review = input("\nEnter a movie review: ")
        
        if review.lower() == 'exit':
            break
        
        if not review.strip():
            print("Review cannot be empty. Please try again.")
            continue
        
        # Analyze sentiment
        result = analyze_sentiment(model, review)
        
        print(f"\nSentiment: {result['sentiment']}")
        print(f"Confidence: {result['score']:.4f}")
        print(f"Inference time: {result['inference_time']:.4f} seconds")

def main():
    # Load data
    test_df = load_data()
    # Initialize model
    model = initialize_model()

    # Select a single random review
    sample = test_df.sample(1).iloc[0]
    text = sample['text']
    real_label = "POSITIVE" if sample['label'] == 1 else "NEGATIVE"

    print("\nSelected review:")
    print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"Real label: {real_label}")

    # Run inference
    result = analyze_sentiment(model, text)
    predicted_label = result['sentiment']
    confidence = result['score']
    print(f"Predicted label: {predicted_label} (confidence: {confidence:.4f})")
    print(f"Inference time: {result['inference_time']:.4f} seconds")
    print(f"{'✓' if predicted_label == real_label else '✗'} Prediction {'correct' if predicted_label == real_label else 'incorrect'}.")

if __name__ == "__main__":
    main()
