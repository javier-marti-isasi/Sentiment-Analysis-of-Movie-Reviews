"""
Script to download and save the DistilBERT model for sentiment analysis
"""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    # Define model name and local path
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model_dir = os.path.join("model", "classification_model", model_name)
    
    print(f"Downloading model: {model_name}")
    print(f"This will be saved to: {os.path.abspath(model_dir)}")
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Download and save tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_dir)
    print("Tokenizer saved successfully.")
    
    # Download and save model
    print("Downloading classification model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.save_pretrained(model_dir)
    print("Model saved successfully.")
    
    print(f"\nModel and tokenizer have been downloaded and saved to: {os.path.abspath(model_dir)}")
    print("You can now use this local model in your sentiment analysis scripts.")

if __name__ == "__main__":
    main()
