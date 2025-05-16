"""
Script to test the DistilBERT model with RAG on movie reviews
"""
import os
import time
import pathlib
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import chromadb
from sentence_transformers import SentenceTransformer

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

def load_embedding_model():
    """
    Initialize the embedding model for RAG
    """
    embedding_model_dir = str(pathlib.Path(__file__).parent.parent / "model" / "embedding_model")
    print(f"Loading embedding model from: {embedding_model_dir}")
    embedder = SentenceTransformer(embedding_model_dir, device="cpu")
    return embedder

def connect_to_vector_db():
    """
    Connect to the ChromaDB vector database
    """
    chroma_dir = str(pathlib.Path(__file__).parent.parent / "data" / "vectorial_database")
    print(f"Connecting to vector database at: {chroma_dir}")
    
    if not os.path.exists(chroma_dir):
        raise FileNotFoundError(f"Vector database not found at {chroma_dir}. Run create_vector_db.py first.")
    
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection("reviews")
    return collection

def get_similar_reviews(collection, embedder, text, n_results=3):
    """
    Get similar reviews using the vector database
    """
    # Encode the query text
    query_embedding = embedder.encode(text, normalize_embeddings=True)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    similar_reviews = []
    for i in range(n_results):
        try:
            similar_reviews.append({
                'text': results['documents'][0][i],
                'label': results['metadatas'][0][i]['label'],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        except IndexError:
            print(f"Warning: Unable to retrieve result {i}. Collection may have fewer than {n_results} entries.")
            break
    
    return similar_reviews

def analyze_sentiment(model, text):
    """
    Analyze sentiment for a single text and measure inference time
    """
    start_time = time.time()
    result = model(text, truncation=True, max_length=512)
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

def main():
    # Load data
    test_df = load_data()
    
    # Initialize models
    sentiment_model = initialize_model()
    embedding_model = load_embedding_model()
    
    # Connect to vector database
    vector_db = connect_to_vector_db()

    # Select a single random review
    sample = test_df.sample(1).iloc[0]
    text = sample['text']
    real_label = "POSITIVE" if sample['label'] == 1 else "NEGATIVE"

    print("\nSelected review:")
    print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"Real label: {real_label}")

    # Get similar reviews from vector database
    print("\nRetrieving similar reviews from vector database...")
    similar_reviews = get_similar_reviews(vector_db, embedding_model, text, n_results=3)
    
    print("\nTop 3 most similar reviews:")
    for i, review in enumerate(similar_reviews):
        print(f"\n--- Similar Review #{i+1} ---")
        print(f"Text: {review['text'][:150]}{'...' if len(review['text']) > 150 else ''}")
        label = "POSITIVE" if review['label'] == '1' else "NEGATIVE"
        print(f"Label: {label}")
        if review['distance'] is not None:
            print(f"Similarity: {1 - review['distance']:.4f}")

    # Run inference on original review
    print("\nRunning sentiment analysis on the selected review...")
    result = analyze_sentiment(sentiment_model, text)
    predicted_label = result['sentiment']
    confidence = result['score']
    print(f"Predicted label: {predicted_label} (confidence: {confidence:.4f})")
    print(f"Inference time: {result['inference_time']:.4f} seconds")
    print(f"{'✓' if predicted_label == real_label else '✗'} Prediction {'correct' if predicted_label == real_label else 'incorrect'}.")

if __name__ == "__main__":
    main()
