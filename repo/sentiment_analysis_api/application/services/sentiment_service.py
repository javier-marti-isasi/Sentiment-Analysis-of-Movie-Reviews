import os
import pathlib
import time
from typing import List, Dict, Any, Tuple

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import chromadb

# Paths to models and databases
RESOURCES_DIR = pathlib.Path(__file__).parent.parent.parent / "resources"
MODEL_DIR = RESOURCES_DIR / "classification_model" / "distilbert-base-uncased-finetuned-sst-2-english"
EMBEDDING_MODEL_DIR = RESOURCES_DIR / "embedding_model"
VECTOR_DB_DIR = RESOURCES_DIR / "vectorial_database"

# Initialize models and vector DB
sentiment_model = None
embedding_model = None
vector_db = None

def initialize_models():
    """Initialize models and vector database"""
    global sentiment_model, embedding_model, vector_db
    
    # Initialize sentiment model
    print(f"Loading sentiment model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    # Initialize embedding model
    print(f"Loading embedding model from: {EMBEDDING_MODEL_DIR}")
    embedding_model = SentenceTransformer(str(EMBEDDING_MODEL_DIR), device="cpu")
    
    # Connect to vector database
    print(f"Connecting to vector database at: {VECTOR_DB_DIR}")
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    vector_db = client.get_collection("reviews")
    
    print("All models and databases initialized successfully")

def analyze_sentiment(review_text: str) -> Tuple[str, float]:
    """
    Analyze sentiment of a review
    
    Args:
        review_text: The text to analyze
        
    Returns:
        Tuple of (sentiment, confidence)
    """
    # Initialize models if not already initialized
    if sentiment_model is None:
        initialize_models()
    
    # Get sentiment prediction
    start_time = time.time()
    result = sentiment_model(review_text, truncation=True, max_length=512)
    
    # Format the result
    sentiment = result[0]['label']
    confidence = result[0]['score']
    
    # Convert to lowercase for consistent API response
    sentiment = sentiment.lower()
    
    return sentiment, confidence

def get_similar_reviews(review_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Get similar reviews from the vector database
    
    Args:
        review_text: The text to find similar reviews for
        n_results: Number of similar reviews to return
        
    Returns:
        List of similar reviews with text, label, and similarity score
    """
    # Initialize models if not already initialized
    if embedding_model is None or vector_db is None:
        initialize_models()
    
    # Encode the query text
    query_embedding = embedding_model.encode(review_text, normalize_embeddings=True)
    
    # Query the collection
    results = vector_db.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    similar_reviews = []
    for i in range(min(n_results, len(results['documents'][0]))):
        # Get review text
        text = results['documents'][0][i]
        
        # Get label and convert to string format
        label = "positive" if results['metadatas'][0][i]['label'] == "1" else "negative"
        
        # Calculate similarity (1 - distance)
        similarity = 1 - results['distances'][0][i] if 'distances' in results else 0.0
        
        similar_reviews.append({
            'text': text,
            'label': label,
            'similarity': similarity
        })
    
    return similar_reviews

def process_review(review_text: str) -> Dict[str, Any]:
    """
    Process a review for sentiment analysis and find similar reviews
    
    Args:
        review_text: The review text to process
        
    Returns:
        Dictionary with sentiment, confidence, and similar reviews
    """
    # Get sentiment analysis
    sentiment, confidence = analyze_sentiment(review_text)
    
    # Get similar reviews
    similar_reviews = get_similar_reviews(review_text)
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'similar_reviews': similar_reviews
    }
