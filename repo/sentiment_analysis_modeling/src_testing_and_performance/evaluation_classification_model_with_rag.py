"""
Script to evaluate a hybrid approach combining the transformer model's prediction
with RAG-based predictions from similar reviews.

This approach uses:
1. The transformer model prediction with its confidence score
2. The 3 most similar reviews from the vector database
3. A weighted combination of both, based on the model's confidence
"""
import os
import time
import pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import json
import chromadb
from sentence_transformers import SentenceTransformer

def load_data():
    """Load test data from parquet file or Hugging Face dataset as fallback."""
    test_path = os.path.join("data", "raw", "test.parquet")
    if os.path.exists(test_path):
        print("Loading test data from parquet file...")
        test_df = pd.read_parquet(test_path)
    else:
        print("Parquet file not found. Loading from IMDB dataset...")
        from datasets import load_dataset
        dataset = load_dataset("imdb")
        test_df = pd.DataFrame(dataset['test'])
    return test_df

def initialize_model():
    """Load the sentiment analysis model and tokenizer from local directory if available."""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model_dir = os.path.join("model", "classification_model", model_name)
    if os.path.exists(model_dir):
        print(f"Loading model from local directory: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    else:
        print("Local model not found. Loading from Hugging Face...")
        sentiment_model = pipeline("sentiment-analysis", model=model_name)
    return sentiment_model

def load_embedding_model():
    """Initialize the embedding model for RAG"""
    embedding_model_dir = str(pathlib.Path(__file__).parent.parent / "model" / "embedding_model")
    print(f"Loading embedding model from: {embedding_model_dir}")
    embedder = SentenceTransformer(embedding_model_dir, device="cpu")
    return embedder

def connect_to_vector_db():
    """Connect to the ChromaDB vector database"""
    chroma_dir = str(pathlib.Path(__file__).parent.parent / "data" / "vectorial_database")
    print(f"Connecting to vector database at: {chroma_dir}")
    
    if not os.path.exists(chroma_dir):
        raise FileNotFoundError(f"Vector database not found at {chroma_dir}. Run create_vector_db.py first.")
    
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection("reviews")
    return collection

def get_classifier_prediction(model, text):
    """Get prediction from the transformer model with confidence"""
    result = model(text, truncation=True, max_length=512)
    # Convert to binary format (0 or 1)
    label = 1 if result[0]['label'].upper().startswith('POS') else 0
    confidence = result[0]['score']
    return label, confidence

def get_rag_prediction(collection, embedder, text, n_results=3):
    """Get RAG-based prediction from similar reviews"""
    # Encode the query text
    query_embedding = embedder.encode(text, normalize_embeddings=True)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    # Calculate weights and labels
    weights = []
    labels = []
    
    for i in range(min(n_results, len(results['documents'][0]))):
        # Get label
        label = int(results['metadatas'][0][i]['label'])
        labels.append(label)
        
        # Calculate weight based on similarity (1 - distance)
        if 'distances' in results:
            similarity = 1 - results['distances'][0][i]
            weights.append(similarity)
        else:
            # If distance is not available, use equal weights
            weights.append(1.0)
    
    # Normalize weights to sum to 1
    if weights:
        weights = np.array(weights) / sum(weights)
        
        # Calculate weighted prediction (weighted average of labels)
        weighted_pred = sum(w * l for w, l in zip(weights, labels))
        
        # Convert to binary prediction (0 or 1)
        prediction = 1 if weighted_pred >= 0.5 else 0
    else:
        # Fallback if no similar reviews found
        prediction = 0
    
    return prediction

def predict_with_hybrid_approach(model, collection, embedder, text, n_results=3, confidence_threshold=0.85):
    """
    Make prediction using a threshold-based hybrid approach:
    - If the classification model's confidence is below the threshold, use the RAG prediction
    - Otherwise, use the classification model's prediction
    
    Args:
        model: The classification model
        collection: The vector database collection
        embedder: The embedding model
        text: The text to classify
        n_results: Number of similar reviews to consider
        confidence_threshold: Threshold below which to use RAG prediction instead of model prediction
    """
    start = time.time()
    
    # Get model prediction and confidence
    model_pred, model_confidence = get_classifier_prediction(model, text)
    
    # Decide whether to use RAG based on model confidence
    if model_confidence < confidence_threshold:
        # Model is not confident enough, use RAG prediction instead
        rag_pred = get_rag_prediction(collection, embedder, text, n_results)
        final_pred = rag_pred
    else:
        # Model is confident enough, use its prediction
        final_pred = model_pred
    
    end = time.time()
    inference_time = end - start
    
    return final_pred, inference_time, model_confidence < confidence_threshold  # Return flag indicating if RAG was used

def evaluate_hybrid_approach(model, collection, embedder, df, n_results=3, confidence_threshold=0.85):
    """
    Evaluate the hybrid approach on the test set.
    Returns classification_report and average inference time.
    
    Args:
        model: The classification model
        collection: The vector database collection
        embedder: The embedding model
        df: DataFrame with test data
        n_results: Number of similar reviews to consider
        confidence_threshold: Threshold below which to use RAG prediction
    """
    texts = df['text'].tolist()
    true_labels = df['label'].tolist()
    pred_labels = []
    inference_times = []
    rag_usage_count = 0

    print(f"Evaluating on {len(texts)} samples (entire test set)...")
    print(f"Using confidence threshold: {confidence_threshold} (below this, RAG is used)")
    
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(texts)} samples evaluated...")
        
        pred, inference_time, used_rag = predict_with_hybrid_approach(
            model, collection, embedder, text, n_results, confidence_threshold
        )
        pred_labels.append(pred)
        inference_times.append(inference_time)
        if used_rag:
            rag_usage_count += 1

    report = classification_report(true_labels, pred_labels, target_names=["NEGATIVE", "POSITIVE"])
    avg_inference_time = sum(inference_times) / len(inference_times)
    rag_usage_percentage = (rag_usage_count / len(texts)) * 100
    
    print(f"\nRAG was used for {rag_usage_count} samples ({rag_usage_percentage:.2f}% of total)")
    
    return report, avg_inference_time, true_labels, pred_labels, rag_usage_count, rag_usage_percentage

def main():
    # Load test data
    test_df = load_data()
    
    # Initialize models
    classifier = initialize_model()
    embedder = load_embedding_model()
    
    # Connect to vector database
    collection = connect_to_vector_db()
    
    # Set confidence threshold parameter
    # Below this threshold, we use RAG prediction instead of model prediction
    # Higher threshold = more RAG usage
    # Lower threshold = more model usage
    CONFIDENCE_THRESHOLD = 0.99  # Hardcoded parameter - adjust as needed
    
    # Start evaluation
    print("Evaluating threshold-based hybrid approach...")
    start_total = time.time()
    report, avg_time, true_labels, pred_labels, rag_count, rag_percentage = evaluate_hybrid_approach(
        classifier, collection, embedder, test_df, n_results=3, confidence_threshold=CONFIDENCE_THRESHOLD
    )
    total_time = time.time() - start_total
    
    print("\nClassification Report (hybrid_model_with_rag):")
    print(report)
    print(f"Average inference time per review: {avg_time:.4f} seconds")
    print(f"Total time to evaluate all samples: {total_time:.2f} seconds")

    # Save results in results/hybrid_model_with_rag
    results_dir = os.path.join("results", "threshold_hybrid_model_with_rag")
    os.makedirs(results_dir, exist_ok=True)
    txt_path = os.path.join(results_dir, "classification_report.txt")
    json_path = os.path.join(results_dir, "metrics.json")

    # Save classification report as text
    with open(txt_path, "w") as f:
        f.write("Classification Report (threshold_hybrid_model_with_rag):\n")
        f.write(report)
        f.write(f"\nAverage inference time per review: {avg_time:.4f} seconds\n")
        f.write(f"Total time to evaluate all samples: {total_time:.2f} seconds\n")
        f.write(f"Confidence threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"RAG was used for {rag_count} samples ({rag_percentage:.2f}% of total)\n")

    # Save metrics as JSON (precision, recall, f1-score, support)
    from sklearn.metrics import classification_report as cr_json
    metrics = cr_json(true_labels, pred_labels, target_names=["NEGATIVE", "POSITIVE"], output_dict=True)
    metrics['average_inference_time'] = avg_time
    metrics['total_evaluation_time'] = total_time
    metrics['method'] = "threshold_hybrid_model_with_rag"
    metrics['rag_n_results'] = 3
    metrics['confidence_threshold'] = CONFIDENCE_THRESHOLD
    metrics['rag_usage_count'] = rag_count
    metrics['rag_usage_percentage'] = rag_percentage
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
