"""
Script to evaluate the performance of a simplified RAG-based sentiment classification approach.
This script makes predictions based ONLY on the single closest review in the vector database,
without any weighted averaging of multiple reviews.
"""
import os
import time
import pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
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

def predict_with_single_retrieval(collection, embedder, text):
    """
    Predict sentiment using only the single closest review from the vector database
    """
    start = time.time()
    
    # Encode the query text
    query_embedding = embedder.encode(text, normalize_embeddings=True)
    
    # Query the collection for a single result
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=1
    )
    
    # Extract the label from the closest match
    if results['documents'] and results['documents'][0]:
        # Convert string label to integer (0 for negative, 1 for positive)
        label = int(results['metadatas'][0][0]['label'])
        prediction = label
    else:
        # Fallback if no results
        prediction = 0
    
    end = time.time()
    inference_time = end - start
    
    return prediction, inference_time

def evaluate_single_retrieval(collection, embedder, df):
    """
    Evaluate the single-retrieval approach on the test set.
    Returns classification_report and average inference time.
    """
    texts = df['text'].tolist()
    true_labels = df['label'].tolist()
    pred_labels = []
    inference_times = []

    print(f"Evaluating on {len(texts)} samples (entire test set)...")
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(texts)} samples evaluated...")
        
        pred, inference_time = predict_with_single_retrieval(collection, embedder, text)
        pred_labels.append(pred)
        inference_times.append(inference_time)

    report = classification_report(true_labels, pred_labels, target_names=["NEGATIVE", "POSITIVE"])
    avg_inference_time = sum(inference_times) / len(inference_times)
    return report, avg_inference_time, true_labels, pred_labels

def main():
    # Load test data
    test_df = load_data()
    
    # Load embedding model
    embedder = load_embedding_model()
    
    # Connect to vector database
    collection = connect_to_vector_db()
    
    # Start evaluation
    print("Evaluating sentiment classification using only the single closest review...")
    start_total = time.time()
    report, avg_time, true_labels, pred_labels = evaluate_single_retrieval(collection, embedder, test_df)
    total_time = time.time() - start_total
    
    print("\nClassification Report (only_rag_one_retrieval):")
    print(report)
    print(f"Average inference time per review: {avg_time:.4f} seconds")
    print(f"Total time to evaluate all samples: {total_time:.2f} seconds")

    # Save results in results/only_rag_one_retrieval
    results_dir = os.path.join("results", "only_rag_one_retrieval")
    os.makedirs(results_dir, exist_ok=True)
    txt_path = os.path.join(results_dir, "classification_report.txt")
    json_path = os.path.join(results_dir, "metrics.json")

    # Save classification report as text
    with open(txt_path, "w") as f:
        f.write("Classification Report (only_rag_one_retrieval):\n")
        f.write(report)
        f.write(f"\nAverage inference time per review: {avg_time:.4f} seconds\n")
        f.write(f"Total time to evaluate all samples: {total_time:.2f} seconds\n")

    # Save metrics as JSON (precision, recall, f1-score, support)
    from sklearn.metrics import classification_report as cr_json
    metrics = cr_json(true_labels, pred_labels, target_names=["NEGATIVE", "POSITIVE"], output_dict=True)
    metrics['average_inference_time'] = avg_time
    metrics['total_evaluation_time'] = total_time
    metrics['method'] = "single_closest_review"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
