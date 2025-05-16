"""
Script to evaluate the performance of a RAG-based sentiment classification approach on the test set.
Instead of using the transformer model directly, this script uses retrieval-augmented generation (RAG)
by finding similar reviews in the vector database and making predictions based on their labels.
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

def get_similar_reviews(collection, embedder, text, n_results=3):
    """Get similar reviews using the vector database"""
    # Encode the query text
    query_embedding = embedder.encode(text, normalize_embeddings=True)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    similar_reviews = []
    for i in range(min(n_results, len(results['documents'][0]))):
        similar_reviews.append({
            'text': results['documents'][0][i],
            'label': results['metadatas'][0][i]['label'],
            'distance': results['distances'][0][i] if 'distances' in results else None
        })
    
    return similar_reviews

def predict_with_rag(collection, embedder, text, n_results=3):
    """
    Predict sentiment using RAG approach by finding similar reviews
    and weighting their labels according to similarity
    """
    start = time.time()
    
    # Get similar reviews
    similar_reviews = get_similar_reviews(collection, embedder, text, n_results)
    
    # Calculate weights based on similarity (1 - distance)
    weights = []
    labels = []
    
    for review in similar_reviews:
        # Convert string label to integer (0 for negative, 1 for positive)
        label = int(review['label'])
        labels.append(label)
        
        # Calculate weight based on similarity (1 - distance)
        if review['distance'] is not None:
            similarity = 1 - review['distance']
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
    
    end = time.time()
    inference_time = end - start
    
    return prediction, inference_time

def evaluate_rag(collection, embedder, df, n_results=3):
    """
    Evaluate RAG approach on the test set.
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
        
        pred, inference_time = predict_with_rag(collection, embedder, text, n_results)
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
    print(f"Evaluating RAG-based sentiment classification with top-{3} similar reviews...")
    start_total = time.time()
    report, avg_time, true_labels, pred_labels = evaluate_rag(collection, embedder, test_df, n_results=3)
    total_time = time.time() - start_total
    
    print("\nClassification Report (only_rag):")
    print(report)
    print(f"Average inference time per review: {avg_time:.4f} seconds")
    print(f"Total time to evaluate all samples: {total_time:.2f} seconds")

    # Save results in results/only_rag
    results_dir = os.path.join("results", "only_rag_three_retrievals")
    os.makedirs(results_dir, exist_ok=True)
    txt_path = os.path.join(results_dir, "classification_report.txt")
    json_path = os.path.join(results_dir, "metrics.json")

    # Save classification report as text
    with open(txt_path, "w") as f:
        f.write("Classification Report (only_rag):\n")
        f.write(report)
        f.write(f"\nAverage inference time per review: {avg_time:.4f} seconds\n")
        f.write(f"Total time to evaluate all samples: {total_time:.2f} seconds\n")

    # Save metrics as JSON (precision, recall, f1-score, support)
    from sklearn.metrics import classification_report as cr_json
    metrics = cr_json(true_labels, pred_labels, target_names=["NEGATIVE", "POSITIVE"], output_dict=True)
    metrics['average_inference_time'] = avg_time
    metrics['total_evaluation_time'] = total_time
    metrics['rag_n_results'] = 3
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
