"""
Script to evaluate the performance of the sentiment classification model on the test set.
Outputs accuracy metrics (precision, recall, F1-score) and inference speed (average time per prediction).
"""
import os
import time
import pandas as pd
from sklearn.metrics import classification_report
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import json
import time

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

def evaluate_model(model, df):
    """
    Evaluate model on the entire test set.
    Returns classification_report and average inference time.
    """
    texts = df['text'].tolist()
    true_labels = df['label'].tolist()
    pred_labels = []
    inference_times = []

    print(f"Evaluating on {len(texts)} samples (entire test set)...")
    for text in texts:
        start = time.time()
        result = model(text, truncation=True, max_length=512)
        end = time.time()
        pred = result[0]['label']
        # Convert model label to 1/0
        pred_labels.append(1 if pred.upper().startswith('POS') else 0)
        inference_times.append(end - start)

    report = classification_report(true_labels, pred_labels, target_names=["NEGATIVE", "POSITIVE"])
    avg_inference_time = sum(inference_times) / len(inference_times)
    return report, avg_inference_time


def main():
    test_df = load_data()
    model = initialize_model()
    start_total = time.time()
    report, avg_time = evaluate_model(model, test_df)
    total_time = time.time() - start_total
    print("\nClassification Report (only_transformer_model):")
    print(report)
    print(f"Average inference time per review: {avg_time:.4f} seconds")
    print(f"Total time to evaluate all samples: {total_time:.2f} seconds")

    # Save results in results/only_transformer_model
    results_dir = os.path.join("results", "only_transformer_model")
    os.makedirs(results_dir, exist_ok=True)
    txt_path = os.path.join(results_dir, "classification_report.txt")
    json_path = os.path.join(results_dir, "metrics.json")

    # Save classification report as text
    with open(txt_path, "w") as f:
        f.write("Classification Report (only_transformer_model):\n")
        f.write(report)
        f.write(f"\nAverage inference time per review: {avg_time:.4f} seconds\n")
        f.write(f"Total time to evaluate all samples: {total_time:.2f} seconds\n")

    # Save metrics as JSON (precision, recall, f1-score, support)
    from sklearn.metrics import classification_report as cr_json
    true_labels = test_df['label'].tolist()
    # Re-run predictions for JSON output on the entire test set
    pred_labels = []
    for text in test_df['text'].tolist():
        result = model(text, truncation=True, max_length=512)
        pred = result[0]['label']
        pred_labels.append(1 if pred.upper().startswith('POS') else 0)
    metrics = cr_json(true_labels, pred_labels, target_names=["NEGATIVE", "POSITIVE"], output_dict=True)
    metrics['average_inference_time'] = avg_time
    metrics['total_evaluation_time'] = total_time
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

