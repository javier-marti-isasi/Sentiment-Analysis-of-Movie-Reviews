from datetime import datetime
from application.services.sentiment_service import process_review

# Single hardcoded review for testing
SAMPLE_REVIEW = """This movie exceeded my expectations with its brilliant screenplay and outstanding performances. 
The director's vision was perfectly executed, and the cinematography was breathtaking. 
I would highly recommend it to anyone who appreciates thoughtful storytelling."""

def test_sentiment_analysis():
    """Simple test to process a single review and print results"""
    print(f"Test started at: {datetime.now()}")
    print("\nProcessing sample review...")
    print("-" * 60)
    print(f"Review text: {SAMPLE_REVIEW[:100]}...")
    
    # Process the review directly using the service function
    result = process_review(SAMPLE_REVIEW)
    
    # Print results
    print("\n=== SENTIMENT ANALYSIS RESULTS ===")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    print("\n=== SIMILAR REVIEWS ===")
    for i, review in enumerate(result['similar_reviews'], 1):
        print(f"\nSimilar Review #{i}:")
        print(f"Text: {review['text'][:100]}...")
        print(f"Label: {review['label']}")
        print(f"Similarity: {review['similarity']:.4f}")

if __name__ == "__main__":
    test_sentiment_analysis()
