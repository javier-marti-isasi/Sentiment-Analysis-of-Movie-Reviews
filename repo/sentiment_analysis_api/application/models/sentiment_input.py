from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ReviewRequest(BaseModel):
    """Request model for sentiment analysis of movie reviews"""
    review_text: str

    class Config:
        json_schema_extra = {
            "example": {
                "review_text": "This movie exceeded my expectations..."
            }
        }
    
class SimilarReview(BaseModel):
    """Model representing a similar review retrieved from the vector database"""
    text: str
    label: str
    similarity: float

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""
    sentiment: str  # 'positive' or 'negative'
    confidence: float
    similar_reviews: List[SimilarReview]
