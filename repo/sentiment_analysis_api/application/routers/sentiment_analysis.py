from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from application.models.sentiment_input import ReviewRequest, SentimentResponse
from application.services.sentiment_service import process_review

router = APIRouter(
    prefix="/sentiment",
    tags=["sentiment"]
)

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_review(request: ReviewRequest):
    """
    Analyze a movie review for sentiment and retrieve similar reviews.
    
    Returns sentiment prediction (positive/negative), confidence score,
    and a list of similar reviews from the database.
    """
    try:
        result = process_review(request.review_text)
        return SentimentResponse(
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            similar_reviews=result['similar_reviews']
        )
    
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing review: {str(e)}")
