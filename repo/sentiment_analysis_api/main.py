import sys
import os
import site

# Add site-packages to path to ensure all installed packages are found
site_packages = site.getsitepackages()
for path in site_packages:
    if path not in sys.path:
        sys.path.append(path)

# Import necessary modules
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from application.routers import sentiment_analysis

app = FastAPI(
    title='Sentiment Analysis API',
    version='1.0',
    description='API for movie review sentiment analysis with similar review retrieval'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

app.include_router(sentiment_analysis.router)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
