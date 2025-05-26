from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymongo
from dotenv import load_dotenv
import os
import uuid

# Initialize FastAPI app
app = FastAPI(title="Video Recommendation API")

# Load environment variables
load_dotenv()

# MongoDB connection
client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client["tastetube"]

# Pydantic model for response
class Video(BaseModel):
    video_id: str
    text: str

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Video]

# Function to fetch video data
def get_video_data():
    videos = db.videos.find()
    return [
        {
            "video_id": str(vid["_id"]),
            "text": f"{vid['title']} {vid['description']} {' '.join(vid['hashtags'])}",
        }
        for vid in videos
    ]

# Function to compute TF-IDF and cosine similarity
def compute_cosine_similarity(video_data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform([v["text"] for v in video_data])
    return cosine_similarity(tfidf_matrix, tfidf_matrix), video_data

# Recommendation function
def recommend_videos(video_id: str, video_data: List[dict], cosine_sim: List[List[float]], num_recommendations: int = 5):
    idx = next((i for i, video in enumerate(video_data) if video["video_id"] == video_id), None)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]  # Exclude self
    video_indices = [i[0] for i in sim_scores]
    return [video_data[i] for i in video_indices]

# API endpoint to get recommendations
@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: str, num_recommendations: int = 5):
    try:
        # Fetch user data (e.g., liked or viewed videos)
        user = db.users.find_one({"user_id": user_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch video data
        video_data = get_video_data()
        if not video_data:
            raise HTTPException(status_code=404, detail="No videos available")

        # Compute cosine similarity
        cosine_sim, video_data = compute_cosine_similarity(video_data)

        # Get user's viewed or liked videos (assuming stored in user document)
        user_videos = user.get("viewed_videos", []) or user.get("liked_videos", [])
        if not user_videos:
            raise HTTPException(status_code=400, detail="User has no viewed or liked videos")

        # Generate recommendations based on the first viewed/liked video
        # For simplicity, using the first video; you can extend to aggregate multiple
        recommendations = recommend_videos(user_videos[0], video_data, cosine_sim, num_recommendations)

        return {
            "user_id": user_id,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app (for local testing, use: uvicorn main:app --reload)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)