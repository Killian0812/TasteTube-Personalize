from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
from scipy.spatial.distance import cosine
from datetime import datetime
import logging
import os

app = FastAPI()
logger = logging.getLogger(__name__)

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "tastetube")
client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]


class VideoResponse(BaseModel):
    id: str
    user_id: str
    title: Optional[str]
    description: Optional[str]
    thumbnail: Optional[str]
    url: str
    hashtags: List[str]
    views: int
    duration: float
    created_at: datetime

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


def get_user_interactions(user_id: str) -> dict:
    """Fetch user interactions for recommendation scoring"""
    interactions = db.interactions.find({"userId": ObjectId(user_id)})
    return {
        str(interaction["videoId"]): {
            "likes": interaction.get("likes", 0),
            "views": interaction.get("views", 0),
            "watchTime": interaction.get("watchTime", 0),
            "bookmarked": interaction.get("bookmarked", False),
        }
        for interaction in interactions
    }


def calculate_content_similarity(
    video_embedding: dict, all_videos: List[dict]
) -> List[dict]:
    """Calculate cosine similarity between video embeddings"""
    if not video_embedding:
        return []

    similarities = []
    for video in all_videos:
        if video.get("embedding") and str(video["_id"]) != str(video_embedding["_id"]):
            similarity = 1 - cosine(video_embedding["embedding"], video["embedding"])
            similarities.append({"video_id": str(video["_id"]), "score": similarity})

    return sorted(similarities, key=lambda x: x["score"], reverse=True)


def get_recommendation_score(
    video: dict, user_interactions: dict, content_similarities: List[dict]
) -> float:
    """Calculate recommendation score based on multiple factors"""
    video_id = str(video["_id"])
    base_score = video.get("views", 0) / (
        video.get("views", 1) + 100
    )  # Normalize views

    # Content-based scoring
    content_score = 0
    for sim in content_similarities:
        if sim["video_id"] == video_id:
            content_score = sim["score"]
            break

    # Interaction-based scoring
    interaction_score = 0
    if video_id in user_interactions:
        interaction = user_interactions[video_id]
        interaction_score = (
            interaction["likes"] * 0.4
            + interaction["views"] * 0.2
            + interaction["watchTime"] * 0.1
            + (1.0 if interaction["bookmarked"] else 0.0) * 0.3
        )

    # Time decay factor
    time_diff = (datetime.now() - video["createdAt"]).days
    time_decay = max(0.1, 1.0 - (time_diff / 30.0))

    # Combine scores (weights can be adjusted)
    return (
        content_score * 0.5 + interaction_score * 0.3 + base_score * 0.2
    ) * time_decay


@app.get("/api/feed", response_model=List[VideoResponse])
async def get_video_feed(user_id: str, limit: int = 20, skip: int = 0):
    """
    Get personalized video feed for a user
    Parameters:
    - user_id: ID of the user
    - limit: Number of videos to return
    - skip: Number of videos to skip (for pagination)
    """
    try:
        # Validate user exists
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get user interactions
        user_interactions = get_user_interactions(user_id)

        # Get all active videos
        videos = list(
            db.videos.find(
                {
                    "status": "ACTIVE",
                    "$or": [
                        {"visibility": "PUBLIC"},
                        {
                            "visibility": "FOLLOWERS_ONLY",
                            "userId": {"$in": user.get("following", [])},
                        },
                        {"userId": ObjectId(user_id)},
                    ],
                }
            )
        )

        # Calculate content-based similarities
        user_liked_videos = db.interactions.find(
            {"userId": ObjectId(user_id), "likes": {"$gt": 0}}
        )
        liked_video_ids = [
            str(interaction["videoId"]) for interaction in user_liked_videos
        ]

        # Get average embedding of liked videos
        liked_videos = list(
            db.videos.find({"_id": {"$in": [ObjectId(vid) for vid in liked_video_ids]}})
        )
        valid_embeddings = [v["embedding"] for v in liked_videos if v.get("embedding")]
        avg_embedding = np.mean(valid_embeddings, axis=0) if valid_embeddings else None

        # Calculate similarities
        content_similarities = calculate_content_similarity(
            (
                {"_id": "", "embedding": avg_embedding}
                if avg_embedding is not None
                else {}
            ),
            videos,
        )

        # Score and rank videos
        scored_videos = []
        for video in videos:
            score = get_recommendation_score(
                video, user_interactions, content_similarities
            )
            scored_videos.append((video, score))

        # Sort by score and apply pagination
        scored_videos.sort(key=lambda x: x[1], reverse=True)
        paginated_videos = scored_videos[skip : skip + limit]

        # Format response
        response = []
        for video, _ in paginated_videos:
            response.append(
                VideoResponse(
                    id=str(video["_id"]),
                    user_id=str(video["userId"]),
                    title=video.get("title"),
                    description=video.get("description"),
                    thumbnail=video.get("thumbnail"),
                    url=video["url"],
                    hashtags=video.get("hashtags", []),
                    views=video.get("views", 0),
                    duration=video.get("duration", 0),
                    created_at=video["createdAt"],
                )
            )

        return response

    except Exception as e:
        logger.error(f"Error generating video feed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
