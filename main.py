from fastapi import FastAPI, HTTPException
from typing import List
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
from scipy.spatial.distance import cosine
from datetime import datetime
import logging
import os
import redis
import json
import time
from dotenv import load_dotenv

from model import VideoResponse

# Load environment variables
load_dotenv()

app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("video_feed.log"),  # Output to file
    ],
)
logger = logging.getLogger(__name__)

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "tastetube")
client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB = os.getenv("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)  # None if no password
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=int(REDIS_PORT),
        password=REDIS_PASSWORD,
        decode_responses=True,
    )
    # Test Redis connection
    redis_client.ping()
except redis.RedisError as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    raise Exception("Redis connection failed")


def get_user_interactions(user_id: str) -> dict:
    """Fetch user interactions with caching"""
    cache_key = f"user_interactions:{user_id}"
    try:
        cached = redis_client.get(cache_key)
    except redis.RedisError as e:
        logger.error(f"Redis error while fetching user interactions: {str(e)}")
        cached = None

    if cached:
        return json.loads(cached)

    interactions = db.interactions.find({"userId": ObjectId(user_id)})
    result = {
        str(interaction["videoId"]): {
            "likes": interaction.get("likes", 0),
            "views": interaction.get("views", 0),
            "watchTime": interaction.get("watchTime", 0),
            "bookmarked": interaction.get("bookmarked", False),
        }
        for interaction in interactions
    }

    # Cache for 5 minutes
    try:
        redis_client.setex(cache_key, 300, json.dumps(result))
    except redis.RedisError as e:
        logger.error(f"Redis error while setting user interactions: {str(e)}")

    return result


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


def get_active_videos(user_id: str, following: List[str]) -> List[dict]:
    """Fetch active videos with populated userId, targetUserId, and products"""
    match_stage = {
        "$match": {
            "status": "ACTIVE",
            "$or": [
                {"visibility": "PUBLIC"},
                {
                    "visibility": "FOLLOWERS_ONLY",
                    "userId": {"$in": [ObjectId(f) for f in following]},
                },
            ],
            "userId": {"$ne": ObjectId(user_id)},
        }
    }

    pipeline = [
        match_stage,
        # Project out fields that are explicitly deleted in Mongoose's toJSON/toObject
        {
            "$project": {"jobId": 0, "muxAssetId": 0}
        },  # Keep embedding for similarity calc for now
        # Lookup userId
        {
            "$lookup": {
                "from": "users",
                "localField": "userId",
                "foreignField": "_id",
                "as": "userId",
                "pipeline": [
                    {"$project": {"_id": 1, "username": 1, "image": 1}},
                ],
            }
        },
        {"$unwind": {"path": "$userId", "preserveNullAndEmptyArrays": True}},
        # Lookup targetUserId
        {
            "$lookup": {
                "from": "users",
                "localField": "targetUserId",
                "foreignField": "_id",
                "as": "targetUserId",
                "pipeline": [
                    {"$project": {"_id": 1, "username": 1, "image": 1}},
                ],
            }
        },
        {"$unwind": {"path": "$targetUserId", "preserveNullAndEmptyArrays": True}},
        # Lookup products with nested lookups
        {
            "$lookup": {
                "from": "products",
                "localField": "products",
                "foreignField": "_id",
                "as": "products",
                "pipeline": [
                    {
                        "$lookup": {
                            "from": "categories",
                            "localField": "category",
                            "foreignField": "_id",
                            "as": "category",
                            "pipeline": [
                                {"$project": {"_id": 1, "name": 1}},
                            ],
                        }
                    },
                    {
                        "$unwind": {
                            "path": "$category",
                            "preserveNullAndEmptyArrays": True,
                        }
                    },
                    {
                        "$lookup": {
                            "from": "users",
                            "localField": "userId",  # This is the product owner
                            "foreignField": "_id",
                            "as": "userId",
                            "pipeline": [
                                {
                                    "$project": {
                                        "_id": 1,
                                        "image": 1,
                                        "username": 1,
                                        "phone": 1,
                                    }
                                },
                            ],
                        }
                    },
                    {
                        "$unwind": {
                            "path": "$userId",  # Unwind the product owner
                            "preserveNullAndEmptyArrays": True,
                        }
                    },
                ],
            }
        },
    ]

    return list(db.videos.aggregate(pipeline))


@app.get("/")
async def root():
    return {"message": "TasteTube Personalize API is running"}


@app.get("/api/feed", response_model=List[VideoResponse])
async def get_video_feed(user_id: str, limit: int = 20, skip: int = 0):
    """
    Get personalized video feed for a user
    Parameters:
    - user_id: ID of the user
    - limit: Number of videos to return
    - skip: Number of videos to skip (for pagination)
    """
    start_time = time.perf_counter()
    try:
        logger.info("=== Start generating video feed ===")

        # Step 1: Validate user
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        logger.info(
            f"User validation completed in {time.perf_counter() - start_time:.4f} seconds"
        )

        # Step 2: Get user interactions
        t1 = time.perf_counter()
        user_interactions = get_user_interactions(user_id)
        logger.info(
            f"Fetched user interactions in {time.perf_counter() - t1:.4f} seconds"
        )

        # Step 3: Get active videos
        t2 = time.perf_counter()
        videos = get_active_videos(user_id, user.get("following", []))
        logger.info(f"Fetched active videos in {time.perf_counter() - t2:.4f} seconds")

        # Step 4: Fetch liked videos and calculate average embedding
        t3 = time.perf_counter()
        user_liked_videos = db.interactions.find(
            {"userId": ObjectId(user_id), "likes": {"$gt": 0}}
        )
        liked_video_ids = [
            str(interaction["videoId"]) for interaction in user_liked_videos
        ]
        liked_videos = list(
            db.videos.find({"_id": {"$in": [ObjectId(vid) for vid in liked_video_ids]}})
        )
        valid_embeddings = [v["embedding"] for v in liked_videos if v.get("embedding")]
        avg_embedding = np.mean(valid_embeddings, axis=0) if valid_embeddings else None
        logger.info(
            f"Calculated average embedding in {time.perf_counter() - t3:.4f} seconds"
        )

        # Step 5: Content similarity calculation
        t4 = time.perf_counter()
        content_similarities = calculate_content_similarity(
            (
                {"_id": "", "embedding": avg_embedding}
                if avg_embedding is not None
                else {}
            ),
            videos,
        )
        logger.info(
            f"Calculated content similarities in {time.perf_counter() - t4:.4f} seconds"
        )

        # Step 6: Scoring videos
        t5 = time.perf_counter()
        scored_videos = [
            (
                video,
                get_recommendation_score(
                    video, user_interactions, content_similarities
                ),
            )
            for video in videos
        ]
        scored_videos.sort(key=lambda x: x[1], reverse=True)
        paginated_videos = scored_videos[skip : skip + limit]
        logger.info(
            f"Scored and paginated videos in {time.perf_counter() - t5:.4f} seconds"
        )

        # Step 7: Formatting response using VideoResponse model
        t6 = time.perf_counter()
        response = []
        for video, _ in paginated_videos:
            try:
                # The MongoDB aggregation pipeline already populates these fields
                # We just need to ensure the dict keys match Pydantic's field names/aliases
                # and that nested objects are passed as dictionaries which Pydantic will validate.
                video_data = {
                    "id": str(video["_id"]),
                    "userId": video["userId"],  # Already populated by lookup
                    "targetUserId": video.get(
                        "targetUserId"
                    ),  # Already populated by lookup, can be None
                    "url": video["url"],
                    "filename": video["filename"],
                    "direction": video.get("direction"),
                    "title": video.get("title"),
                    "description": video.get("description"),
                    "thumbnail": video.get("thumbnail"),
                    "hashtags": video.get("hashtags", []),
                    "products": video.get(
                        "products", []
                    ),  # Already populated by lookup
                    "visibility": video["visibility"],
                    "views": video.get("views", 0),
                    "manifestUrl": video.get("manifestUrl"),
                    "status": video["status"],
                    "duration": video.get("duration", 0),
                    "createdAt": video["createdAt"],
                    "updatedAt": video["updatedAt"],
                }
                response.append(VideoResponse(**video_data))
            except Exception as e:
                logger.error(
                    f"Error validating video data with Pydantic: {str(e)} for video ID: {video.get('_id')}"
                )
                # Log the video data that caused the error for debugging
                logger.debug(f"Problematic video data: {video}")
                continue  # Skip the problematic video and continue processing others

        logger.info(f"Formatted response in {time.perf_counter() - t6:.4f} seconds")

        logger.info(
            f"=== Total feed generation time: {time.perf_counter() - start_time:.4f} seconds ==="
        )
        return response

    except Exception as e:
        logger.error(f"Error generating video feed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
