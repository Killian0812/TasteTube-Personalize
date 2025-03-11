import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymongo

client = pymongo.MongoClient("")
db = client["tastetube"]

# Fetch video metadata from MongoDB
videos = db.videos.find()
video_data = [
    {"video_id": vid["_id"], "text": f"{vid['title']} {vid['description']}"}
    for vid in videos
]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform([v["text"] for v in video_data])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Get recommendations for a video
def recommend_videos(video_id, num_recommendations=5):
    idx = videos[videos["video_id"] == video_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : num_recommendations + 1]  # Exclude self
    video_indices = [i[0] for i in sim_scores]
    return videos["video_id"].iloc[video_indices]


print(recommend_videos("video1"))
