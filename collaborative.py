from lightfm import LightFM
import pymongo

client = pymongo.MongoClient("")
db = client["tastetube"]
interactions = db.interactions.find()

weighted_data = []
for doc in interactions:
    weight = (
        # 5.0 for first like
        doc["likes"] * 3.0
        + 2.0
        + doc["shares"] * 4.0
        + doc["bookmarked"] * 3.0
        + doc["views"] * 1.0
    )
    weighted_data.append((doc["userId"], doc["videoId"], weight))

dataset = Dataset()
dataset.fit(users=all_users, items=all_videos)
interactions_matrix, _ = dataset.build_interactions(weighted_data)

model = LightFM(loss="warp")
model.fit(interactions_matrix, epochs=30)


# Generate recommendations for a user
def recommend_for_user(user_id, num_recommendations=5):
    user_idx = dataset.mapping()[0][user_id]
    all_video_ids = list(dataset.mapping()[2].values())
    scores = model.predict(user_idx, all_video_ids)
    top_video_indices = scores.argsort()[-num_recommendations:][::-1]
    return [list(dataset.mapping()[2].keys())[i] for i in top_video_indices]


print(recommend_for_user("user1"))
