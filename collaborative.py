from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
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

# Convert the data into a format suitable for surprise
reader = Reader(rating_scale=(0, max(weight for _, _, weight in weighted_data)))
data = Dataset.load_from_df(pd.DataFrame(weighted_data, columns=["userId", "videoId", "weight"]), reader)

trainset, testset = train_test_split(data, test_size=0.25)

model = SVD()
model.fit(trainset)

# Generate recommendations for a user
def recommend_for_user(user_id, num_recommendations=5):
    user_inner_id = trainset.to_inner_uid(user_id)
    all_video_ids = trainset.all_items()
    scores = [(trainset.to_raw_iid(video_id), model.predict(user_id, trainset.to_raw_iid(video_id)).est) for video_id in all_video_ids]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [video_id for video_id, _ in scores[:num_recommendations]]

print(recommend_for_user("user1"))
