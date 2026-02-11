# from sklearn.cluster import KMeans

# def customer_segmentation(data):
#     kmeans = KMeans(n_clusters=4, random_state=42)
#     data["segment"] = kmeans.fit_predict(data)
#     return data

import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("data/processed/customers_cleaned.csv")

X = df.drop("Churn", axis=1)

kmeans = KMeans(n_clusters=4, random_state=42)
df["Segment"] = kmeans.fit_predict(X)

df.to_csv("data/processed/customer_segments.csv", index=False)
