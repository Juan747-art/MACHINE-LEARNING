import pandas as pd
from sklearn.cluster import KMeans
import pickle

class KMeansModel:
    def __init__(self):
        self.model = None

    def train(self, data, k=3):
        self.model = KMeans(n_clusters=k, random_state=42)
        self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)

    def get_centroids(self):
        return self.model.cluster_centers_

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)