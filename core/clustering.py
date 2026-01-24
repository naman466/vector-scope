import numpy as np
from typing import Tuple, List
from sklearn.cluster import KMeans, DBSCAN

class ClusterAnalyzer:

    def __init__(self, method : str = "KMeans", n_clusters : int = 5):
        self.method = method
        self.n_clusters = n_clusters
        self.clusterer = None
        self.labels_ = None

    def fit(self, embeddings : np.ndarray) -> np.ndarray:
        if self.method == "KMeans":
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state = 51)
            self.labels_ = self.clusterer.fit_predict(embeddings)
        elif self.method == "DBSCAN":
            self.clusterer = DBSCAN(eps = 0.5, min_samples = 5)
            self.labels_ = self.clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")
        
        return self.labels_
    
    def calculate_density(self, embeddings : np.ndarray, labels : np.ndarray) -> dict:
        densities = {}
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue
            cluster_points = embeddings[labels == cluster_id]
            if len(cluster_points) > 1:
                center = cluster.points.mean(axis = 0)
                distances = np.linalg.norm(cluster_points - center, axis = 1)
                densities[cluster_id] = {
                    'mean_distance' : distances.mean(),
                    'std_distance' : distances.std(),
                    'size' : len(cluster_points)
                }
        
        return densities

