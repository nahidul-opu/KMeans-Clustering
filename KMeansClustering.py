import pandas as pd
import numpy as np
class KMeans:
    def __init__(self, n_clusters, n_iteration=1000):
        self.n_clusters = n_clusters
        self.n_iteration = n_iteration

    def random_cluster_centers(self, data):
        centers = data.sample(n=self.n_clusters)
        return centers.to_numpy()

    def assign_cluster(self, data, centers):
        clusters = [[] for _ in range(self.n_clusters)]
        for instance in data:
            clusters[self.find_nearest_cluster(centers, instance)].append(instance)
        return clusters

    def update_centers(self, centers, clusters):
        new_centers = []
        for center, cluster in zip(centers, clusters):
            if len(cluster) == 1:
                continue
            center = np.mean(cluster, axis=0)
            new_centers.append(center)
        return np.array(new_centers)

    def calculate_distance(self, center, instance):
        distance = np.linalg.norm(center - instance)
        return distance

    def find_nearest_cluster(self, centers, instance):
        min_distance = 1000
        cluster_no = 0
        for i, center in enumerate(centers):
            distance = self.calculate_distance(center, instance)
            if distance < min_distance:
                min_distance = distance
                cluster_no = i
        return cluster_no

    def fit(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Requires a Pandas DataFrame")
        self.n_features = data.shape[1]
        centers = self.random_cluster_centers(data)
        data_origin=data.copy()
        data = data.to_numpy()
        for i in range(self.n_iteration):
            clusters = self.assign_cluster(data, centers)
            new_centers = self.update_centers(centers, clusters)
            if (centers == new_centers).all():
                break
            centers = new_centers
        self.n_iteration = i + 1
        self.centers = centers
        self.labels = self.predict(data_origin)
    def predict(self, data):
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        if isinstance(data, list):
            data = pd.DataFrame(data)
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data.reshape(-1, self.n_features))
        elif not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Need a Pandas series or DataFrame or a List or a numpy array"
            )
        clusters = []
        for row in data.iterrows():
            row = row[1]
            cluster_no = self.find_nearest_cluster(self.centers, row)
            clusters.append(cluster_no)
        return np.array(clusters)