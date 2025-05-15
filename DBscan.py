import numpy as np

class DBscan:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.core_sample_indices = []

    def fit(self, X):
        n = len(X)
        self.labels = np.full(n, -1)  # initialize all labels to -1 (noise)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)

        for idx in range(n):
            if visited[idx]:
                continue
            visited[idx] = True
            neighbors = self.region_query(X, idx)

            if len(neighbors) < self.min_samples:
                continue  # label remains -1 (noise)
            else:
                self.expand_cluster(X, idx, neighbors, cluster_id, visited)
                cluster_id += 1

    def expand_cluster(self, X, idx, neighbors, cluster_id, visited):
        self.labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            point_idx = neighbors[i]
            if not visited[point_idx]:
                visited[point_idx] = True
                new_neighbors = self.region_query(X, point_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            if self.labels[point_idx] == -1:
                self.labels[point_idx] = cluster_id
            i += 1

    def region_query(self, X, idx):
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]
