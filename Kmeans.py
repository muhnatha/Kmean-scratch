import numpy as np

class Kmeans:
    def __init__(self, n_clusters=3, max_iters=100, ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters 
        self.cluster_centers = {}
        self.random = 42
        self.labels = None
        self.cluster_centers =  None
        
    def initialize_centroids(self, X):
        # takes random data point as the start point
        rng = np.random.default_rng(self.random)
        idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[idx]
    
    def assign_clusters(self, X, centroids):
        # calculate using aeuclidean distance
        labels = []
        for x in X:
            min_dis = float('inf')
            label = -1
            for idx, centroid in enumerate(centroids):
                distance = np.linalg.norm(x - centroid, axis=1)
                if distance < min_dis:
                    min_dis = distance
                    label = idx
            
            labels.append(label)
        return np.array(labels)
    
    def update_centroids(self, X, clusters):
        for i in range(self.n_clusters):
            points = np.array(clusters[i]['points'])
                


            if points.shape[0] > 0:
                new_center = points.mean(axis=0)
                clusters[i]['center'] = new_center
                clusters[i]['points'] = []
        
        return clusters
    
    def fit(self, X):
        # intialize the centroids
        self.cluster_centers = self.initialize_centroids(X)
            
        for _ in range(self.max_iters):
            labels = self.assign_clusters(X, self.cluster_centers)
            # create dicrectory
            clusters = {
                i:{
                    'center': self.cluster_centers[i],
                    'points':[]
                } for i in range(self.n_clusters)
            }
            # add points to centroids    
            for x, label in zip(X, labels):
                clusters[label]['points'].append(x)
            
            # update centroids
            clusters = self.update_centroids(X, clusters)

            new_centers = np.array([clusters[i]['center']for i in range(self.n_clusters)])

            # check convergence
            if np.allclose(self.cluster_centers, new_centers):
                break

            self.cluster_centers = new_centers
        
        self.labels = self.assign_clusters(X, self.cluster_centers)
     
    def predict(self, X):
        labels = []

        for x in X:
            label = -1
            min_dist = float('inf')

            for idx, centroid in enumerate(self.cluster_centers):
                distance = np.linalg.norm(x-centroid)
                if distance < min_dist:
                    min_dist = distance
                    label = idx
            

            labels.append(label)
        return np.array(labels)
