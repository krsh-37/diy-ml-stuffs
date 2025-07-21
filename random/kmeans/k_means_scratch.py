"""
Kmeans clustering implementation 
"""

import seaborn as sns
import numpy as np

class KMeansScratch():
    def __init__(self, k_cluster):
        self.k_cluster = k_cluster
    
    def random_init_cetroids(self, X):
        rand_indices = np.random.choice(len(X), self.k_cluster, replace=False)
        return X[rand_indices]
    
    def euclidean_dist(self, a, b):
        return np.linalg.norm( a - b)

    def assign_clusters(self, X, centroids):
        labels = []
        for pnt in X:
            dists = [self.euclidean_dist(pnt, ctrd) for ctrd in centroids]
            min_cluster_idx = np.argmin(dists)
            labels.append(min_cluster_idx)
        return np.array(labels)

    def recalculate_centroids(self, X, labels):
        new_centroids = []
        for i in range(self.k_cluster):
            cluster_points = X[labels == i]
            new_centroids.append(np.mean(cluster_points, axis=0))
        return np.array(new_centroids)
   
    def check_convergence(self, centroids, new_centroids, threshold):
        return np.allclose(centroids, new_centroids, rtol=threshold)

    def fit(self, X, max_steps=100, threshold=1e-4):
        centroids = self.random_init_cetroids(X)
        for s in range(max_steps):
            labels = self.assign_clusters(X, centroids)
            new_centroids = self.recalculate_centroids(X, labels)
            if self.check_convergence(centroids, new_centroids, threshold):
                break
            centroids = new_centroids
        self.centroids = centroids
        return centroids, labels
    
    def predict(self, X):
        X_cp = X.copy()
        ## handle if there is predict one or array by adding everything to 2D for easier handling.
        if X.ndim == 1:
            X_cp = X_cp[np.newaxis, :]
        result = [self._predict_one(x) for x in X_cp]
        return np.array(result)

    def _predict_one(self, x):
        dists = [self.euclidean_dist(x, ctrd) for ctrd in self.centroids]
        return np.argmin(dists)

if __name__ == "__main__":
    ## Testing code
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    X, _ = make_blobs(n_samples=300, centers=3, n_features=2)

    kmeans = KMeansScratch(k_cluster=3)
    centroids, labels = kmeans.fit(X, 20)

    print("Centroids:\n", centroids)
    print("Cluster assignment for first point:", kmeans.predict(X[0]))
    print("labels shape:", labels.shape)

    sns.scatterplot(x = X[:, 0], y=X[:, 1], hue=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label='Centroids')
    plt.legend()
    plt.title("After KMeans Clustering")
    plt.show()

