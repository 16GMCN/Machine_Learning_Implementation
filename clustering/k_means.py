import numpy as np 
from scipy.spatial.distance import cdist


class KMeans(object):

	def __init__(self, num_clusters):
		self.num_clusters = num_clusters

	def fit(self, X, max_iteration = 100):
		"""
		K-means algorithm

		Input
		-----------
		X: (num_sample, num_features)

		max_iteration: Int, maximum number of iterations

		Output
		-----------
		centers : (n_clusters, n_features) 
		"""
		I = np.eye(self.num_clusters)
		## initial: randomly get the clusters
		centers = X[np.random.choice(len(X), self.num_clusters, replace = False)]
		for _ in range(max_iteration):
			previous_centers = centers 
			distance = cdist(X, centers)
			cluster_index = I[np.argmin(distance, axis = 1)]
			centers = np.sum(X[:, None, :] * cluster_index[:, :, None], axis = 0) / np.sum(cluster_index, axis = 0)[:, None]
 			if np.allclose(previous_centers, centers):
 				break
 		self.centers = centers

 	def predict(self, X):
 		"""
 		Input
 		-------
 		X: (num_sample, n_features)

 		Output
 		-------
 		index: (num_sample,)
 			each sample with a cluster id
 		"""
 		distance = cdist(X, self.centers)
 		return np.argmin(distance, axis = 1)