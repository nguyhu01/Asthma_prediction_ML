import numpy as np

from sklearn.neighbors import NearestNeighbors

class DataAugmentor:
    def __init__(self, noise_level=0.1):
        """
        Initialize the FeatureAugmentation class.
        :param noise_level: float, the level of noise to be added to the features.
        """
        self.noise_level = noise_level

    def add_noise(self, X):
        """
        Add random noise to the features.
        :param X: array-like, input features.
        :return: array-like, features with added noise.
        """
        # Generate random noise with the same shape as input features
        noise = np.random.normal(scale=self.noise_level, size=X.shape)
        
        # Add noise to the features
        augmented_X = X + noise
        
        return augmented_X

    def _find_k_nearest_neighbors(self, X, y, k = 3):

        neighbors = NearestNeighbors(n_neighbors = k).fit(X)
        return neighbors.kneighbors(X, return_distance=False)

    def _find_k_nearest_neighbors_by_class(self, X, y,  k = 3):

        neighbor_array = []
        labels = np.unique(y)

        for l in labels:
            Xl = X[y == l]
            neighbor_array += list(self._find_k_nearest_neighbors(Xl, k))

        return neighbor_array

    def interpolate(self, X, y,  lambda_val=0.5, k=3, noise_ratio = None):

        neighbor_list = self._find_k_nearest_neighbors_by_class(X, y, k)
        new_samples_X = []

        for sample in neighbor_list:
            for j in sample:
                for k in sample:
                    if j != k:
                        new_sample_X = (X[k] - X[j]) * lambda_val + X[j]
                        new_samples_X.append(new_sample_X)
                  

        return np.array(new_samples_X), y

    def extrapolate(self, X, y, lambda_val=0.5, k=3, noise_ratio = None):

        neighbor_list = self._find_k_nearest_neighbors_by_class(X, y, k)
        new_samples_X = []

        for sample in neighbor_list:
            for j in sample:
                for k in sample:
                    if j != k:
                        new_sample_X = (X[j] - X[k]) * lambda_val + X[j]
                        new_samples_X.append(new_sample_X)

        return np.array(new_samples_X), y
        
