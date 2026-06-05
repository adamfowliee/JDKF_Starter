import numpy as np


class diffusion:

    def __init__(self, data, epsilon=None, xp=np):
        self.xp = xp
        self.data = xp.asarray(data)

        self.distance_matrix = self._pairwise_squared_distances(self.data)
        self.epsilon = self._resolve_epsilon(epsilon)
        self.W = self._rbf_kernel(self.distance_matrix, self.epsilon)
        self.P = self._row_normalize(self.W)

        # Find eigenvalues and eigenvectors 
        self.eval, self.evec = xp.linalg.eigh(self.P)
        idx = xp.argsort(self.eval)[::-1]
        self.eval = self.eval[idx]
        self.evec = self.evec[:, idx]


    def _pairwise_squared_distances(self, data):
        xp = self.xp
        squared_norms = xp.sum(data * data, axis=1, keepdims=True)
        distances = squared_norms + squared_norms.T - 2.0 * (data @ data.T)
        return xp.maximum(distances, 0.0)

    def _resolve_epsilon(self, epsilon):
        if epsilon is not None:
            return epsilon

        xp = self.xp
        n_samples = self.distance_matrix.shape[0]

        mask = ~xp.eye(n_samples, dtype=bool)
        off_diagonal_distances = self.distance_matrix[mask]

        estimated = xp.median(off_diagonal_distances)
        return estimated

    def _rbf_kernel(self, distances, epsilon):
        return self.xp.exp(-distances / (2.0 * epsilon))

    def _row_normalize(self, matrix):
        row_sums = self.xp.sum(matrix, axis=1, keepdims=True)
        return matrix / row_sums

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_adjacency_matrix(self):
        return self.W

    def get_P_matrix(self):
        return self.P
