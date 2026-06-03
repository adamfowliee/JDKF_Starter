import numpy as np


class diffusion:
    """Build an RBF diffusion matrix from a matrix of datapoints.

    The implementation is backend-agnostic: pass ``numpy`` now, or ``cupy``
    later if you want the same code path to run on the GPU.
    """

    def __init__(self, data, epsilon=None, xp=np):
        self.xp = xp
        self.data = xp.asarray(data)

        if self.data.ndim != 2:
            raise ValueError("data must be a 2D matrix of shape (n_samples, n_features)")

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
            if epsilon <= 0:
                raise ValueError("epsilon must be positive")
            return epsilon

        xp = self.xp
        n_samples = self.distance_matrix.shape[0]
        if n_samples < 2:
            raise ValueError("at least two datapoints are required to estimate epsilon")

        mask = ~xp.eye(n_samples, dtype=bool)
        off_diagonal_distances = self.distance_matrix[mask]
        if off_diagonal_distances.size == 0:
            raise ValueError("cannot estimate epsilon from an empty distance set")

        estimated = xp.median(off_diagonal_distances)
        if float(estimated) <= 0:
            raise ValueError("median pairwise distance must be positive to estimate epsilon")
        return estimated

    def _rbf_kernel(self, distances, epsilon):
        return self.xp.exp(-distances / (2.0 * epsilon))

    def _row_normalize(self, matrix):
        row_sums = self.xp.sum(matrix, axis=1, keepdims=True)
        if self.xp.any(row_sums == 0):
            raise ValueError("cannot normalize a matrix with a zero row sum")
        return matrix / row_sums

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_adjacency_matrix(self):
        return self.W

    def get_P_matrix(self):
        return self.P
