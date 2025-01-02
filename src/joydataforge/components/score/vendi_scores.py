"""
This module defines the "VendiScore" class.

The file includes the following import statements:
import numpy as np
from vendi_score import vendi
from scipy.spatial.distance import cdist

The file also includes the following classes and functions:
- VendiScore class with methods: __init__, compute_similarity_matrix, compute_similarity_matrix_fast, score

To use this module, you can instantiate the VendiScore class with embeddings as a parameter, and call the 'score' method to compute the Vendi score for the embeddings.

"""
import numpy as np
from vendi_score import vendi
from scipy.spatial.distance import cdist


class VendiScore(object):
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    async def compute_similarity_matrix(self, embeddings: np.ndarray = None):
        if embeddings is None:
            embeddings = self.embeddings
        n_samples = len(embeddings)  # Number of samples
        similarity_matrix = np.zeros((n_samples, n_samples))

        # Compute the similarity matrix
        for i in range(n_samples):
            for j in range(i, n_samples):  # Only need to compute the upper triangular part of the matrix
                if i == j:
                    # The similarity of an element to itself is set to the maximum value of 1
                    similarity_matrix[i, j] = 1.0
                else:
                    # Compute the similarity between sample i and j
                    similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # The matrix is symmetric

        return similarity_matrix

    async def compute_similarity_matrix_fast(self, embeddings: np.ndarray = None):
        if embeddings is None or not embeddings.any():
            embeddings = self.embeddings
        # Compute the similarity matrix
        similarity_matrix = 1 - cdist(embeddings, embeddings, metric='cosine')

        # Set the diagonal elements to 1
        np.fill_diagonal(similarity_matrix, 1.0)

        return similarity_matrix

    async def score(self, embeddings: np.ndarray = None):
        if embeddings is not None and embeddings.any():
            self.embeddings = embeddings
        return vendi.score_K(await self.compute_similarity_matrix_fast(self.embeddings))
