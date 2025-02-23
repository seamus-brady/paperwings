#  Copyright (c) 2025. Prediction By Invention https://predictionbyinvention.com/
#
import unittest
from src.paperwings.vector.vector import BipolarVector, BinarySparseVector
from src.paperwings.vector.vector_space import VectorSpace

import numpy as np
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor


class PrunedBruteForceResonator:
    def __init__(self, vector_space, early_stop=True, top_k=20, use_parallel=True):
        """
        Brute-force search with hierarchical pruning.

        :param vector_space: A VectorSpace object containing known vectors.
        :param early_stop: If True, stops as soon as an exact match is found.
        :param top_k: Limits the number of candidate vectors for pruning.
        :param use_parallel: If True, enables multi-threaded search.
        """
        self.vector_space = vector_space  # Store the vector space reference
        self.early_stop = early_stop
        self.top_k = min(top_k, len(vector_space.vectors))  # Ensure top_k does not exceed available vectors
        self.use_parallel = use_parallel

    def cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    def prefilter_candidates(self, bound_vector):
        """Pre-filter dictionary vectors based on cosine similarity to the bound vector."""
        similarities = [(name, v, self.cosine_similarity(bound_vector.value, v.value)) for name, v in
                        self.vector_space.vectors.items()]
        similarities.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
        return [(name, v) for name, v, _ in similarities[:self.top_k]]  # Take top-K candidates

    def compute_binding(self, v1, v2, v3):
        """Compute the binding of three vectors, supporting both dense and sparse types."""
        if hasattr(v1, "rep") and v1.rep == "binary_sparse":
            candidate_binding = np.bitwise_or(v1.value, v2.value)  # Sparse OR operation
            candidate_binding = np.bitwise_or(candidate_binding, v3.value)
        else:
            candidate_binding = v1.mul(v1.value, v2.value)
            candidate_binding = v3.mul(candidate_binding, v3.value)
        return candidate_binding

    def hamming_distance(self, v1, v2):
        """Compute Hamming distance between two vectors."""
        return np.sum(np.abs(v1 - v2))

    def check_triplet(self, triplet, bound_vector, min_distance, best_match):
        """Compute binding and check distance for a given triplet."""
        (name1, v1), (name2, v2), (name3, v3) = triplet
        candidate_binding = self.compute_binding(v1, v2, v3)
        distance = self.hamming_distance(candidate_binding, bound_vector.value)

        if self.early_stop and distance == 0:
            return (name1, name2, name3), 0  # Exact match

        if distance < min_distance:
            return (name1, name2, name3), distance

        return best_match, min_distance

    def unbind(self, bound_vector):
        """
        Find the closest matching bound triplet in the dictionary using pruning-based brute force.

        :param bound_vector: The bound vector.
        :return: The best (subject, predicate, object) triplet from VectorSpace.
        """
        best_match = None
        min_distance = float('inf')

        # **Step 1: Pre-prune candidates using cosine similarity**
        filtered_candidates = self.prefilter_candidates(bound_vector)
        candidate_count = len(filtered_candidates)

        # **Step 2: Hierarchical pruning strategy**
        if self.use_parallel and candidate_count > 20:
            with ThreadPoolExecutor() as executor:
                futures = []
                for triplet in combinations(filtered_candidates, 3):
                    futures.append(executor.submit(self.check_triplet, triplet, bound_vector, min_distance, best_match))

                for future in futures:
                    candidate, distance = future.result()
                    if distance < min_distance:
                        min_distance = distance
                        best_match = candidate
                        if self.early_stop and min_distance == 0:
                            return best_match
        else:
            for triplet in combinations(filtered_candidates, 3):
                best_match, min_distance = self.check_triplet(triplet, bound_vector, min_distance, best_match)
                if self.early_stop and min_distance == 0:
                    return best_match

        return best_match


class TestOntologyTripleEncoding(unittest.TestCase):
    def setUp(self):
        """Initialize a vector space and encode an ontology triple."""
        self.vector_size = 1000
        self.vector_space = VectorSpace(size=self.vector_size, rep="binary_sparse")

        # add some random vectors
        for i in range(0, 100000):
            self.vector_space.add_vector()

        # Step 1: Define base hypervectors for ontology triple using VectorSpace
        self.vector_space.add_vector(name="Socrates")
        self.vector_space.add_vector(name="is_a")
        self.vector_space.add_vector(name="man")


        # Step 2: Bind them using BinarySparseVector's `mul()` method
        subject = self.vector_space["Socrates"]
        predicate = self.vector_space["is_a"]
        object_ = self.vector_space["man"]

        self.bound_vector = BinarySparseVector(self.vector_size)
        self.bound_vector.value = subject.mul(subject.value, predicate.value)
        self.bound_vector.value = object_.mul(self.bound_vector.value, object_.value)

        # Step 3: Initialize the pruned brute-force resonator
        self.resonator = PrunedBruteForceResonator(self.vector_space, early_stop=True, top_k=10, use_parallel=True)

    def test_ontology_unbinding(self):
        """Test if the pruned brute-force resonator correctly recovers the original ontology triple."""
        recovered_factors = self.resonator.unbind(self.bound_vector)

        # Expected vector names
        expected_factors = {"Socrates", "is_a", "man"}

        # Assert that the recovered factors match the original ones
        self.assertEqual(set(recovered_factors), expected_factors, "Failed to recover ontology triple correctly.")

if __name__ == '__main__':
    unittest.main()

