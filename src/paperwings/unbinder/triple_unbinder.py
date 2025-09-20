# MIT License
# Copyright (c) 2025 seamus@corvideon.ie
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from src.paperwings.exceptions.memory_exception import MemoryException
from src.paperwings.util.logging_util import LoggingUtil
from src.paperwings.vector.vector import AbstractVector
from src.paperwings.vector.vector_space import VectorSpace


class TripleUnbinder:
    """
    A class for unbinding a bound vector into a triplet of vectors using a brute-force search with hierarchical pruning.
    """

    LOGGER = LoggingUtil.instance("<TripleUnbinder>")

    def __init__(
        self,
        vector_space: VectorSpace,
        early_stop: bool = True,
        top_k: int = 20,
        use_parallel: bool = True,
    ) -> None:
        """
        Brute-force search with hierarchical pruning.

        :param vector_space: A VectorSpace object containing known vectors.
        :param early_stop: If True, stops as soon as an exact match is found.
        :param top_k: Limits the number of candidate vectors for pruning.
        :param use_parallel: If True, enables multi-threaded search.
        """

        try:
            self.vector_space: VectorSpace = (
                vector_space  # Store the vector space reference
            )
            self.early_stop: bool = early_stop
            self.top_k: int = min(
                top_k, len(vector_space.vectors)
            )  # Ensure top_k does not exceed available vectors
            self.use_parallel: bool = use_parallel
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def prefilter_candidates(
        self, bound_vector: AbstractVector
    ) -> List[Tuple[str, AbstractVector]]:
        """Pre-filter dictionary vectors based on cosine similarity to the bound vector."""
        try:
            similarities: List[Tuple[str, AbstractVector, float]] = [
                (name, v, self.cosine_similarity(bound_vector.value, v.value))
                for name, v in self.vector_space.vectors.items()
            ]
            similarities.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
            return [
                (name, v) for name, v, _ in similarities[: self.top_k]
            ]  # Take top-K candidates
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def compute_binding(
        self, v1: AbstractVector, v2: AbstractVector, v3: AbstractVector
    ) -> np.ndarray:
        """Compute the binding of three vectors, supporting both dense and sparse types."""
        try:
            if hasattr(v1, "rep") and v1.rep == "binary_sparse":
                candidate_binding: np.ndarray = np.bitwise_or(
                    v1.value, v2.value
                )  # Sparse OR operation
                candidate_binding = np.bitwise_or(candidate_binding, v3.value)
            else:
                candidate_binding = v1.mul(v1.value, v2.value)
                candidate_binding = v3.mul(candidate_binding, v3.value)
            return candidate_binding
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def hamming_distance(self, v1: np.ndarray, v2: np.ndarray) -> int:
        """Compute Hamming distance between two vectors."""
        try:
            return int(np.sum(np.abs(v1 - v2)))
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def check_triplet(
        self,
        triplet: Tuple[
            Tuple[str, AbstractVector],
            Tuple[str, AbstractVector],
            Tuple[str, AbstractVector],
        ],
        bound_vector: AbstractVector,
        min_distance: float,
        best_match: Optional[Tuple[str, str, str]],
    ) -> Tuple[Optional[Tuple[str, str, str]], float]:
        """Compute binding and check distance for a given triplet."""
        try:
            (name1, v1), (name2, v2), (name3, v3) = triplet
            candidate_binding = self.compute_binding(v1, v2, v3)
            distance = self.hamming_distance(candidate_binding, bound_vector.value)

            if self.early_stop and distance == 0:
                return (name1, name2, name3), 0  # Exact match

            if distance < min_distance:
                return (name1, name2, name3), distance

            return best_match, min_distance
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def unbind(self, bound_vector: AbstractVector) -> Optional[Tuple[str, str, str]]:
        """
        Find the closest matching bound triplet in the dictionary using pruning-based brute force.

        :param bound_vector: The bound vector.
        :return: The best (subject, predicate, object) triplet from VectorSpace.
        """
        try:
            best_match: Optional[Tuple[str, str, str]] = None
            min_distance: float = float("inf")

            # **Step 1: Pre-prune candidates using cosine similarity**
            filtered_candidates: List[Tuple[str, AbstractVector]] = (
                self.prefilter_candidates(bound_vector)
            )
            candidate_count: int = len(filtered_candidates)

            # **Step 2: Hierarchical pruning strategy**
            if self.use_parallel and candidate_count > 20:
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for triplet in combinations(filtered_candidates, 3):
                        futures.append(
                            executor.submit(
                                self.check_triplet,
                                triplet,
                                bound_vector,
                                min_distance,
                                best_match,
                            )
                        )

                    for future in futures:
                        candidate, distance = future.result()
                        if distance < min_distance:
                            min_distance = distance
                            best_match = candidate
                            if self.early_stop and min_distance == 0:
                                return best_match
            else:
                for triplet in combinations(filtered_candidates, 3):
                    best_match, min_distance = self.check_triplet(
                        triplet, bound_vector, min_distance, best_match
                    )
                    if self.early_stop and min_distance == 0:
                        return best_match

            return best_match
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))
