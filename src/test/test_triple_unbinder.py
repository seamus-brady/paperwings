#  Copyright (c) 2025. Prediction By Invention https://predictionbyinvention.com/
#
#  THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#  PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import unittest

from src.paperwings.unbinder.triple_unbinder import TripleUnbinder
from src.paperwings.vector.vector import BinarySparseVector
from src.paperwings.vector.vector_space import VectorSpace


class TestTripleUnbinder(unittest.TestCase):
    def setUp(self):
        """Initialize a vector space and encode an ontology triple."""
        self.vector_size = 1000
        self.vector_space = VectorSpace(size=self.vector_size, rep="binary_sparse")

        # add some random vectors
        for i in range(0, 1000):
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
        self.resonator = TripleUnbinder(
            self.vector_space, early_stop=True, top_k=10, use_parallel=True
        )

    def test_ontology_unbinding(self):
        """Test if the pruned brute-force resonator correctly recovers the original ontology triple."""
        recovered_factors = self.resonator.unbind(self.bound_vector)

        # Expected vector names
        expected_factors = {"Socrates", "is_a", "man"}

        # Assert that the recovered factors match the original ones
        self.assertEqual(
            set(recovered_factors),
            expected_factors,
            "Failed to recover ontology triple correctly.",
        )


if __name__ == "__main__":
    unittest.main()
