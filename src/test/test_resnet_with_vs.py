#  Copyright (c) 2025. Prediction By Invention https://predictionbyinvention.com/
#
#  THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#  PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import unittest
from src.paperwings.vector.vector_space import VectorSpace
from src.paperwings.resonator_network.resonator_network import ResonatorNetwork


class TestResonatorNetworkWithVectorSpace(unittest.TestCase):
    def setUp(self):
        self.vector_dim = 1000
        self.num_vectors = 5

        # Initialize a vector space
        self.vector_space = VectorSpace(self.vector_dim)

        # Create a codebook with random vectors
        self.codebook = {
            f"factor_{i}": np.tile(
                np.random.choice([-1, 1], size=(self.vector_dim, 1)),
                (1, self.num_vectors)
            )
            for i in range(self.num_vectors)
        }

        # Generate a composite vector
        self.composite_vector = np.prod(np.hstack(list(self.codebook.values())), axis=1).flatten()

        # Initialize the resonator network
        self.resonator_network = ResonatorNetwork()

    def test_resonator_network_convergence(self):
        factor_states, iterations, limit_cycle_info = self.resonator_network.run(
            composite_vec=self.composite_vector,
            factor_codebooks=self.codebook,
            lim_cycle_detection_len=10
        )

        # Ensure factor states are returned correctly
        self.assertEqual(len(factor_states), self.num_vectors)

        # Ensure all factor states are vectors of the correct shape
        for factor in factor_states.values():
            self.assertEqual(factor.shape[0], self.vector_dim)

        # Check that the network ran for at least 1 iteration
        self.assertGreater(iterations, 0)

        # Verify limit cycle detection
        self.assertIn("found", limit_cycle_info)


if __name__ == "__main__":
    unittest.main()
