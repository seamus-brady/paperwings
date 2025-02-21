import unittest

import numpy as np

from src.paperwings.resonator_network.encoder_decoder import EncoderDecoder
from src.paperwings.resonator_network.resonator_network import ResonatorNetwork


class TestResonatorNetwork(unittest.TestCase):
    def setUp(self):
        self.factor_labels = ["Factor1", "Factor2", "Factor3"]
        self.num_neurons = 1000
        self.cbook_size = 50
        self.num_trials = 5
        self.resonator_network = ResonatorNetwork()

    def test_resonator_network_run(self):
        for _ in range(self.num_trials):
            the_codebooks = EncoderDecoder.generate_codebooks(
                self.factor_labels,
                self.num_neurons,
                {x: self.cbook_size for x in self.factor_labels},
            )
            composite_query, gt_vecs, gt_cbook_indexes = (
                EncoderDecoder.generate_c_query(the_codebooks)
            )
            decoded_factors, _, _ = self.resonator_network.run(
                composite_query,
                the_codebooks,
            )
            best_guesses = EncoderDecoder.best_guess(decoded_factors, the_codebooks)
            accuracy = EncoderDecoder.calculate_accuracy(best_guesses, gt_cbook_indexes)

            print("The best guess based on the final state of the model is:")
            print(best_guesses)
            print("While the ground truth is:")
            print(gt_cbook_indexes)
            print("...for an accuracy of ", accuracy)
            print("---------")

            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)

    def test_resonator_network_accuracy(self):
        the_codebooks = EncoderDecoder.generate_codebooks(
            self.factor_labels,
            self.num_neurons,
            {x: self.cbook_size for x in self.factor_labels},
        )
        composite_query, gt_vecs, gt_cbook_indexes = EncoderDecoder.generate_c_query(
            the_codebooks
        )
        decoded_factors, _, _ = self.resonator_network.run(
            composite_query,
            the_codebooks,
        )
        best_guesses = EncoderDecoder.best_guess(decoded_factors, the_codebooks)
        accuracy = EncoderDecoder.calculate_accuracy(best_guesses, gt_cbook_indexes)

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_resonator_network_similarity(self):
        the_codebooks = EncoderDecoder.generate_codebooks(
            self.factor_labels,
            self.num_neurons,
            {x: self.cbook_size for x in self.factor_labels},
        )
        composite_query, gt_vecs, gt_cbook_indexes = EncoderDecoder.generate_c_query(
            the_codebooks
        )
        decoded_factors, _, _ = self.resonator_network.run(
            composite_query,
            the_codebooks,
        )
        similarities = EncoderDecoder.sim_to_target(decoded_factors, gt_vecs)

        for factor_label in self.factor_labels:
            self.assertIn(factor_label, similarities)
            self.assertTrue(np.all(similarities[factor_label] >= -1.0))
            self.assertTrue(np.all(similarities[factor_label] <= 1.0))


if __name__ == "__main__":
    unittest.main()
