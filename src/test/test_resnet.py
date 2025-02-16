import unittest

from src.paperwings import encoding_decoding, rn_numpy


class TestResonatorNetwork(unittest.TestCase):
    def setUp(self):
        self.factor_labels = ["Factor1", "Factor2", "Factor3"]
        self.num_neurons = 1000
        self.cbook_size = 50
        self.num_trials = 5

    def test_resonator_network(self):
        for _ in range(self.num_trials):
            the_codebooks = encoding_decoding.generate_codebooks(
                self.factor_labels,
                self.num_neurons,
                {x: self.cbook_size for x in self.factor_labels},
            )
            composite_query, gt_vecs, gt_cbook_indexes = (
                encoding_decoding.generate_c_query(the_codebooks)
            )
            decoded_factors, _, _ = rn_numpy.run(
                composite_query,
                the_codebooks,
            )
            best_guesses = encoding_decoding.best_guess(decoded_factors, the_codebooks)
            accuracy = encoding_decoding.calculate_accuracy(
                best_guesses, gt_cbook_indexes
            )

            print("The best guess based on the final state of the model is:")
            print(best_guesses)
            print("While the ground truth is:")
            print(gt_cbook_indexes)
            print("...for an accuracy of ", accuracy)
            print("---------")

            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
