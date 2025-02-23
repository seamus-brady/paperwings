import numpy as np
from src.paperwings.resonator_network.resonator_network import ResonatorNetwork
from src.paperwings.resonator_network.encoder_decoder import EncoderDecoder

def generate_and_bind_factors(vector_size: int, factor_labels: list):
    """
    Generates one random bipolar vector per factor and binds them into a composite vector.
    """
    factor_vectors = {factor: 2 * np.random.binomial(1, 0.5, vector_size) - 1 for factor in factor_labels}
    composite_vector = np.prod(list(factor_vectors.values()), axis=0).astype(np.int8)
    return factor_vectors, composite_vector

def create_separate_codebooks(vector_size: int, num_vectors: int, factor_vectors: dict):
    """
    Creates a unique codebook for each factor while ensuring the corresponding factor vector is included.
    """
    factor_codebooks = {}
    for factor, vec in factor_vectors.items():
        random_vectors = (2 * np.random.binomial(1, 0.5, (vector_size, num_vectors - 1)) - 1).astype(np.int8)
        factor_codebooks[factor] = np.hstack((vec.reshape(-1, 1), random_vectors))  # Insert factor vec at index 0
    return factor_codebooks

# 1. Define parameters
vector_size = 100  # Number of neurons
num_vectors = 10   # Total number of vectors per codebook
factor_labels = ["Factor1", "Factor2", "Factor3"]  # Multiple factors

# 2. Generate unique factor vectors and bind them into a composite vector
factor_vectors, composite_vec = generate_and_bind_factors(vector_size, factor_labels)

# 3. Create a unique codebook per factor
factor_codebooks = create_separate_codebooks(vector_size, num_vectors, factor_vectors)

# 4. Initialize Resonator Network
res_net = ResonatorNetwork()

# 5. Run the Resonator Network
decoded_factors, iterations, limit_cycle_info = res_net.run(composite_vec, factor_codebooks)

# 6. Evaluate Results
best_guesses = EncoderDecoder.best_guess(decoded_factors, factor_codebooks)
ground_truth = {factor: 0 for factor in factor_labels}  # Expect index 0 for each factor
accuracy = EncoderDecoder.calculate_accuracy(best_guesses, ground_truth)

# 7. Print Results
print("Decoded Factors:", decoded_factors)
print("Best Guesses:", best_guesses)
print("Accuracy:", accuracy)
print("Iterations:", iterations)
print("Limit Cycle Info:", limit_cycle_info)
