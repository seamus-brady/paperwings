# MIT License
#
# This code is based on https://github.com/spencerkent/resonator-networks/ by Spencer Kent.
# Modifications have been made by Seamus Brady in 2025.

# Copyright (c) 2020 Spencer Kent
# Copyright (c) 2025 Seamus Brady

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

import numpy as np

from src.paperwings.exceptions.res_net_exception import ResNetException
from src.paperwings.util.logging_util import LoggingUtil


class EncoderDecoder:
    """
    Defines some useful utilities for generating and decoding bipolar vectors.

    These can be generalized to {0, 1}^N vectors, real vectors, and complex
    vectors, but for brevity I just include the case for bipolar vectors
    """

    LOGGER = LoggingUtil.instance("<EncoderDecoder>")

    @staticmethod
    def cosine_sim(vecs1, vecs2):
        """
        Computes the cosine similarity between two sets of vectors

        Parameters
        ----------
        vecs1 : ndarray
            In the general case, a N x D1 matrix containing D1 vectors,
            one each in its columns. We can also handle individual vectors in
            which case the output will only have one element in the first dimension.
        vecs2 : ndarray
            In the general case, a N x D2 matrix containing D2 vectors,
            one each in its columns. We can also handle individual vectors in
            which case the output will only have one element in the second dimension.

        Returns
        -------
        sim_matrix : ndarray
            The cosine similarity between every pair of vectors, one drawn from vecs1
            and the other drawn from vecs2. The dimension is D1 x D2 where D1 is the
            number of vectors in vecs1 and D2 is the number of vectors in vecs2 and
            entry (i, j) gives the cosine similarity between the i'th vector in
            vecs1 and the j'th vector in vecs2. We'll squeeze down the dimensions
            if either of the inputs is 1d
        """

        try:
            if vecs1.ndim == 1:
                vecs1_l2 = np.linalg.norm(vecs1, 2)
            else:
                vecs1_l2 = np.linalg.norm(vecs1, 2, axis=0)[:, None]
            if vecs2.ndim == 1:
                vecs2_l2 = np.linalg.norm(vecs2, 2)
            else:
                vecs2_l2 = np.linalg.norm(vecs2, 2, axis=0)[None, :]
            normalizing_values = vecs1_l2 * vecs2_l2
            if type(normalizing_values) is np.ndarray:
                normalizing_values[normalizing_values == 0] = 1  # avoid divide by zero
            else:
                if normalizing_values == 0:
                    normalizing_values = 1
            return np.dot(1.0 * vecs1.T, vecs2) / np.squeeze(normalizing_values)
        except Exception as error:
            EncoderDecoder.LOGGER.error(str(error))
            raise ResNetException(str(error))

    @staticmethod
    def generate_codebooks(
        factor_labels, v_size, cbook_sizes, encoding="random", force_unique=False
    ):
        """
        Returns a codebook one can use in experiments

        Parameters
        ----------
        factor_labels : list
            A label for each factor
        v_size : int
            The size of the vectors, N.
        cbook_sizes : dict
            The number of vectors in the codebook for each factor.
        encoding : str, optional
            Currently just {'random'}, but other encodings possible.
        force_unique : bool, optional
            If True, make sure each codevector is unique. Default False.

        Returns
        -------
        factor_codebooks : dict
            Keys are factor labels, values are the codebook matrices
        """
        try:
            if encoding != "random":
                raise KeyError("Unrecognized encoding type " + encoding)
            if force_unique and any([cbook_sizes[x] > 2**v_size for x in cbook_sizes]):
                raise ValueError(
                    "You have asked for distinct codevectors, but there "
                    + "are not enough for this dimensionality."
                )
            factor_codebooks = {}
            for factor_label in factor_labels:
                factor_codebooks[factor_label] = (
                    2 * np.random.binomial(1, 0.5, (v_size, cbook_sizes[factor_label]))
                    - 1
                ).astype("int8")
                if force_unique:
                    duplicates_flag = True
                    while duplicates_flag:
                        no_duplicates = np.unique(
                            factor_codebooks[factor_label], axis=1
                        )
                        if no_duplicates.shape[1] == cbook_sizes[factor_label]:
                            # we're done, move onto the next codebook
                            duplicates_flag = False
                        else:
                            additional_needed = (
                                cbook_sizes[factor_label] - no_duplicates.shape[1]
                            )
                            factor_codebooks[factor_label] = np.hstack(
                                [
                                    no_duplicates,
                                    (
                                        2
                                        * np.random.binomial(
                                            1, 0.5, (v_size, additional_needed)
                                        )
                                        - 1
                                    ).astype("int8"),
                                ]
                            )

            return factor_codebooks
        except Exception as error:
            EncoderDecoder.LOGGER.error(str(error))
            raise ResNetException(str(error))

    @staticmethod
    def generate_c_query(
        factor_codebooks, codebook_inds=None, perturbation_params=None
    ):
        """
        Draw a composite query vector, directly from codebook_inds or randomly

        Parameters
        ----------
        factor_codebooks : dict
            The dictionary containing the codebooks for each factor. Keys label the
            factor and each value is a N x D matrix containing one N-dimensional
            vector in each of its D columns.
        codebook_inds : dict, optional
            Used to pick a specific set of factor vectors, keys are factor labels,
            values are ints specifying an index into the codebook. Default None.
        perturbation_params : dict, optional
            'type' : the type of perturbation. Currently just 'bitflip'
            type-specific additional keys:
              'bitflip': 'ratio'. The ratio of total bits to randomly pick and flip

        Returns
        -------
        composite_vec : ndarray(int8, size=(N,))
            The N x 1 composite query vector
        factor_vecs : dictionary
            The actual factor vectors that were used (before perturbation) to
            construct composite_vec.
        codebook_inds : dictionary
            The indices into each codebook for each of the selected factor vectors.
        """
        try:
            if perturbation_params is not None:
                if perturbation_params["type"] not in ["bitflip"]:
                    raise KeyError("Unrecognized perturbation type")

            if codebook_inds is None:
                codebook_inds = {}
                for factor_label in factor_codebooks:
                    codebook_inds[factor_label] = np.random.randint(
                        0, factor_codebooks[factor_label].shape[1]
                    )

            factor_vecs = {}
            for label in codebook_inds:
                # assert factor_codebooks[label].dtype == "int8"
                factor_vecs[label] = factor_codebooks[label][:, codebook_inds[label]]

            composite_vec = np.prod(
                np.array([factor_vecs[x] for x in factor_vecs]), axis=0
            ).astype("int8")

            if perturbation_params is not None:
                if perturbation_params["type"] == "bitflip":
                    corrupted_indeces = np.random.choice(
                        np.arange(len(composite_vec)),
                        int(perturbation_params["ratio"] * len(composite_vec)),
                        replace=False,
                    )
                    composite_vec[corrupted_indeces] = (
                        composite_vec[corrupted_indeces] * -1
                    )

            return composite_vec, factor_vecs, codebook_inds
        except Exception as error:
            EncoderDecoder.LOGGER.error(str(error))
            raise ResNetException(str(error))

    @staticmethod
    def best_guess(factor_states, factor_codebooks, sim_type="cosine"):
        """
        Given the factor states and the codebooks, return the best guess for the codebook index of each factor.
        """
        try:
            best_guess = {}
            for factor_label in factor_states:
                if sim_type == "cosine":
                    best_guess[factor_label] = np.argmax(
                        EncoderDecoder.cosine_sim(
                            factor_states[factor_label], factor_codebooks[factor_label]
                        )
                    )
                else:
                    raise KeyError("Unrecognized similarity type ", sim_type)
            return best_guess
        except Exception as error:
            EncoderDecoder.LOGGER.error(str(error))
            raise ResNetException(str(error))

    @staticmethod
    def sim_to_target(factor_states, targets, sim_type="cosine"):
        """
        Calculate the similarity between the factor states and the target vectors
        """
        try:
            similarities = {}
            for factor_label in factor_states:
                if sim_type == "cosine":
                    similarities[factor_label] = EncoderDecoder.cosine_sim(
                        factor_states[factor_label], targets[factor_label]
                    )
                else:
                    raise KeyError("Unrecognized similarity type ", sim_type)
            return similarities
        except Exception as error:
            EncoderDecoder.LOGGER.error(str(error))
            raise ResNetException(str(error))

    @staticmethod
    def calculate_accuracy(guesses, ground_truth):
        """
        Calculate the accuracy of the best guesses. Accuracy is proportion of best guesses which are correct.
        """
        try:
            match_accum = 0
            for factor_label in guesses:
                if guesses[factor_label] == ground_truth[factor_label]:
                    match_accum += 1
            return match_accum / len(guesses)
        except Exception as error:
            EncoderDecoder.LOGGER.error(str(error))
            raise ResNetException(str(error))
