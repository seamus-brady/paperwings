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

import copy
from typing import Dict, Tuple, Union

import numpy as np

from src.paperwings.exceptions.res_net_exception import ResNetException
from src.paperwings.resonator_network.encoder_decoder import EncoderDecoder
from src.paperwings.resonator_network.limit_cycle_catcher import LimitCycleCatcher
from src.paperwings.util.logging_util import LoggingUtil


class ResonatorNetwork:
    """
    A NumPy implementation of (discrete-time, bipolar) Resonator Networks
    """

    LOGGER = LoggingUtil.instance("<ResonatorNetwork>")

    # One of {'OLS', 'OP'}, specifies whether the synaptic weight matrices
    # are computed using Ordinary Least Squares or the Outer Product rule. Default is 'OP'.
    DEFAULT_SYNAPSE_TYPE = "OP"

    # This model does not guarantee convergence. Optionally, it can be stopped after a specified number of
    # discrete-time iterations. In typical cases, this may be a few dozen to a few hundred iterations, but
    # for larger problems, it may require significantly more. If the model fails to find the correct factorization,
    # increasing the iteration count may help. The default is 10,000.
    DEFAULT_MAX_NUM_ITERS = 10000

    def __init__(
        self,
        synapse_type: str = DEFAULT_SYNAPSE_TYPE,
        max_num_iters: int = DEFAULT_MAX_NUM_ITERS,
    ):
        self.synapse_type = synapse_type
        self.max_num_iters = max_num_iters

    def run(
        self,
        composite_vec: np.ndarray,
        factor_codebooks: Dict[Union[int, str], np.ndarray],
        lim_cycle_detection_len: int = 0,
    ) -> Tuple[
        Dict[Union[int, str], np.ndarray],
        int,
        Dict[str, Union[bool, Dict[Union[int, str], int]]],
    ]:
        """
        Factors composite_vec based on the vectors in factor_codebooks

        Parameters
        ----------
        composite_vec : ndarray(int8, size=(N,))
          The N-dimensional vector that we would like to factor
        factor_codebooks : dictionary
          Keys index the label of the factor (could be 0, 1, 2, ... or 'color',
          'shape' etc.) with the dictionary values being matrices whose columns
          are the codebook vectors for that factor.
        lim_cycle_detection_len: length of the limit cycle to detect
        """
        try:
            self.LOGGER.debug("Running Resonator Network")
            factor_states: Dict[Union[int, str], np.ndarray] = dict.fromkeys(factor_codebooks)  # type: ignore
            codebook_pseudo_inverse: Dict[Union[int, str], np.ndarray] = dict.fromkeys(factor_codebooks)  # type: ignore
            limit_cycle_detectors: Dict[
                Union[int, str], "LimitCycleCatcher"
            ] = dict.fromkeys(
                factor_codebooks
            )  # type: ignore
            factor_ordering = list(factor_codebooks.keys())

            for factor_label in factor_ordering:

                factor_states[factor_label] = self.activation(
                    np.sum(factor_codebooks[factor_label], axis=1).astype(np.float32)
                )

                if self.synapse_type == "OLS":
                    codebook_pseudo_inverse[factor_label] = np.linalg.pinv(
                        factor_codebooks[factor_label]
                    )

                if lim_cycle_detection_len > 1:
                    limit_cycle_detectors[factor_label] = LimitCycleCatcher(
                        len(composite_vec), max_lim_cycle_len=lim_cycle_detection_len
                    )

            iter_idx = 0
            converged = False
            limit_cycle_found = False
            cycle_lengths = {}

            while (
                not converged
                and not limit_cycle_found
                and iter_idx < self.max_num_iters
            ):
                previous_states = copy.deepcopy(factor_states)
                factor_converged = []
                factor_has_limit_cycle = []

                for factor_label in factor_ordering:
                    product_other_factors = np.prod(
                        np.array(
                            [
                                factor_states[x]
                                for x in factor_states
                                if x != factor_label
                            ]
                        ),
                        axis=0,
                    )

                    if self.synapse_type == "OLS":
                        factor_states[factor_label] = self.activation(
                            np.dot(
                                factor_codebooks[factor_label],
                                np.dot(
                                    codebook_pseudo_inverse[factor_label],
                                    composite_vec * product_other_factors,
                                ),
                            )
                        )
                    else:
                        factor_states[factor_label] = self.activation(
                            np.dot(
                                factor_codebooks[factor_label],
                                np.dot(
                                    factor_codebooks[factor_label].T,
                                    composite_vec * product_other_factors,
                                ),
                            )
                        )

                    if lim_cycle_detection_len > 1:
                        limit_cycle_detectors[factor_label].update_buffers(
                            factor_states[factor_label], iter_idx
                        )
                        factor_has_limit_cycle.append(
                            limit_cycle_detectors[factor_label].check_for_limit_cycle()
                        )
                    else:
                        factor_has_limit_cycle.append(False)

                    factor_converged.append(
                        (
                            previous_states[factor_label] == factor_states[factor_label]
                        ).all()
                    )  # noqa

                iter_idx += 1

                if all(factor_converged):
                    converged = True

                if all(factor_has_limit_cycle):
                    limit_cycle_found = True
                    cycle_lengths = {
                        factor_label: limit_cycle_detectors[
                            factor_label
                        ].length_smallest_lim_cycle()
                        for factor_label in factor_ordering
                    }

            for factor_label in factor_ordering:
                factor_states[factor_label] = factor_states[factor_label].astype(
                    np.int8
                )

            for factor_label in factor_ordering:
                cosine_sims = EncoderDecoder.cosine_sim(
                    factor_states[factor_label], factor_codebooks[factor_label]
                )
                winner = np.argmax(np.abs(cosine_sims))
                if cosine_sims[winner] < 0.0:
                    factor_states[factor_label] = factor_states[factor_label] * -1

            if converged:
                self.LOGGER.debug(f"Converged in {iter_idx} iterations")
            elif limit_cycle_found:
                self.LOGGER.debug(f"Limit cycle detected at iteration {iter_idx}")
            else:
                self.LOGGER.debug(
                    f"Forcibly stopped at {self.max_num_iters} iterations"
                )

            lim_cycle_return = {"found": limit_cycle_found}
            if limit_cycle_found:
                lim_cycle_return["lengths"] = cycle_lengths  # type: ignore

            return factor_states, iter_idx, lim_cycle_return  # type: ignore
        except Exception as error:
            self.LOGGER.error(str(error))
            raise ResNetException(str(error))

    def activation(self, membrane_potential: np.ndarray) -> np.ndarray:
        """
        Activation function for the neurons in the network
        """
        try:
            temp = np.sign(membrane_potential)
            temp[temp == 0] = 1
            return temp
        except Exception as error:
            self.LOGGER.error(str(error))
            raise ResNetException(str(error))
