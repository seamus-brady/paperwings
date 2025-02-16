import copy
from typing import Dict, Tuple, Union

import numpy as np

from src.paperwings.encoding_decoding import cosine_sim

"""
synapse_type : 
One of {'OLS', 'OP'}, specifies whether the synaptic weight matrices
are computed from the {O}rdinary {L}east {S}quares or the {O}uter
{P}roduct rule. Default 'OP'.
"""
DEFAULT_SYNAPSE_TYPE = "OP"

"""
max_num_iters : int, optional
This model is NOT GUARANTEED TO CONVERGE. We can optionally stop it after
this many discrete-time iterations. In the normal operating regime, this
may be just a few tens or hundreds of iterations, but depending on how
large the problem is, can be much larger. If the model is not finding the
correct factorization, try giving it more iterations. Default 10000.
"""
DEFAULT_MAX_NUM_ITERS = 10000


def run(
    composite_vec: np.ndarray,
    factor_codebooks: Dict[Union[int, str], np.ndarray],
    lim_cycle_detection_len: int = 0,
) -> Tuple[
    Dict[Union[int, str], np.ndarray],
    int,
    Dict[str, Union[bool, Dict[Union[int, str], int]]],
]:
    # assert composite_vec.dtype == np.int8
    synapse_type = DEFAULT_SYNAPSE_TYPE

    factor_states: Dict[Union[int, str], np.ndarray] = dict.fromkeys(factor_codebooks)  # type: ignore
    codebook_pseudoinverse: Dict[Union[int, str], np.ndarray] = dict.fromkeys(factor_codebooks)  # type: ignore
    limit_cycle_detectors: Dict[Union[int, str], "LimitCycleCatcher"] = dict.fromkeys(factor_codebooks)  # type: ignore
    factor_ordering = list(factor_codebooks.keys())

    for factor_label in factor_ordering:
        # assert factor_codebooks[factor_label].dtype == np.int8

        factor_states[factor_label] = activation(
            np.sum(factor_codebooks[factor_label], axis=1).astype(np.float32)
        )

        if synapse_type == "OLS":
            codebook_pseudoinverse[factor_label] = np.linalg.pinv(
                factor_codebooks[factor_label]
            )

        if lim_cycle_detection_len > 1:
            limit_cycle_detectors[factor_label] = LimitCycleCatcher(
                len(composite_vec), max_lim_cycle_len=lim_cycle_detection_len
            )

    iter_idx = 0
    converged = False
    limit_cycle_found = False

    while not converged and not limit_cycle_found and iter_idx < DEFAULT_MAX_NUM_ITERS:
        previous_states = copy.deepcopy(factor_states)
        factor_converged = []
        factor_has_limit_cycle = []

        for factor_label in factor_ordering:
            product_other_factors = np.prod(
                np.array(
                    [factor_states[x] for x in factor_states if x != factor_label]
                ),
                axis=0,
            )

            if synapse_type == "OLS":
                factor_states[factor_label] = activation(
                    np.dot(
                        factor_codebooks[factor_label],
                        np.dot(
                            codebook_pseudoinverse[factor_label],
                            composite_vec * product_other_factors,
                        ),
                    )
                )
            else:
                factor_states[factor_label] = activation(
                    np.dot(
                        factor_codebooks[factor_label],
                        np.dot(
                            factor_codebooks[factor_label].T,
                            composite_vec * product_other_factors,
                        ),
                    )
                )

            if lim_cycle_detection_len > 1:
                limit_cycle_detectors[factor_label].UpdateBuffers(
                    factor_states[factor_label], iter_idx
                )
                factor_has_limit_cycle.append(
                    limit_cycle_detectors[factor_label].CheckForLimitCycle()
                )
            else:
                factor_has_limit_cycle.append(False)

            factor_converged.append(
                (previous_states[factor_label] == factor_states[factor_label]).all()
            )

        iter_idx += 1

        if all(factor_converged):
            # assert not all(factor_has_limit_cycle)
            converged = True

        if all(factor_has_limit_cycle):
            # assert not all(factor_converged)
            limit_cycle_found = True
            cycle_lengths = {
                factor_label: limit_cycle_detectors[
                    factor_label
                ].LengthSmallestLimCycle()
                for factor_label in factor_ordering
            }

    for factor_label in factor_ordering:
        factor_states[factor_label] = factor_states[factor_label].astype(np.int8)

    for factor_label in factor_ordering:
        cosine_sims = cosine_sim(
            factor_states[factor_label], factor_codebooks[factor_label]
        )
        winner = np.argmax(np.abs(cosine_sims))
        if cosine_sims[winner] < 0.0:
            factor_states[factor_label] = factor_states[factor_label] * -1

    if converged:
        print("Converged in", iter_idx, "iterations")
    elif limit_cycle_found:
        print("Limit cycle detected at iteration", iter_idx)
    else:
        print("Forcibly stopped at", DEFAULT_MAX_NUM_ITERS, "iterations")

    lim_cycle_return = {"found": limit_cycle_found}
    if limit_cycle_found:
        lim_cycle_return["lengths"] = cycle_lengths  # type: ignore

    return factor_states, iter_idx, lim_cycle_return  # type: ignore


def activation(membrane_potential: np.ndarray) -> np.ndarray:
    temp = np.sign(membrane_potential)
    temp[temp == 0] = 1
    return temp


class LimitCycleCatcher:
    def __init__(self, state_space_size: int, max_lim_cycle_len: int = 20) -> None:
        # assert max_lim_cycle_len > 1, "limit cycles can be of length 2 or larger"
        self.state_buffers: Dict[int, np.ndarray] = {
            k: np.zeros([k, state_space_size], dtype=np.float32)
            for k in range(2, max_lim_cycle_len + 1)
        }
        self.buffer_repeat_counters: Dict[int, int] = {
            k: 0 for k in range(2, max_lim_cycle_len + 1)
        }

    def UpdateBuffers(self, new_state: np.ndarray, global_iter_idx: int) -> None:
        for buffer_sz in self.state_buffers:
            idx_in_buffer = global_iter_idx % buffer_sz
            if np.array_equal(self.state_buffers[buffer_sz][idx_in_buffer], new_state):
                self.buffer_repeat_counters[buffer_sz] += 1
            else:
                self.state_buffers[buffer_sz][idx_in_buffer] = new_state
                self.buffer_repeat_counters[buffer_sz] = 0

    def CheckForLimitCycle(self) -> bool:
        return any(
            self.buffer_repeat_counters[x] >= x for x in self.buffer_repeat_counters
        )

    def LengthSmallestLimCycle(self) -> int:
        valid_cycles = [
            x
            for x in self.buffer_repeat_counters
            if self.buffer_repeat_counters[x] >= x
        ]
        return min(valid_cycles) if valid_cycles else 0
