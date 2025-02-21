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

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.


from typing import Dict

import numpy as np

from src.paperwings.exceptions.res_net_exception import ResNetException
from src.paperwings.util.logging_util import LoggingUtil


class LimitCycleCatcher:
    """
    A class for detecting limit cycles in a time series of states.
    """

    LOGGER = LoggingUtil.instance("<LimitCycleCatcher>")

    def __init__(self, state_space_size: int, max_lim_cycle_len: int = 20) -> None:

        self.state_buffers: Dict[int, np.ndarray] = {
            k: np.zeros([k, state_space_size], dtype=np.float32)
            for k in range(2, max_lim_cycle_len + 1)
        }
        self.buffer_repeat_counters: Dict[int, int] = {
            k: 0 for k in range(2, max_lim_cycle_len + 1)
        }

    def update_buffers(self, new_state: np.ndarray, global_iter_idx: int) -> None:
        """
        Update the state buffers with the new state and increment the repeat counter.
        """

        try:
            for buffer_sz in self.state_buffers:
                idx_in_buffer = global_iter_idx % buffer_sz
                if np.array_equal(
                    self.state_buffers[buffer_sz][idx_in_buffer], new_state
                ):
                    self.buffer_repeat_counters[buffer_sz] += 1
                else:
                    self.state_buffers[buffer_sz][idx_in_buffer] = new_state
                    self.buffer_repeat_counters[buffer_sz] = 0
        except Exception as error:
            self.LOGGER.error(str(error))
            raise ResNetException(str(error))

    def check_for_limit_cycle(self) -> bool:
        """
        Check if a limit cycle has been detected.
        """

        try:
            return any(
                self.buffer_repeat_counters[x] >= x for x in self.buffer_repeat_counters
            )
        except Exception as error:
            self.LOGGER.error(str(error))
            raise ResNetException(str(error))

    def length_smallest_lim_cycle(self) -> int:
        """
        Return the length of the smallest limit cycle detected.
        """
        try:
            valid_cycles = [
                x
                for x in self.buffer_repeat_counters
                if self.buffer_repeat_counters[x] >= x
            ]
            return min(valid_cycles) if valid_cycles else 0
        except Exception as error:
            self.LOGGER.error(str(error))
            raise ResNetException(str(error))
