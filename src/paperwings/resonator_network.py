#  Copyright (c) 2025. Prediction By Invention https://predictionbyinvention.com/
#
#  THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#  PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from typing import Dict, Union

import numpy as np


class ResonatorNetwork:
    """
    NumPy implementation of (discrete-time, bipolar) Resonator Networks
    Based on https://github.com/spencerkent/resonator-networks
    """

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
        composite_vec: np.ndarray,
        factor_codebooks: Dict[Union[int, str], np.ndarray],
        lim_cycle_detection_len: int = 0,
    ):
        # assert composite_vec.dtype == np.int8
        self.composite_vec = composite_vec
        self.factor_codebooks = factor_codebooks
        self.synapse_type = self.DEFAULT_SYNAPSE_TYPE

        self.factor_states: Dict[Union[int, str], np.ndarray] = dict.fromkeys(factor_codebooks)  # type: ignore
