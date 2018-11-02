import numpy as np
import copy

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.frequency_modeling import FrequencyModeling

__all__ = ['SimpleExponetialObjective']

__docformat__ = "restructuredtext en"


class SimpleExponetialObjective(ObjectiveFunctionBase):

    """Class for a simple objective to test the Gradient Test Module"""

    def __init__(self, solver=None, parallel_wrap_shot=ParallelWrapShotNull()):
        """Construct the simple objective class
            We construct the following objective function:

            f(x) = exp(-0.5 * |x|^2)

            The corresponding gradient is

            g(x) = - exp(-0.5 * |x|^2) * x

            ____________________________________________________________________

            In this class, the input solver and parallel_wrap_shot are not
            necessary. The reason that we keep them is that we want to make this
            class consistant with the objective class of pysit. So that, they can
            be called in the same way in the Gradient Test. 

        """

        self.parallel_wrap_shot = parallel_wrap_shot
        self.solver = solver

    def evaluate(self, shots, m0):
        norm_m = np.linalg.norm(m0.data)
        norm_m2 = norm_m**2.0
        obj_value = np.exp(-0.5 * norm_m2)

        return obj_value

    def compute_gradient(self, shots, m0, aux_info):
        grad = copy.deepcopy(m0)
        norm_m = np.linalg.norm(m0.data)
        norm_m2 = norm_m**2.0
        grad.data = -np.exp(-0.5 * norm_m2) * m0.data

        return grad
