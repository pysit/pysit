import numpy as np

from solver_base import *
from model_parameter import *

__all__=['SolverBase']

__docformat__ = "restructuredtext en"

class ConstantDensityAcousticBase(SolverBase):
    """ Base class for solvers that use the Constant Density Acoustic Model
    (e.g., in the wave, helmholtz, and laplace domains).

    """

    ModelParameters = ConstantDensityAcousticParameters

    SolverData = None


