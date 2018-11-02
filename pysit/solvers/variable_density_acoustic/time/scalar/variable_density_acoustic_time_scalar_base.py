from ..variable_density_acoustic_time_base import *
from pysit.solvers.solver_data import SolverDataTimeBase

from pysit.util.solvers import inherit_dict

__all__ = ['VariableDensityAcousticTimeScalarBase']

__docformat__ = "restructuredtext en"

class _VariableDensityAcousticTimeScalar_SolverData(SolverDataTimeBase):

    def __init__(self, solver, temporal_accuracy_order, **kwargs):

        self.solver = solver

        self.temporal_accuracy_order = temporal_accuracy_order

        # self.us[0] is kp1, [1] is k or current, [2] is km1, [3] is km2, etc
        self.us = [solver.WavefieldVector(solver.mesh, dtype=solver.dtype) for x in range(3)]

    def advance(self):

        self.us[-1] *= 0
        self.us.insert(0, self.us.pop(-1))

    @property
    def kp1(self):
        return self.us[0]

    @kp1.setter
    def kp1(self, arg):
        self.us[0] = arg

    @property
    def k(self):
        return self.us[1]

    @k.setter
    def k(self, arg):
        self.us[1] = arg

    @property
    def km1(self):
        return self.us[2]

    @km1.setter
    def km1(self, arg):
        self.us[2] = arg


@inherit_dict('supports', '_local_support_spec')
class VariableDensityAcousticTimeScalarBase(VariableDensityAcousticTimeBase):

    _local_support_spec = {'equation_formulation': 'scalar',
                           'temporal_integrator': 'leap-frog',
                           'temporal_accuracy_order': 2}

    def __init__(self, mesh, **kwargs):

        self.A_km1 = None
        self.A_k   = None
        self.A_f   = None

        self.temporal_accuracy_order = 2

        VariableDensityAcousticTimeBase.__init__(self, mesh, **kwargs)

    def time_step(self, solver_data, rhs_k, rhs_kp1):
        u_km1 = solver_data.km1
        u_k   = solver_data.k
        u_kp1 = solver_data.kp1

        f_bar = self.WavefieldVector(self.mesh, dtype=self.dtype)
        f_bar.u += rhs_k

        u_kp1 += self.A_k*u_k.data + self.A_km1*u_km1.data + self.A_f*f_bar.data

    _SolverData = _VariableDensityAcousticTimeScalar_SolverData

    def SolverData(self, *args, **kwargs):
        return self._SolverData(self, self.temporal_accuracy_order, **kwargs)
