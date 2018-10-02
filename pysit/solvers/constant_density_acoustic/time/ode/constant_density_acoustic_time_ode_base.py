from ..constant_density_acoustic_time_base import *
from pysit.solvers.solver_data import SolverDataTimeBase

from pysit.util.solvers import inherit_dict

__all__ = ['ConstantDensityAcousticTimeODEBase']

__docformat__ = "restructuredtext en"

multistep_coeffs = {1: [],
                    2: [],
                    3: [],
                    4: []}

class _ConstantDensityAcousticTimeODE_SolverData(SolverDataTimeBase):

    def __init__(self, solver, temporal_accuracy_order, integrator, **kwargs):

        self.solver = solver

        self.temporal_accuracy_order = temporal_accuracy_order

        # self.us[0] is kp1, [1] is k or current, [2] is km1, [3] is km2, etc
        self.us = [solver.WavefieldVector(solver.mesh, dtype=solver.dtype) for x in range(3)]

        if integrator == 'multistep':
            self.u_primes = [solver.WavefieldVector(solver.mesh, dtype=solver.dtype) for x in range(temporal_accuracy_order)]
        else:
            self.u_primes = list()

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

    def advance(self):

        self.us[-1] *= 0
        self.us.insert(0, self.us.pop(-1))

        if len(self.u_primes) > 0:
            self.u_primes[-1] *= 0
            self.u_primes.insert(0, self.us.pop(-1))


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeODEBase(ConstantDensityAcousticTimeBase):

    _local_support_spec = {'equation_formulation': 'ode',
                           'temporal_integrator': ['rk', 'runge-kutta'],
                           'temporal_accuracy_order': [2, 4]}

    def __init__(self,
                 mesh,
                 temporal_integrator='rk',
                 temporal_accuracy_order=4,
                 **kwargs):

        self.temporal_integrator = temporal_integrator
        self.temporal_accuracy_order = temporal_accuracy_order

        self.A = None

        ConstantDensityAcousticTimeBase.__init__(self,
                                                 mesh,
                                                 temporal_integrator='rk',
                                                 temporal_accuracy_order=temporal_accuracy_order,
                                                 **kwargs)

    def time_step(self, solver_data, rhs_k, rhs_kp1):

        if self.temporal_integrator == 'rk':
            if self.temporal_accuracy_order == 4:
                self._rk4(solver_data, rhs_k, rhs_kp1)
            else:
                self._rk2(solver_data, rhs_k, rhs_kp1)

    def _ode_rhs(self, u_bar, f):
        data = self.A*u_bar.data
        step = self.WavefieldVector(u_bar.mesh, dtype=u_bar.dtype, data=data)
        step.v += self.operator_components.m_inv*f
        return step

    def _rk4(self, solver_data, rhs_k, rhs_kp1):
        u_bar = solver_data.k

        k = self.dt * self._ode_rhs(u_bar, rhs_k)
        solver_data.kp1 += u_bar + (1./6)*k

        rhs_half = 0.5*(rhs_k + rhs_kp1)

        k = self.dt * self._ode_rhs(u_bar + 0.5*k, rhs_half)
        solver_data.kp1 += (1./3)*k

        k = self.dt * self._ode_rhs(u_bar + 0.5*k, rhs_half)
        solver_data.kp1 += (1./3)*k

        k = self.dt * self._ode_rhs(u_bar + k, rhs_kp1)
        solver_data.kp1 += (1./6)*k

    def _rk2(self, solver_data, rhs_k, rhs_kp1):
        u_bar = solver_data.k

        k1 = self.dt*self._ode_rhs(u_bar, rhs_k)
        k2 = self.dt*self._ode_rhs(u_bar+k1, rhs_kp1)
        solver_data.kp1 += u_bar + 0.5*(k1+k2)

    _SolverData = _ConstantDensityAcousticTimeODE_SolverData

    def SolverData(self, *args, **kwargs):
        return self._SolverData(self,
                                self.temporal_accuracy_order,
                                self.temporal_integrator,
                                **kwargs)
