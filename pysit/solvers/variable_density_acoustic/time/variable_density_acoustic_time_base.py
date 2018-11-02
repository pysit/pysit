

import math

import numpy as np

from ..variable_density_acoustic_base import *

from pysit.util.solvers import inherit_dict

__all__ = ['VariableDensityAcousticTimeBase']

__docformat__ = "restructuredtext en"

@inherit_dict('supports', '_local_support_spec')
class VariableDensityAcousticTimeBase(VariableDensityAcousticBase):

    _local_support_spec = {'equation_dynamics': 'time',
                           # These should be defined by subclasses.
                           'temporal_integrator': None,
                           'temporal_accuracy_order': None,
                           'spatial_discretization': None,
                           'spatial_accuracy_order': None,
                           'kernel_implementation': None,
                           'spatial_dimension': None,
                           'boundary_conditions': None,
                           'precision': None}

    def __init__(self,
                 mesh,
                 trange=(0.0, 1.0),
                 cfl_safety=1/6,
                 **kwargs):

        self.trange = trange
        self.cfl_safety = cfl_safety

        self.t0, self.tf = trange
        self.dt = 0.0
        self.nsteps = 0

        VariableDensityAcousticBase.__init__(self,
                                             mesh,
                                             trange=trange,
                                             cfl_safety=cfl_safety,
                                             **kwargs)

    def ts(self):
        """Returns a numpy array of the time values serviced by the specified dt
        and trange."""
        return np.arange(self.nsteps)*self.dt

    def _process_mp_reset(self, *args, **kwargs):

        CFL = self.cfl_safety
        t0, tf = self.trange

        min_deltas = np.min(self.mesh.deltas)
        kappa = self._mp.kappa
        rho = self._mp.rho

        C = (kappa/rho)**0.5
        max_C = max(abs(C.min()), C.max())  # faster than C.abs().max()

        dt = CFL*min_deltas / max_C
        nsteps = int(math.ceil((tf - t0)/dt))

        self.dt = dt
        self.nsteps = nsteps

        self._rebuild_operators()

    def _rebuild_operators(self, *args, **kwargs):
        pass

    def time_step(self, solver_data, rhs, **kwargs):
        raise NotImplementedError("Function 'time_step' Must be implemented by subclass.")






