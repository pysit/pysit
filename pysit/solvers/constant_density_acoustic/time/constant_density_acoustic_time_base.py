from __future__ import division

import math

import numpy as np

from ..constant_density_acoustic_base import *

__all__ = ['ConstantDensityAcousticTimeBase']

__docformat__ = "restructuredtext en"

def reduce_or(bool_list):
    return reduce(operator.or_, bool_list)

class ConstantDensityAcousticTimeBase(ConstantDensityAcousticBase):

    supports_equation_dynamics = 'time'

    # These should be defined by subclasses.
    supports_temporal_integrator = None
    supports_temporal_accuracy_order = None
    supports_spatial_discretization = None
    supports_spatial_accuracy_order = None
    supports_kernel_implementation = None
    supports_spatial_dimension = None
    supports_boundary_conditions = None

    def __init__(self, mesh,
                       trange=(0.0,0.0),
                       cfl_safety=1/6,
                       **kwargs):

        self.trange = trange
        self.cfl_safety = cfl_safety

        self.t0, self.tf = trange
        self.dt = 0.0
        self.nsteps = 0

        self.time_accuracy_order = kwargs.get('time_accuracy_order')

        ConstantDensityAcousticBase.__init__(self, mesh, **kwargs)

    def _factory_validation_function(self, mesh, *args, **kwargs):

        valid_bc = True
        for i in xrange(mesh.dim):
            L = reduce_or([mesh.parameters[i].lbc.type in x for x in self.supports_boundary_conditions])
            R = reduce_or([mesh.parameters[i].rbc.type in x for x in self.supports_boundary_conditions])
            valid_bc &= L and R

        return (mesh.dim == self.supports_spatial_dimension and
                valid_bc and
                kwargs['equation_formulation'] == self.supports_equation_formulation and
                kwargs['temporal_integrator'] == self.supports_temporal_integrator and
                kwargs['temporal_accuracy_order'] in self.supports_temporal_accuracy_order and
                kwargs['spatial_discretization'] == self.supports_spatial_discretization and
                kwargs['spatial_accuracy_order'] in self.supports_spatial_accuracy_order and
                kwargs['kernel_implementation'] == self.supports_kernel_implementation)

    def ts(self):
        """Returns a numpy array of the time values serviced by the specified dt
        and trange."""
        return np.arange(self.nsteps)*self.dt

    def _process_mp_reset(self, *args, **kwargs):

        CFL = self.cfl_safety
        t0, tf = self.trange

        min_deltas = np.min(self.mesh.deltas)

        C = self._mp.C
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






