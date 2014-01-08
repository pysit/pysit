from __future__ import division

import math

import numpy as np

from ..constant_density_acoustic_base import *

__all__ = ['ConstantDensityAcousticTimeBase']

__docformat__ = "restructuredtext en"

class ConstantDensityAcousticTimeBase(ConstantDensityAcousticBase):

    equation_dynamics = 'time'

    # These should be defined by subclasses.
    temporal_integrator = None
    spatial_discretization = None
    spatial_accuracy_order = None
    kernel_implementation = None
    spatial_dimension = None
    boundary_conditions = None

    def __init__(self, mesh,
                       trange=(0.0,0.0),
                       cfl_safety=1/6,
                       time_accuracy_order=2,
                       **kwargs):

        self.trange = trange
        self.cfl_safety = cfl_safety

        self.t0, self.tf = trange
        self.dt = 0.0
        self.nsteps = 0

        self.time_accuracy_order = time_accuracy_order

        ConstantDensityAcousticBase.__init__(self, mesh, **kwargs)

    def _factory_validation_function(self, mesh, *args, **kwargs):

        valid_bc = True
        for i in xrange(mesh.dim):
            L = reduce(operator.or_, [mesh.parameters[i].lbc.type in x for x in self.boundary_conditions])
            R = reduce(operator.or_, [mesh.parameters[i].rbc.type in x for x in self.boundary_conditions])
            valid_bc &= L and R

        return (mesh.dim == self.spatial_dimension and
                valid_bc and
                kwargs['equation_formulation'] == self.equation_formulation and
                kwargs['temporal_integrator'] == self.temporal_integrator and
                kwargs['spatial_discretization'] == self.spatial_discretization and
                kwargs['spatial_accuracy_order'] in self.spatial_accuracy_order and
                kwargs['kernel_implementation'] == self.kernel_implementation and
                kwargs['equation_formulation'] == self.equation_formulation)

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

        # If we are not using CPU acceleration, the operators need to be rebuilt
        if not self.use_cpp_acceleration:
            self._rebuild_operators()

    def time_step(self, solver_data, rhs, **kwargs):
        raise NotImplementedError("Function 'time_step' Must be implemented by subclass.")






