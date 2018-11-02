import numpy as np

from pysit.solvers.solver_base import *
from pysit.solvers.model_parameter import *

from pysit.util.solvers import inherit_dict

__all__ = ['ConstantDensityAcousticBase']

__docformat__ = "restructuredtext en"


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticBase(SolverBase):
    """ Base class for solvers that use the Constant Density Acoustic Model
    (e.g., in the wave, helmholtz, and laplace domains).

    """

    _local_support_spec = {'equation_physics': 'constant-density-acoustic'}

    ModelParameters = ConstantDensityAcousticParameters

    def _compute_dWaveOp_time(self, solver_data):
        ukm1 = solver_data.km1.primary_wavefield
        uk   = solver_data.k.primary_wavefield
        ukp1 = solver_data.kp1.primary_wavefield

        if self.mesh.dim == 1:
            dPMLu = solver_data.solver.operator_components.sz.reshape((1, -1)).T*(ukp1-ukm1)/(2.0*self.dt)
        elif self.mesh.dim == 2:
            dPMLu = solver_data.solver.operator_components.sxPsz.reshape((1, -1)).T*(ukp1-ukm1)/(2.0*self.dt) \
                    + solver_data.solver.operator_components.sxsz.reshape((1, -1)).T*uk
        else:
            psik = solver_data.k.psi
            dPMLu = solver_data.solver.operator_components.sxPsyPsz.reshape((1, -1)).T*(ukp1-ukm1)/(2.0*self.dt) \
                + solver_data.solver.operator_components.sxsyPsxszPsysz.reshape((1, -1)).T*uk \
                + solver_data.solver.operator_components.sxsysz.reshape((1, -1)).T*psik
                
        return (ukp1-2*uk+ukm1)/(self.dt**2) + dPMLu

    def _compute_dWaveOp_frequency(self, uk_hat, nu):
        omega = (2*np.pi*nu)
        Bmat = -(omega)**2 * self.dM + omega * 1j * self.dC + self.dK
        return Bmat * uk_hat
        # Comment out by Zhilong
        # omega2 = (2*np.pi*nu)**2.0
        # return -1*omega2*uk_hat

    def _compute_dWaveOp_laplace(self, *args):
        raise NotImplementedError('Derivative Laplace domain operator not yet implemented.')
