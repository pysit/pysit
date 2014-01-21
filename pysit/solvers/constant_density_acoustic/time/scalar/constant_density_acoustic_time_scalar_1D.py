import numpy as np
import scipy.sparse as spsp

from pysit.solvers.wavefield_vector import *
from constant_density_acoustic_time_scalar_base import *

from pysit.util import Bunch
from pysit.util import PositiveEvenIntegers
from pysit.util.derivatives import build_derivative_matrix
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

from pysit.util.solvers import inherit_dict

from constant_density_acoustic_time_scalar_cpp import (
    constant_density_acoustic_time_scalar_1D_2os,
    constant_density_acoustic_time_scalar_1D_4os,
    constant_density_acoustic_time_scalar_1D_6os,
    constant_density_acoustic_time_scalar_1D_8os)

__all__ = ['ConstantDensityAcousticTimeScalar_1D_numpy',
           'ConstantDensityAcousticTimeScalar_1D_cpp']

__docformat__ = "restructuredtext en"


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeScalar_1D(ConstantDensityAcousticTimeScalarBase):

    _local_support_spec = {'spatial_discretization': 'finite-difference',
                           'spatial_dimension': 1,
                           'boundary_conditions': ['pml', 'pml-sim', 'dirichlet']}

    def __init__(self, mesh, **kwargs):

        self.operator_components = Bunch()

        ConstantDensityAcousticTimeScalarBase.__init__(self, mesh, **kwargs)

    def _rebuild_operators(self):

        oc = self.operator_components

        built = oc.get('_base_components_built', False)

        # build the static components
        if not built:
            oc.sz = build_sigma(self.mesh, self.mesh.z)

            oc._base_components_built = True

    class WavefieldVector(WavefieldVectorBase):

        aux_names = ['Phiz']


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeScalar_1D_numpy(ConstantDensityAcousticTimeScalar_1D):

    _local_support_spec = {'kernel_implementation': 'numpy',
                           'spatial_accuracy_order': PositiveEvenIntegers}

    def _rebuild_operators(self):

        ConstantDensityAcousticTimeScalar_1D._rebuild_operators(self)

        dof = self.mesh.dof(include_bc=True)

        oc = self.operator_components

        built = oc.get('_numpy_components_built', False)

        # build the static components
        if not built:
            # build laplacian
            oc.L = build_derivative_matrix(self.mesh,
                                           2,
                                           self.spatial_accuracy_order,
                                           use_shifted_differences=self.spatial_shifted_differences)

            # build sigmaz
            sz = build_sigma(self.mesh, self.mesh.z)
            oc.sigmaz = make_diag_mtx(sz)

            # build Dz
            oc.Dz = build_derivative_matrix(self.mesh,
                                            1,
                                            self.spatial_accuracy_order,
                                            dimension='z',
                                            use_shifted_differences=self.spatial_shifted_differences)

            # build other useful things
            oc.I = spsp.eye(dof, dof)
            oc.empty = spsp.csr_matrix((dof, dof))

            # Stiffness matrix K doesn't change
            oc.K = spsp.bmat([[          -oc.L,    -oc.Dz],
                              [oc.sigmaz*oc.Dz, oc.sigmaz]])

            oc._numpy_components_built = True

        C = self.model_parameters.C
        oc.m = make_diag_mtx((C**-2).reshape(-1,))

        C = spsp.bmat([[oc.sigmaz*oc.m, None],
                       [          None, oc.I]]) / self.dt

        M = spsp.bmat([[oc.m,     None],
                       [None, oc.empty]]) / self.dt**2

        Stilde_inv = M+C
        Stilde_inv.data = 1./Stilde_inv.data

        self.A_k   = Stilde_inv*(2*M - oc.K + C)
        self.A_km1 = -1*Stilde_inv*(M)
        self.A_f   = Stilde_inv


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeScalar_1D_cpp(ConstantDensityAcousticTimeScalar_1D):

    _local_support_spec = {'kernel_implementation': 'cpp',
                           'spatial_accuracy_order': [2, 4, 6, 8]}

    _cpp_funcs = {2: constant_density_acoustic_time_scalar_1D_2os,
                  4: constant_density_acoustic_time_scalar_1D_4os,
                  6: constant_density_acoustic_time_scalar_1D_6os,
                  8: constant_density_acoustic_time_scalar_1D_8os}

    def time_step(self, solver_data, rhs_k, rhs_kp1):

        lpmlz = self.mesh.z.lbc.sigma if self.mesh.z.lbc.type is 'pml' else np.array([])
        rpmlz = self.mesh.z.rbc.sigma if self.mesh.z.rbc.type is 'pml' else np.array([])
        nz = self.mesh.dof(include_bc=True)

        self._cpp_funcs[self.spatial_accuracy_order](solver_data.km1.u,
                                                     solver_data.k.Phiz,
                                                     solver_data.k.u,
                                                     self.model_parameters.C,
                                                     rhs_k,
                                                     lpmlz, rpmlz,
                                                     self.dt,
                                                     self.mesh.z.delta,
                                                     nz,
                                                     solver_data.kp1.Phiz,
                                                     solver_data.kp1.u)
