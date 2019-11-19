import numpy as np
import scipy.sparse as spsp
import os

from pysit.solvers.wavefield_vector import *
from .constant_density_acoustic_time_scalar_base import *

from pysit.util import Bunch
from pysit.util import PositiveEvenIntegers
from pysit.util.derivatives import build_derivative_matrix
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

from pysit.util.solvers import inherit_dict

from ._constant_density_acoustic_time_scalar_cpp import (
    constant_density_acoustic_time_scalar_3D_2os,
    constant_density_acoustic_time_scalar_3D_4os,
    constant_density_acoustic_time_scalar_3D_6os,
    constant_density_acoustic_time_scalar_3D_8os,
    constant_density_acoustic_time_scalar_3D_4omp,
    constant_density_acoustic_time_scalar_3D_6omp)

__all__ = ['ConstantDensityAcousticTimeScalar_3D_numpy',
           'ConstantDensityAcousticTimeScalar_3D_cpp',
           'ConstantDensityAcousticTimeScalar_3D_omp']

__docformat__ = "restructuredtext en"


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeScalar_3D(ConstantDensityAcousticTimeScalarBase):

    _local_support_spec = {'spatial_discretization': 'finite-difference',
                           'spatial_dimension': 3,
                           'boundary_conditions': ['pml', 'pml-sim', 'dirichlet']}

    def __init__(self, mesh, spatial_accuracy_order=4, **kwargs):

        self.operator_components = Bunch()

        self.spatial_accuracy_order = spatial_accuracy_order

        ConstantDensityAcousticTimeScalarBase.__init__(self,
                                                       mesh,
                                                       spatial_accuracy_order=spatial_accuracy_order,
                                                       **kwargs)
        if self.mesh.x.lbc.type == 'pml':
            if self.mesh.x.lbc.domain_bc.compact:
                raise NotImplementedError('Compact option is not available for time solvers')

    def _rebuild_operators(self):

        oc = self.operator_components

        built = oc.get('_base_components_built', False)

        # build the static components
        if not built:
            oc.sx = build_sigma(self.mesh, self.mesh.x)
            oc.sy = build_sigma(self.mesh, self.mesh.y)
            oc.sz = build_sigma(self.mesh, self.mesh.z)

            oc.sxPsyPsz = oc.sx + oc.sy + oc.sz
            oc.sxsyPsxszPsysz = oc.sx*oc.sy + oc.sx*oc.sz + oc.sy*oc.sz
            oc.sxsysz = oc.sx * oc.sy * oc.sz

            oc._base_components_built = True

    class WavefieldVector(WavefieldVectorBase):

        aux_names = ['psi', 'Phix', 'Phiy', 'Phiz']


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeScalar_3D_numpy(ConstantDensityAcousticTimeScalar_3D):

    _local_support_spec = {'kernel_implementation': 'numpy',
                           'spatial_accuracy_order': PositiveEvenIntegers,
                           'precision': ['single', 'double']}

    def _rebuild_operators(self):

        ConstantDensityAcousticTimeScalar_3D._rebuild_operators(self)

        dof = self.mesh.dof(include_bc=True)

        oc = self.operator_components

        built = oc.get('_numpy_components_built', False)

        # build the static components
        if not built:
            # build laplacian
            oc.L = build_derivative_matrix(self.mesh,
                                           2,
                                           self.spatial_accuracy_order)

            # build sigmax
            sx = build_sigma(self.mesh, self.mesh.x)
            oc.sigmax = make_diag_mtx(sx)

            # build sigmay
            sy = build_sigma(self.mesh, self.mesh.y)
            oc.sigmay = make_diag_mtx(sy)

            # build sigmaz
            sz = build_sigma(self.mesh, self.mesh.z)
            oc.sigmaz = make_diag_mtx(sz)

            # build Dx
            oc.minus_Dx = build_derivative_matrix(self.mesh,
                                                  1,
                                                  self.spatial_accuracy_order,
                                                  dimension='x')
            oc.minus_Dx.data *= -1

            # build Dy
            oc.minus_Dy = build_derivative_matrix(self.mesh,
                                                  1,
                                                  self.spatial_accuracy_order,
                                                  dimension='y')
            oc.minus_Dy.data *= -1

            # build Dz
            oc.minus_Dz = build_derivative_matrix(self.mesh,
                                                  1,
                                                  self.spatial_accuracy_order,
                                                  dimension='z')
            oc.minus_Dz.data *= -1

            # build other useful things
            oc.I = spsp.eye(dof, dof)
            oc.minus_I = -1*oc.I
            oc.empty = spsp.csr_matrix((dof, dof))

            # useful intermediates
            oc.sigma_sum_pair_prod = make_diag_mtx(sx*sy + sx*sz + sy*sz)
            oc.sigma_sum = make_diag_mtx(sx+sy+sz)
            oc.sigma_prod = make_diag_mtx(sx*sy*sz)
            oc.minus_sigma_yPzMx_Dx = make_diag_mtx(sy+sz-sx)*oc.minus_Dx
            oc.minus_sigma_xPzMy_Dy = make_diag_mtx(sx+sz-sy)*oc.minus_Dy
            oc.minus_sigma_xPyMz_Dz = make_diag_mtx(sx+sy-sz)*oc.minus_Dz

            oc.minus_sigma_yz_Dx = make_diag_mtx(sy*sz)*oc.minus_Dx
            oc.minus_sigma_zx_Dy = make_diag_mtx(sz*sx)*oc.minus_Dy
            oc.minus_sigma_xy_Dz = make_diag_mtx(sx*sy)*oc.minus_Dz

            oc._numpy_components_built = True

        C = self.model_parameters.C
        oc.m = make_diag_mtx((C**-2).reshape(-1,))

        K = spsp.bmat([[oc.m*oc.sigma_sum_pair_prod-oc.L, oc.m*oc.sigma_prod,   oc.minus_Dx, oc.minus_Dy, oc.minus_Dz],
                       [oc.minus_I,                       oc.empty,
                           oc.empty,    oc.empty,    oc.empty],
                       [oc.minus_sigma_yPzMx_Dx,          oc.minus_sigma_yz_Dx,
                           oc.sigmax,   oc.empty,    oc.empty],
                       [oc.minus_sigma_xPzMy_Dy,          oc.minus_sigma_zx_Dy,
                           oc.empty,    oc.sigmay,   oc.empty],
                       [oc.minus_sigma_xPyMz_Dz,          oc.minus_sigma_xy_Dz, oc.empty,    oc.empty,    oc.sigmaz]])

        C = spsp.bmat([[oc.m*oc.sigma_sum, oc.empty, oc.empty, oc.empty, oc.empty],
                       [oc.empty,          oc.I,     oc.empty, oc.empty, oc.empty],
                       [oc.empty,          oc.empty, oc.I,     oc.empty, oc.empty],
                       [oc.empty,          oc.empty, oc.empty, oc.I,     oc.empty],
                       [oc.empty,          oc.empty, oc.empty, oc.empty, oc.I]]) / self.dt

        M = spsp.bmat([[oc.m, oc.empty, oc.empty, oc.empty, oc.empty],
                       [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty],
                       [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty],
                       [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty],
                       [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty]]) / self.dt**2

        Stilde_inv = M+C
        Stilde_inv.data = 1./Stilde_inv.data

        self.A_k = Stilde_inv*(2*M - K + C)
        self.A_km1 = -1*Stilde_inv*(M)
        self.A_f = Stilde_inv


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeScalar_3D_cpp(ConstantDensityAcousticTimeScalar_3D):

    _local_support_spec = {'kernel_implementation': 'cpp',
                           'spatial_accuracy_order': [2, 4, 6, 8],
                           'precision': ['single', 'double']}

    _cpp_funcs = {2: constant_density_acoustic_time_scalar_3D_2os,
                  4: constant_density_acoustic_time_scalar_3D_4os,
                  6: constant_density_acoustic_time_scalar_3D_6os,
                  8: constant_density_acoustic_time_scalar_3D_8os}

    def time_step(self, solver_data, rhs_k, rhs_kp1):

        lpmlx = self.mesh.x.lbc.sigma if self.mesh.x.lbc.type is 'pml' else np.array([])
        rpmlx = self.mesh.x.rbc.sigma if self.mesh.x.rbc.type is 'pml' else np.array([])

        lpmly = self.mesh.y.lbc.sigma if self.mesh.y.lbc.type is 'pml' else np.array([])
        rpmly = self.mesh.y.rbc.sigma if self.mesh.y.rbc.type is 'pml' else np.array([])

        lpmlz = self.mesh.z.lbc.sigma if self.mesh.z.lbc.type is 'pml' else np.array([])
        rpmlz = self.mesh.z.rbc.sigma if self.mesh.z.rbc.type is 'pml' else np.array([])

        nx, ny, nz = self.mesh.shape(include_bc=True, as_grid=True)

        self._cpp_funcs[self.spatial_accuracy_order](solver_data.km1.u,
                                                     solver_data.k.Phix,
                                                     solver_data.k.Phiy,
                                                     solver_data.k.Phiz,
                                                     solver_data.k.psi,
                                                     solver_data.k.u,
                                                     self.model_parameters.C,
                                                     rhs_k,
                                                     lpmlx, rpmlx,
                                                     lpmly, rpmly,
                                                     lpmlz, rpmlz,
                                                     self.dt,
                                                     self.mesh.x.delta,
                                                     self.mesh.y.delta,
                                                     self.mesh.z.delta,
                                                     nx, ny, nz,
                                                     solver_data.kp1.Phix,
                                                     solver_data.kp1.Phiy,
                                                     solver_data.kp1.Phiz,
                                                     solver_data.kp1.psi,
                                                     solver_data.kp1.u)


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeScalar_3D_omp(ConstantDensityAcousticTimeScalar_3D):

    _local_support_spec = {'kernel_implementation': 'omp',
                           'spatial_accuracy_order': [4, 6],
                           'precision': ['single', 'double']}

    _omp_funcs = {4: constant_density_acoustic_time_scalar_3D_4omp,
                  6: constant_density_acoustic_time_scalar_3D_6omp}

    def time_step(self, solver_data, rhs_k, rhs_kp1):

        lpmlx = self.mesh.x.lbc.sigma if self.mesh.x.lbc.type is 'pml' else np.array([])
        rpmlx = self.mesh.x.rbc.sigma if self.mesh.x.rbc.type is 'pml' else np.array([])

        lpmly = self.mesh.y.lbc.sigma if self.mesh.y.lbc.type is 'pml' else np.array([])
        rpmly = self.mesh.y.rbc.sigma if self.mesh.y.rbc.type is 'pml' else np.array([])

        lpmlz = self.mesh.z.lbc.sigma if self.mesh.z.lbc.type is 'pml' else np.array([])
        rpmlz = self.mesh.z.rbc.sigma if self.mesh.z.rbc.type is 'pml' else np.array([])

        nx, ny, nz = self.mesh.shape(include_bc=True, as_grid=True)

        try:
            a = int(os.environ["OMP_NUM_THREADS"])
        except ValueError:
            raise ValueError('The enviroment variable \"OMP_NUM_THREADS\" has no integer\
                            set the value and relaunch your script')
        except KeyError:
            raise KeyError('The enviroment variable \"OMP_NUM_THREADS\" is not defined\
                          assign a value and relaunch your script')
        except:
            raise ImportError('The enviroment variable \"OMP_NUM_THREADS\" is unreadable\
                             please assign it an integer value')

        self._omp_funcs[self.spatial_accuracy_order](solver_data.km1.u,
                                                     solver_data.k.Phix,
                                                     solver_data.k.Phiy,
                                                     solver_data.k.Phiz,
                                                     solver_data.k.psi,
                                                     solver_data.k.u,
                                                     self.model_parameters.C,
                                                     rhs_k,
                                                     lpmlx, rpmlx,
                                                     lpmly, rpmly,
                                                     lpmlz, rpmlz,
                                                     self.dt,
                                                     self.mesh.x.delta,
                                                     self.mesh.y.delta,
                                                     self.mesh.z.delta,
                                                     nx, ny, nz,
                                                     solver_data.kp1.Phix,
                                                     solver_data.kp1.Phiy,
                                                     solver_data.kp1.Phiz,
                                                     solver_data.kp1.psi,
                                                     solver_data.kp1.u)
