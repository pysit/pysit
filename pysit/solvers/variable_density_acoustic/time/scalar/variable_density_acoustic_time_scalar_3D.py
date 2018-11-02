import numpy as np
import scipy.sparse as spsp

from pysit.solvers.wavefield_vector import *
from .variable_density_acoustic_time_scalar_base import *

from pysit.util import Bunch
from pysit.util import PositiveEvenIntegers
from pysit.util.derivatives import build_derivative_matrix
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

from pysit.util.solvers import inherit_dict


__all__ = ['VariableDensityAcousticTimeScalar_3D_numpy']

__docformat__ = "restructuredtext en"

### Currently not Implemented.

@inherit_dict('supports', '_local_support_spec')
class VariableDensityAcousticTimeScalar_3D(VariableDensityAcousticTimeScalarBase):

    _local_support_spec = {'spatial_discretization': 'finite-difference',
                           'spatial_dimension': 3,
                           'boundary_conditions': ['pml', 'pml-sim', 'dirichlet']}

    def __init__(self, mesh, spatial_accuracy_order=4, **kwargs):

        raise ValidationFunctionError(" This solver is in construction and is not yet complete. For variable density, only 2D time and frequency is currently running.")

        self.operator_components = Bunch()

        self.spatial_accuracy_order = spatial_accuracy_order

        VariableDensityAcousticTimeScalarBase.__init__(self,
                                                       mesh,
                                                       spatial_accuracy_order=spatial_accuracy_order,
                                                       **kwargs)

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
class VariableDensityAcousticTimeScalar_3D_numpy(VariableDensityAcousticTimeScalar_3D):

    _local_support_spec = {'kernel_implementation': 'numpy',
                           'spatial_accuracy_order': PositiveEvenIntegers,
                           'precision': ['single', 'double']}

    def _rebuild_operators(self):

        VariableDensityAcousticTimeScalar_3D._rebuild_operators(self)

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
            oc.sigma_sum_pair_prod  = make_diag_mtx(sx*sy + sx*sz + sy*sz)
            oc.sigma_sum            = make_diag_mtx(sx+sy+sz)
            oc.sigma_prod           = make_diag_mtx(sx*sy*sz)
            oc.minus_sigma_yPzMx_Dx = make_diag_mtx(sy+sz-sx)*oc.minus_Dx
            oc.minus_sigma_xPzMy_Dy = make_diag_mtx(sx+sz-sy)*oc.minus_Dy
            oc.minus_sigma_xPyMz_Dz = make_diag_mtx(sx+sy-sz)*oc.minus_Dz

            oc.minus_sigma_yz_Dx    = make_diag_mtx(sy*sz)*oc.minus_Dx
            oc.minus_sigma_zx_Dy    = make_diag_mtx(sz*sx)*oc.minus_Dy
            oc.minus_sigma_xy_Dz    = make_diag_mtx(sx*sy)*oc.minus_Dz

            oc._numpy_components_built = True

        C = self.model_parameters.C
        oc.m = make_diag_mtx((C**-2).reshape(-1,))

        K = spsp.bmat([[oc.m*oc.sigma_sum_pair_prod-oc.L, oc.m*oc.sigma_prod,   oc.minus_Dx, oc.minus_Dy, oc.minus_Dz ],
                       [oc.minus_I,                       oc.empty,             oc.empty,    oc.empty,    oc.empty    ],
                       [oc.minus_sigma_yPzMx_Dx,          oc.minus_sigma_yz_Dx, oc.sigmax,   oc.empty,    oc.empty    ],
                       [oc.minus_sigma_xPzMy_Dy,          oc.minus_sigma_zx_Dy, oc.empty,    oc.sigmay,   oc.empty    ],
                       [oc.minus_sigma_xPyMz_Dz,          oc.minus_sigma_xy_Dz, oc.empty,    oc.empty,    oc.sigmaz   ]])

        C = spsp.bmat([[oc.m*oc.sigma_sum, oc.empty, oc.empty, oc.empty, oc.empty],
                       [oc.empty,          oc.I,     oc.empty, oc.empty, oc.empty],
                       [oc.empty,          oc.empty, oc.I,     oc.empty, oc.empty],
                       [oc.empty,          oc.empty, oc.empty, oc.I,     oc.empty],
                       [oc.empty,          oc.empty, oc.empty, oc.empty, oc.I    ]]) / self.dt

        M = spsp.bmat([[    oc.m, oc.empty, oc.empty, oc.empty, oc.empty],
                       [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty],
                       [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty],
                       [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty],
                       [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty]]) / self.dt**2

        Stilde_inv = M+C
        Stilde_inv.data = 1./Stilde_inv.data

        self.A_k   = Stilde_inv*(2*M - K + C)
        self.A_km1 = -1*Stilde_inv*(M)
        self.A_f   = Stilde_inv