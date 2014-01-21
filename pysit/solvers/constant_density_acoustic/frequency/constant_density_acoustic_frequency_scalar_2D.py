import scipy.sparse as spsp

from pysit.solvers.wavefield_vector import *
from constant_density_acoustic_frequency_scalar_base import *

from pysit.util import Bunch
from pysit.util.derivatives import build_derivative_matrix
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

from pysit.util.solvers import inherit_dict

__all__ = ['ConstantDensityAcousticFrequencyScalar_2D']

__docformat__ = "restructuredtext en"


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticFrequencyScalar_2D(ConstantDensityAcousticFrequencyScalarBase):

    _local_support_spec = {'spatial_discretization': 'finite-difference',
                           'spatial_dimension': 2,
                           'boundary_conditions': ['pml', 'pml-sim', 'dirichlet']}

    def __init__(self, mesh, **kwargs):

        self.operator_components = Bunch()

        ConstantDensityAcousticFrequencyScalarBase.__init__(self, mesh, **kwargs)

    def _rebuild_operators(self):

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

            # build sigmax
            sx = build_sigma(self.mesh, self.mesh.x)
            oc.sigmax = make_diag_mtx(sx)

            # build sigmaz
            sz = build_sigma(self.mesh, self.mesh.z)
            oc.sigmaz = make_diag_mtx(sz)

            # build Dx
            oc.minus_Dx = build_derivative_matrix(self.mesh,
                                                  1,
                                                  self.spatial_accuracy_order,
                                                  dimension='x',
                                                  use_shifted_differences=self.spatial_shifted_differences)
            oc.minus_Dx.data *= -1

            # build Dz
            oc.minus_Dz = build_derivative_matrix(self.mesh,
                                                  1,
                                                  self.spatial_accuracy_order,
                                                  dimension='z',
                                                  use_shifted_differences=self.spatial_shifted_differences)
            oc.minus_Dz.data *= -1

            # build other useful things
            oc.I = spsp.eye(dof, dof)
            oc.empty = spsp.csr_matrix((dof, dof))

            # useful intermediates
            oc.sigma_xz  = make_diag_mtx(sx*sz)
            oc.sigma_xPz = oc.sigmax + oc.sigmaz

            oc.minus_sigma_zMx_Dx = make_diag_mtx((sz-sx))*oc.minus_Dx
            oc.minus_sigma_xMz_Dz = make_diag_mtx((sx-sz))*oc.minus_Dz

            oc._numpy_components_built = True

        C = self.model_parameters.C
        oc.m = make_diag_mtx((C**-2).reshape(-1,))

        self.K = spsp.bmat([[oc.m*oc.sigma_xz-oc.L, oc.minus_Dx, oc.minus_Dz ],
                            [oc.minus_sigma_zMx_Dx, oc.sigmax,   oc.empty    ],
                            [oc.minus_sigma_xMz_Dz, oc.empty,    oc.sigmaz   ]])

        self.C = spsp.bmat([[oc.m*oc.sigma_xPz, oc.empty, oc.empty],
                            [oc.empty,          oc.I,     oc.empty],
                            [oc.empty,          oc.empty, oc.I    ]])

        self.M = spsp.bmat([[    oc.m, oc.empty, oc.empty],
                            [oc.empty, oc.empty, oc.empty],
                            [oc.empty, oc.empty, oc.empty]])

    class WavefieldVector(WavefieldVectorBase):

        aux_names = ['Phix', 'Phiz']
