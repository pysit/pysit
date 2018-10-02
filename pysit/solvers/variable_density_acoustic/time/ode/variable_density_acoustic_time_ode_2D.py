import numpy as np
import scipy.sparse as spsp

from pysit.solvers.wavefield_vector import *
from .variable_density_acoustic_time_ode_base import *

from pysit.util import Bunch
from pysit.util import PositiveEvenIntegers
from pysit.util.derivatives import build_derivative_matrix
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

from pysit.util.solvers import inherit_dict

__all__ = ['VariableDensityAcousticTimeODE_2D']

__docformat__ = "restructuredtext en"

### Currently not Implemented.

@inherit_dict('supports', '_local_support_spec')
class VariableDensityAcousticTimeODE_2D(VariableDensityAcousticTimeODEBase):

    _local_support_spec = {'kernel_implementation': 'numpy',
                           'spatial_discretization': 'finite-difference',
                           'spatial_dimension': 2,
                           'spatial_accuracy_order': PositiveEvenIntegers,
                           'boundary_conditions': ['pml', 'pml-sim', 'dirichlet'],
                           'precision': ['single', 'double']}

    def __init__(self, mesh, spatial_accuracy_order=4, **kwargs):

        raise ValidationFunctionError(" This solver is in construction and is not yet complete. For variable density, only 2D time and frequency is currently running.")

        self.operator_components = Bunch()

        self.spatial_accuracy_order = spatial_accuracy_order

        VariableDensityAcousticTimeODEBase.__init__(self,
                                                    mesh,
                                                    spatial_accuracy_order=spatial_accuracy_order,
                                                    **kwargs)

    def _rebuild_operators(self):

        dof = self.mesh.dof(include_bc=True)

        # a more readable reference
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
            oc.minus_sigmax = make_diag_mtx(-sx)

            # build sigmaz
            sz = build_sigma(self.mesh, self.mesh.z)
            oc.minus_sigmaz = make_diag_mtx(-sz)

            # build Dx
            oc.Dx = build_derivative_matrix(self.mesh,
                                            1,
                                            self.spatial_accuracy_order,
                                            dimension='x')

            # build Dz
            oc.Dz = build_derivative_matrix(self.mesh,
                                            1,
                                            self.spatial_accuracy_order,
                                            dimension='z')

            # build other useful things
            oc.I     = spsp.eye(dof, dof)
            oc.empty = spsp.csr_matrix((dof, dof))

            # useful intermediates
            oc.sigma_xz = make_diag_mtx(sx*sz)
            oc.minus_sigma_xPz = oc.minus_sigmax + oc.minus_sigmaz
            oc.sigma_zMx_Dx = make_diag_mtx(sz-sx)*oc.Dx
            oc.sigma_xMz_Dz = make_diag_mtx(sx-sz)*oc.Dz

            oc._numpy_components_built = True

        C = self.model_parameters.C
        oc.m_inv = make_diag_mtx((C**2).reshape(-1,))

        self.A = spsp.bmat([[oc.empty,                  oc.I,               oc.empty,        oc.empty        ],
                            [oc.m_inv*oc.L-oc.sigma_xz, oc.minus_sigma_xPz, oc.m_inv*oc.Dx,  oc.m_inv*oc.Dz  ],
                            [oc.sigma_zMx_Dx,           oc.empty,           oc.minus_sigmax, oc.empty        ],
                            [oc.sigma_xMz_Dz,           oc.empty,           oc.empty,        oc.minus_sigmaz ]])

    class WavefieldVector(WavefieldVectorBase):

        aux_names = ['v', 'Phix', 'Phiz']
