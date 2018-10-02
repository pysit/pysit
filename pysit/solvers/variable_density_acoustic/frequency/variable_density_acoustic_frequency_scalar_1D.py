import scipy.sparse as spsp

from pysit.solvers.wavefield_vector import *
from .variable_density_acoustic_frequency_scalar_base import *

from pysit.util import Bunch
from pysit.util import PositiveEvenIntegers
from pysit.util.derivatives import build_derivative_matrix
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

from pysit.util.solvers import inherit_dict

__all__ = ['VariableDensityAcousticFrequencyScalar_1D']

__docformat__ = "restructuredtext en"

### Currently not Implemented.

@inherit_dict('supports', '_local_support_spec')
class VariableDensityAcousticFrequencyScalar_1D(VariableDensityAcousticFrequencyScalarBase):

    _local_support_spec = {'spatial_discretization': 'finite-difference',
                           'spatial_dimension': 1,
                           'spatial_accuracy_order': PositiveEvenIntegers,
                           'boundary_conditions': ['pml', 'pml-sim', 'dirichlet'],
                           'precision': ['single', 'double']}

    def __init__(self,
                 mesh,
                 spatial_accuracy_order=4,
                 spatial_shifted_differences=True,
                 **kwargs):

        raise ValidationFunctionError(" This solver is in construction and is not yet complete. For variable density, only 2D time and frequency is currently running.")

        self.operator_components = Bunch()

        self.spatial_accuracy_order = spatial_accuracy_order

        self.spatial_shifted_differences = spatial_shifted_differences

        VariableDensityAcousticFrequencyScalarBase.__init__(self,
                                                            mesh,
                                                            spatial_accuracy_order=spatial_accuracy_order,
                                                            spatial_shifted_differences=spatial_shifted_differences,
                                                            **kwargs)

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
            oc.I     = spsp.eye(dof, dof)
            oc.empty = spsp.csr_matrix((dof, dof))

            # Stiffness matrix K doesn't change
            self.K = spsp.bmat([[           -oc.L,    -oc.Dz],
                                [ oc.sigmaz*oc.Dz, oc.sigmaz]])

            oc._numpy_components_built = True

        C = self.model_parameters.C
        oc.m = make_diag_mtx((C**-2).reshape(-1,))

        self.C = spsp.bmat([[oc.sigmaz*oc.m, None],
                            [None,           oc.I]])

        self.M = spsp.bmat([[oc.m,     None],
                            [None, oc.empty]])

    class WavefieldVector(WavefieldVectorBase):

        aux_names = ['Phiz']
