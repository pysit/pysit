import copy
import numpy as np
import scipy.sparse as spsp

from pysit.core import CartesianMesh, RectangularDomain
from pysit.solvers.wavefield_vector import *
from variable_density_acoustic_time_scalar_base import *

from pysit.util import Bunch
from pysit.util import PositiveEvenIntegers
from pysit.util.derivatives import build_derivative_matrix, build_derivative_matrix_VDA
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

from pysit.util.solvers import inherit_dict

__all__ = ['VariableDensityAcousticTimeScalar_2D_numpy']

__docformat__ = "restructuredtext en"


@inherit_dict('supports', '_local_support_spec')
class VariableDensityAcousticTimeScalar_2D(VariableDensityAcousticTimeScalarBase):

    _local_support_spec = {'spatial_discretization': 'finite-difference',
                           'spatial_dimension': 2,
                           'boundary_conditions': ['pml', 'pml-sim', 'dirichlet']}

    def __init__(self, mesh, spatial_accuracy_order=4, **kwargs):

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
            oc.sz = build_sigma(self.mesh, self.mesh.z)

            oc.sxPsz = oc.sx + oc.sz
            oc.sxsz = oc.sx * oc.sz

            oc._base_components_built = True

    class WavefieldVector(WavefieldVectorBase):

        aux_names = ['Phix', 'Phiz']


@inherit_dict('supports', '_local_support_spec')
class VariableDensityAcousticTimeScalar_2D_numpy(VariableDensityAcousticTimeScalar_2D):

    _local_support_spec = {'kernel_implementation': 'numpy',
                           'spatial_accuracy_order': PositiveEvenIntegers,
                           'precision': ['single', 'double']}

    def _rebuild_operators(self):
    
            VariableDensityAcousticTimeScalar_2D._rebuild_operators(self)
    
            dof = self.mesh.dof(include_bc=True)
    
            oc = self.operator_components
    
            built = oc.get('_numpy_components_built', False)
    
            # build the static components
            if not built:
                # build sigmax
                sx = build_sigma(self.mesh, self.mesh.x)
                oc.sigmax = make_diag_mtx(sx)
    
                # build sigmaz
                sz = build_sigma(self.mesh, self.mesh.z)
                oc.sigmaz = make_diag_mtx(sz)
    
                # build Dx
                oc.Dx = build_derivative_matrix(self.mesh,
                                                        1,
                                                        self.spatial_accuracy_order,
                                                        dimension='x')
                oc.minus_Dx = copy.deepcopy(oc.Dx)      #more storage, but less computations
                oc.minus_Dx.data *= -1                  
    
                # build Dz
                oc.Dz = build_derivative_matrix(self.mesh,
                                                        1,
                                                        self.spatial_accuracy_order,
                                                        dimension='z')
                oc.minus_Dz = copy.deepcopy(oc.Dz)      #more storage, but less computations
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

            kappa = self.model_parameters.kappa
            rho = self.model_parameters.rho
    
            oc.m1 = make_diag_mtx((kappa**-1).reshape(-1,))
            oc.m2 = make_diag_mtx((rho**-1).reshape(-1,))

            oc.L = build_derivative_matrix_VDA(self.mesh,
                                               2,
                                               self.spatial_accuracy_order,
                                               alpha = rho**-1
                                               )
            
    
            # oc.L is a heterogenous laplacian operator. It computes div(m2 grad), where m2 = 1/rho. 
            # Ian's implementation used the regular Dx operators in the PML even though the heterogeneous Laplacian with staggered derivative operators is used in the physical domain.
            # I (Bram) did not change this or investigate if this causes some problems but am noting it here for completeness. 
            
            K = spsp.bmat([[oc.m1*oc.sigma_xz-oc.L, oc.minus_Dx*oc.m2, oc.minus_Dz*oc.m2 ],
                           [oc.minus_sigma_zMx_Dx, oc.sigmax,   oc.empty    ],
                           [oc.minus_sigma_xMz_Dz, oc.empty,    oc.sigmaz   ]])
    
            C = spsp.bmat([[oc.m1*oc.sigma_xPz, oc.empty, oc.empty],
                           [oc.empty,          oc.I,     oc.empty],
                           [oc.empty,          oc.empty, oc.I    ]]) / self.dt
    
            M = spsp.bmat([[    oc.m1, oc.empty, oc.empty],
                           [oc.empty, oc.empty, oc.empty],
                           [oc.empty, oc.empty, oc.empty]]) / self.dt**2
    
            Stilde_inv = M+C
            Stilde_inv.data = 1./Stilde_inv.data
    
            self.A_k   = Stilde_inv*(2*M - K + C)
            self.A_km1 = -1*Stilde_inv*(M)
            self.A_f   = Stilde_inv
