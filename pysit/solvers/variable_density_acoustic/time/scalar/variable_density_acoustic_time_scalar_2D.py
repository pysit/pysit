import copy
import numpy as np
import scipy.sparse as spsp

from pysit.core import CartesianMesh, RectangularDomain
from pysit.solvers.wavefield_vector import *
from variable_density_acoustic_time_scalar_base import *

from pysit.util import Bunch
from pysit.util import PositiveEvenIntegers
from pysit.util.derivatives import build_derivative_matrix, build_heterogenous_laplacian
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
                # build homogeneous laplacian
                oc.L_hom = build_derivative_matrix(self.mesh,
                                                            2,
                                                            self.spatial_accuracy_order)
                
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

                #Without modification, d_rho_inv_dx and d_rho_inv_dz will be nonzero around the respective
                #outer PML boundaries because the differentiation coefficients don't add up to 1.0.
                #A constant density would result in nonzero derivative values around the PML outer boundary
                
                #The approach I try here is to extend the density outwards by a few pixels in each direction.
                #Then I can use a slightly larger differentiation operator associated with this extended mesh.
                #Now the rows in the matrix that add up to non-zero are in the added padded pixels.
                #Strip these and return the density derivatives within the original mesh.
                
                d = self.mesh.domain
                self.n_pad = self.spatial_accuracy_order/2 + 1
                dx = self.mesh.x.delta
                dz = self.mesh.z.delta
                nx = self.mesh.x.n
                nz = self.mesh.z.n
                
                extended_x_config = (d.x.lbound - self.n_pad*dx, d.x.rbound + self.n_pad*dx, copy.deepcopy(d.x.lbc), copy.deepcopy(d.x.rbc))
                extended_z_config = (d.z.lbound - self.n_pad*dz, d.z.rbound + self.n_pad*dz, copy.deepcopy(d.z.lbc), copy.deepcopy(d.z.rbc))
                
                extended_domain = RectangularDomain(extended_x_config, extended_z_config)
                
                self.extended_mesh = CartesianMesh(extended_domain, nx+2*self.n_pad, nz+2*self.n_pad)
                
                oc.Dx_modified_in_pml = build_derivative_matrix(self.extended_mesh,
                                                                        1,
                                                                        self.spatial_accuracy_order,
                                                                        dimension='x') 

                oc.Dz_modified_in_pml = build_derivative_matrix(self.extended_mesh,
                                                                        1,
                                                                        self.spatial_accuracy_order,
                                                                        dimension='z')

    
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
            
            #temporarily pad rho so we can do the derivative around the boundary
            sh = self.mesh.shape(include_bc = True, as_grid = True)
            rho_2d = np.reshape(rho, sh, 'F')
            rho_2d_padded = np.pad(rho_2d, [(self.n_pad,self.n_pad),(self.n_pad,self.n_pad)], mode='edge')
            rho_padded = np.reshape(rho_2d_padded, ( (sh[0] + 2*self.n_pad) * (sh[1] + 2*self.n_pad), 1 ), 'F')
    
            oc.m1 = make_diag_mtx((kappa**-1).reshape(-1,))
            oc.m2 = make_diag_mtx((rho**-1).reshape(-1,))

            #Take deriv on padded 
            d_rho_padded_inv_dx = oc.Dx_modified_in_pml*(rho_padded**-1)
            d_rho_padded_inv_dz = oc.Dz_modified_in_pml*(rho_padded**-1)

            #go to 2D so we can strip the padded values away
            d_rho_inv_dx_padded_2d = np.reshape(d_rho_padded_inv_dx, (sh[0]+2*self.n_pad, sh[1]+2*self.n_pad), 'F')
            d_rho_inv_dz_padded_2d = np.reshape(d_rho_padded_inv_dz, (sh[0]+2*self.n_pad, sh[1]+2*self.n_pad), 'F') 

            #remove padding
            d_rho_inv_dx_2d = d_rho_inv_dx_padded_2d[self.n_pad:self.n_pad + sh[0],self.n_pad:self.n_pad + sh[1]]
            d_rho_inv_dz_2d = d_rho_inv_dz_padded_2d[self.n_pad:self.n_pad + sh[0],self.n_pad:self.n_pad + sh[1]] 

            #back to 1D
            d_rho_inv_dx = np.reshape(d_rho_inv_dx_2d, (sh[0] * sh[1], 1 ), 'F')
            d_rho_inv_dz = np.reshape(d_rho_inv_dz_2d, (sh[0] * sh[1], 1 ), 'F')

            d_rho_inv_dx_as_diag_mat = make_diag_mtx(d_rho_inv_dx.flatten())
            d_rho_inv_dz_as_diag_mat = make_diag_mtx(d_rho_inv_dz.flatten())
            
            #The next operator components will compute the deriv of 
            #p(x) and multiply by the deriv of 1/rho in the same direction.
            #Then these directions are summed, so we compute the 
            #dot product between (grad(1/rho) and grad (p) at each pixel. 
            #The sum implements (grad 1/rho) dot grad(p)
            
            d_rho_inv_dx_times_d_dx = d_rho_inv_dx_as_diag_mat * oc.Dx #mat-vec
            d_rho_inv_dz_times_d_dz = d_rho_inv_dz_as_diag_mat * oc.Dz #mat-vec
            
            #oc.L applied to wavefield p: = (grad 1/rho) dot grad(p) + 1/rho*Laplacian(p)  
            oc.L = (d_rho_inv_dx_times_d_dx + d_rho_inv_dz_times_d_dz) + oc.m2*oc.L_hom
            
            # build heterogenous laplacian
            #sh = self.mesh.shape(include_bc=True,as_grid=True)
            #deltas = [self.mesh.x.delta,self.mesh.z.delta]
            #oc.L = build_heterogenous_laplacian(sh,rho**-1,deltas)
    
    
    
            # oc.L is a heterogenous laplacian operator. It computes div(m2 grad), where m2 = 1/rho. 
            # Currently the creation of oc.L is slow. This is because we have implemented a cenetered heterogenous laplacian.
            # To speed up computation, we could compute a div(m2 grad) operator that is not centered by simply multiplying
            # a divergence operator by oc.m2 by a gradient operator.
    
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
