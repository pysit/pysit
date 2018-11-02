import scipy.sparse as spsp
import numpy as np

from pysit.solvers.wavefield_vector import *
from .variable_density_acoustic_frequency_scalar_base import *

from pysit.util import Bunch
from pysit.util import PositiveEvenIntegers
from pysit.util.derivatives import build_derivative_matrix, build_derivative_matrix_VDA
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

from pysit.util.solvers import inherit_dict

__all__ = ['VariableDensityAcousticFrequencyScalar_2D']

__docformat__ = "restructuredtext en"


@inherit_dict('supports', '_local_support_spec')
class VariableDensityAcousticFrequencyScalar_2D(VariableDensityAcousticFrequencyScalarBase):

    _local_support_spec = {'spatial_discretization': 'finite-difference',
                           'spatial_dimension': 2,
                           'spatial_accuracy_order': PositiveEvenIntegers,
                           'boundary_conditions': ['pml', 'pml-sim', 'dirichlet'],
                           'precision': ['single', 'double']}

    def __init__(self,
                 mesh,
                 spatial_accuracy_order=4,
                 spatial_shifted_differences=True,
                 **kwargs):

        self.operator_components = Bunch()

        self.spatial_accuracy_order = spatial_accuracy_order

        self.spatial_shifted_differences = spatial_shifted_differences

        VariableDensityAcousticFrequencyScalarBase.__init__(self,
                                                            mesh,
                                                            spatial_accuracy_order=spatial_accuracy_order,
                                                            spatial_shifted_differences=spatial_shifted_differences,
                                                            **kwargs)
        # if the solver is compact we have to indicate that we do not need auxiliary fields
        if self.mesh.x.lbc.type == 'pml':
          self.compact = self.mesh.x.lbc.domain_bc.compact
        else:
          self.compact = False

    # Compact PML for Helmholtz operator 
    def _sigma_PML(self, mesh):
        nx, nz = mesh.shape(include_bc=True, as_grid=True)
        oc = self.operator_components
        oc.nx = nx
        oc.nz = nz

        sx  = np.zeros(nz*nx)
        sz  = np.zeros(nz*nx)
        sxp = np.zeros(nz*nx)
        szp = np.zeros(nz*nx)
        
        npml_x_l = mesh.x.lbc.n
        npml_x_r = mesh.x.rbc.n
        npml_z_l = mesh.z.lbc.n
        npml_z_r = mesh.z.rbc.n
        
        t_x_l = np.linspace(1, 0, npml_x_l)
        t_x_r = np.linspace(0, 1, npml_x_r)
        t_z_l = np.linspace(1, 0, npml_z_l)
        t_z_r = np.linspace(0, 1, npml_z_r)
        
        amplitude_x_l = mesh.x.lbc.domain_bc.amplitude
        amplitude_x_r = mesh.x.rbc.domain_bc.amplitude
        
        amplitude_z_l = mesh.z.lbc.domain_bc.amplitude
        amplitude_z_r = mesh.z.rbc.domain_bc.amplitude
        
        # PML for the x direction
        # left side
        for i in range(nz):
          for j in range(npml_x_l):
            sx [i + j*nz] = amplitude_x_l * t_x_l[j]**2
            sxp[i + j*nz] = -2 * amplitude_x_l * t_x_l[j]
        # right side
        for i in range(nz):
          for j in range(npml_x_r):
            sx [i + (j + (nx - npml_x_r))*nz] = amplitude_x_r * t_x_r[j]**2
            sxp[i + (j + (nx - npml_x_r))*nz] = 2 * amplitude_x_r * t_x_r[j] 
        
        # PML for the z direction
        # left side
        for i in range(nx):
          for j in range(npml_z_l):
            sz [i*nz + j] = amplitude_z_l * t_z_l[j]**2 
            szp[i*nz + j] = -2 * amplitude_z_l * t_z_l[j]

        # rigth side
        for i in range(nx):
          for j in range(npml_z_r):
            sz [i*nz + (nz - npml_z_r) + j] = amplitude_z_r * t_z_r[j]**2
            szp[i*nz + (nz - npml_z_r) + j] = 2 * amplitude_z_r * t_z_r[j]

        return (sx, sz, sxp, szp)

    def _rebuild_operators(self):
        if self.mesh.x.lbc.type == 'pml' and self.compact:
          # build intermediates for the compact operator
          raise ValidationFunctionError(" This solver is in construction and is not yet complete. For variable density and compact PML.")
          dof = self.mesh.dof(include_bc=True)

          oc = self.operator_components

          built = oc.get('_numpy_components_built', False)
          oc.M = make_diag_mtx(self.model_parameters.C.squeeze()**-2)
          # build the static components
          if not built:          
            # build Dxx
            oc.Dxx = build_derivative_matrix(self.mesh,
                                                  2,
                                                  self.spatial_accuracy_order,
                                                  dimension='x',
                                                  use_shifted_differences=self.spatial_shifted_differences)
            # build Dzz
            oc.Dzz = build_derivative_matrix(self.mesh,
                                                  2,
                                                  self.spatial_accuracy_order,
                                                  dimension='z',
                                                  use_shifted_differences=self.spatial_shifted_differences)
            # build Dx
            oc.Dx = build_derivative_matrix(self.mesh,
                                                  1,
                                                  self.spatial_accuracy_order,
                                                  dimension='x',
                                                  use_shifted_differences=self.spatial_shifted_differences)
            
            # build Dz
            oc.Dz = build_derivative_matrix(self.mesh,
                                                  1,
                                                  self.spatial_accuracy_order,
                                                  dimension='z',
                                                  use_shifted_differences=self.spatial_shifted_differences)
                                                  
            # build sigma
            oc.sx, oc.sz, oc.sxp, oc.szp = self._sigma_PML(self.mesh)

            oc._numpy_components_built = True
        else:
          # build intermediates for operator with auxiliary fields
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

        kappa = self.model_parameters.kappa
        rho = self.model_parameters.rho
        oc.m1 = make_diag_mtx((kappa**-1).reshape(-1,))
        oc.m2 = make_diag_mtx((rho**-1).reshape(-1,))
        # build heterogenous laplacian
        sh = self.mesh.shape(include_bc=True,as_grid=True)
        deltas = [self.mesh.x.delta,self.mesh.z.delta]
        oc.L = build_derivative_matrix_VDA(self.mesh,
                                           2,
                                           self.spatial_accuracy_order,
                                           alpha = rho**-1
                                           )

        # oc.L is a heterogenous laplacian operator. It computes div(m2 grad), where m2 = 1/rho. 
        # Currently the creation of oc.L is slow. This is because we have implemented a cenetered heterogenous laplacian.
        # To speed up computation, we could compute a div(m2 grad) operator that is not centered by simply multiplying
        # a divergence operator by oc.m2 by a gradient operator.

        self.K = spsp.bmat([[oc.m1*oc.sigma_xz-oc.L, oc.minus_Dx*oc.m2, oc.minus_Dz*oc.m2 ],
                            [oc.minus_sigma_zMx_Dx, oc.sigmax,   oc.empty    ],
                            [oc.minus_sigma_xMz_Dz, oc.empty,    oc.sigmaz   ]])

        self.C = spsp.bmat([[oc.m1*oc.sigma_xPz, oc.empty, oc.empty],
                            [oc.empty,          oc.I,     oc.empty],
                            [oc.empty,          oc.empty, oc.I    ]])

        self.M = spsp.bmat([[    oc.m1, oc.empty, oc.empty],
                            [oc.empty, oc.empty, oc.empty],
                            [oc.empty, oc.empty, oc.empty]])

    class WavefieldVector(WavefieldVectorBase):

        aux_names = ['Phix', 'Phiz']
