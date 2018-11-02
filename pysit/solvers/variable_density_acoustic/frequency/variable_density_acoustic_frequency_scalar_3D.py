import scipy.sparse as spsp
import numpy as np

from pysit.solvers.wavefield_vector import *
from .variable_density_acoustic_frequency_scalar_base import *

from pysit.util import Bunch
from pysit.util import PositiveEvenIntegers
from pysit.util.derivatives import build_derivative_matrix
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

from pysit.util.solvers import inherit_dict

__all__ = ['VariableDensityAcousticFrequencyScalar_3D']

__docformat__ = "restructuredtext en"

### Currently not Implemented.

@inherit_dict('supports', '_local_support_spec')
class VariableDensityAcousticFrequencyScalar_3D(VariableDensityAcousticFrequencyScalarBase):

    _local_support_spec = {'spatial_discretization': 'finite-difference',
                           'spatial_dimension': 3,
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

        # In 3D there is only compact operator implemented
        if self.mesh.x.lbc.type == 'pml':
          if self.mesh.x.lbc.domain_bc.compact:
            self.compact = self.mesh.x.lbc.domain_bc.compact
          else:
            raise NotImplementedError('3D frequency helmholtz operator with auxilliary fields is not available please use the flag compact=True for setting')
        else:
          self.compact = False
          raise NotImplementedError('Please set boundary conditions as PML with the flag compact=True')

    def _sigma_PML(self, mesh):
        nx, ny, nz = mesh.shape(include_bc=True, as_grid=True)
        # in mesh.py spacial direction seems to be ordered this way
        npml_x_l = mesh.x.lbc.n
        npml_x_r = mesh.x.rbc.n
        npml_z_l = mesh.z.lbc.n
        npml_z_r = mesh.z.rbc.n
        npml_y_l = mesh.y.lbc.n
        npml_y_r = mesh.y.rbc.n
        
        t_x_l = np.linspace(1, 0, npml_x_l)
        t_x_r = np.linspace(0, 1, npml_x_r)
        t_z_l = np.linspace(1, 0, npml_z_l)
        t_z_r = np.linspace(0, 1, npml_z_r)
        t_y_l = np.linspace(1, 0, npml_y_l)
        t_y_r = np.linspace(0, 1, npml_y_r)
        
        amplitude_x_l = mesh.x.lbc.domain_bc.amplitude
        amplitude_x_r = mesh.x.rbc.domain_bc.amplitude
        amplitude_z_l = mesh.z.lbc.domain_bc.amplitude
        amplitude_z_r = mesh.z.rbc.domain_bc.amplitude
        amplitude_y_l = mesh.y.lbc.domain_bc.amplitude
        amplitude_y_r = mesh.y.rbc.domain_bc.amplitude

        sx = np.zeros(nx*ny*nz)
        sy = np.zeros(nx*ny*nz)
        sz = np.zeros(nx*ny*nz)
        sxp = np.zeros(nx*ny*nz)
        syp = np.zeros(nx*ny*nz)
        szp = np.zeros(nx*ny*nz)

        # PML for the x direction
        for i in range(ny*nz):
          for j in range(npml_x_l):
            sx [i + j*nz] = amplitude_x_l * t_x_l[j]**2 
            sxp[i + j*nz] = -2 * amplitude_x_l * t_x_l[j]

        for i in range(ny*nz):
          for j in range(npml_x_r):
            sx [i + (j + (nx - npml_x_r))*nz] = amplitude_x_r * t_x_r[j]**2
            sxp[i + (j + (nx - npml_x_r))*nz] = 2 * amplitude_x_r * t_x_r[j]

        # PML for the y direction
        for i in range(nz):
          for j in range(nx):
            for k in range(npml_y_l):
              sy [i*nx*ny + j + k*nx] = amplitude_y_l * t_y_l[k]**2
              syp[i*nx*ny + j + k*nx] = -2 * amplitude_y_l * t_y_l[k]

        for i in range(nz):
          for j in range(nx):
            for k in range(npml_y_r):
              sy [i*nx*ny + j + (k + (ny - npml_y_r))*nx] = amplitude_y_r * t_y_r[k]**2
              syp[i*nx*ny + j + (k + (ny - npml_y_r))*nx] = 2 * amplitude_y_r * t_y_r[k]

        # PML for the z direction
        for i in range(nx*ny):
          for j in range(npml_z_l):
            sz [i*nz + j] = amplitude_z_l * t_z_l[j]**2
            szp[i*nz + j] = -2 * amplitude_z_l * t_z_l[j]

        for i in range(nx*ny):
          for j in range(npml_z_r):
            sz [i*nz + (nz - npml_z_r) + j] = amplitude_z_r * t_z_r[j]**2
            szp[i*nz + (nz - npml_z_r) + j] = 2 * amplitude_z_r * t_z_r[j]

        return (sx, sy, sz, sxp, syp, szp)

    def _rebuild_operators(self):
        if self.mesh.x.lbc.type == 'pml' and self.compact:
          # build intermediates for compact operator 
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
            # build Dyy
            oc.Dyy = build_derivative_matrix(self.mesh,
                                                  2,
                                                  self.spatial_accuracy_order,
                                                  dimension='y',
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
            # build Dz
            oc.Dy = build_derivative_matrix(self.mesh,
                                                  1,
                                                  self.spatial_accuracy_order,
                                                  dimension='y',
                                                  use_shifted_differences=self.spatial_shifted_differences)
                                                  
            # build sigma
            oc.sx, oc.sy, oc.sz, oc.sxp, oc.syp, oc.szp = self._sigma_PML(self.mesh)

            oc._numpy_components_built = True

        else :
          # build intermediates for operator with auxiliary fields
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
                                                    dimension='x',
                                                    use_shifted_differences=self.spatial_shifted_differences)
              oc.minus_Dx.data *= -1

              # build Dy
              oc.minus_Dy = build_derivative_matrix(self.mesh,
                                                    1,
                                                    self.spatial_accuracy_order,
                                                    dimension='y',
                                                    use_shifted_differences=self.spatial_shifted_differences)
              oc.minus_Dy.data *= -1

              # build Dz
              oc.minus_Dz = build_derivative_matrix(self.mesh,
                                                    1,
                                                    self.spatial_accuracy_order,
                                                    dimension='z',
                                                    use_shifted_differences=self.spatial_shifted_differences)
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

          self.K = spsp.bmat([[oc.m*oc.sigma_sum_pair_prod-oc.L, oc.m*oc.sigma_prod,   oc.minus_Dx, oc.minus_Dy, oc.minus_Dz ],
                              [oc.minus_I,                       oc.empty,             oc.empty,    oc.empty,    oc.empty    ],
                              [oc.minus_sigma_yPzMx_Dx,          oc.minus_sigma_yz_Dx, oc.sigmax,   oc.empty,    oc.empty    ],
                              [oc.minus_sigma_xPzMy_Dy,          oc.minus_sigma_zx_Dy, oc.empty,    oc.sigmay,   oc.empty    ],
                              [oc.minus_sigma_xPyMz_Dz,          oc.minus_sigma_xy_Dz, oc.empty,    oc.empty,    oc.sigmaz   ]])

          self.C = spsp.bmat([[oc.m*oc.sigma_sum, oc.empty, oc.empty, oc.empty, oc.empty],
                              [oc.empty,          oc.I,     oc.empty, oc.empty, oc.empty],
                              [oc.empty,          oc.empty, oc.I,     oc.empty, oc.empty],
                             [oc.empty,          oc.empty, oc.empty, oc.I,     oc.empty],
                             [oc.empty,          oc.empty, oc.empty, oc.empty, oc.I    ]]) / self.dt

          self.M = spsp.bmat([[    oc.m, oc.empty, oc.empty, oc.empty, oc.empty],
                             [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty],
                              [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty],
                             [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty],
                             [oc.empty, oc.empty, oc.empty, oc.empty, oc.empty]])

    class WavefieldVector(WavefieldVectorBase):

        aux_names = ['psi', 'Phix', 'Phiy', 'Phiz']
