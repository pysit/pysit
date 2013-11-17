import numpy as np
import scipy.sparse as spsp

from pysit.solvers.wavefield_vector import *
from constant_density_acoustic_time_ode_base import *

from pysit.util import Bunch
from pysit.util.derivatives import build_derivative_matrix
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

__all__=['ConstantDensityAcousticTimeODE_2D']

__docformat__ = "restructuredtext en"

class ConstantDensityAcousticTimeODE_2D(ConstantDensityAcousticTimeODEBase):
	
	def __init__(self, mesh, **kwargs):
		
		self.operator_components = Bunch()
		
		ConstantDensityAcousticTimeODEBase.__init__(self, mesh, **kwargs)
		
	def _rebuild_operators(self):
		
		dof = self.mesh.dof(include_bc=True)
		
		# a more readable reference
		oc = self.operator_components
		# Check if empty.  If empty, build the static components
		if not self.operator_components: 
			# build laplacian
			oc.L = build_derivative_matrix(self.mesh, 2, self.spatial_accuracy_order, use_shifted_differences=self.spatial_shifted_differences)
			
			# build sigmax
			sx = build_sigma(self.mesh, self.mesh.x)
			oc.minus_sigmax = make_diag_mtx(-sx)
			
			# build sigmaz
			sz = build_sigma(self.mesh, self.mesh.z)
			oc.minus_sigmaz = make_diag_mtx(-sz)
			
			# build Dx
			oc.Dx = build_derivative_matrix(self.mesh, 1, self.spatial_accuracy_order, dimension='x', use_shifted_differences=self.spatial_shifted_differences)
			
			# build Dz
			oc.Dz = build_derivative_matrix(self.mesh, 1, self.spatial_accuracy_order, dimension='z', use_shifted_differences=self.spatial_shifted_differences)
			
			# build other useful things
			oc.I     = spsp.eye(dof,dof)
			oc.empty = spsp.csr_matrix((dof,dof))
			
			# useful intermediates
			oc.sigma_xz  = make_diag_mtx(sx*sz)
			oc.minus_sigma_xPz = oc.minus_sigmax + oc.minus_sigmaz
			oc.sigma_zMx_Dx = make_diag_mtx(sz-sx)*oc.Dx
			oc.sigma_xMz_Dz = make_diag_mtx(sx-sz)*oc.Dz
			
		C = self.model_parameters.C
		oc.m_inv = make_diag_mtx((C**2).reshape(-1,))
		
		self.A = spsp.bmat([[oc.empty,                  oc.I,               oc.empty,        oc.empty        ],
		                    [oc.m_inv*oc.L-oc.sigma_xz, oc.minus_sigma_xPz, oc.m_inv*oc.Dx,  oc.m_inv*oc.Dz  ],
		                    [oc.sigma_zMx_Dx,           oc.empty,           oc.minus_sigmax, oc.empty        ],
		                    [oc.sigma_xMz_Dz,           oc.empty,           oc.empty,        oc.minus_sigmaz ]])
		
	class WavefieldVector(WavefieldVectorBase):

		aux_names = ['v', 'Phix', 'Phiz']