import numpy as np
import scipy.sparse as spsp

from pysit.solvers.wavefield_vector import *
from constant_density_acoustic_frequency_scalar_base import *

from pysit.util import Bunch
from pysit.util.derivatives import build_derivative_matrix
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

__all__=['ConstantDensityAcousticFrequencyScalar_1D']

__docformat__ = "restructuredtext en"
		
class ConstantDensityAcousticFrequencyScalar_1D(ConstantDensityAcousticFrequencyScalarBase):
	
	def __init__(self, mesh, **kwargs):
		
		self.operator_components = Bunch()
		
		ConstantDensityAcousticFrequencyScalarBase.__init__(self, mesh, **kwargs)
	
	def _rebuild_operators(self):
		
		dof = self.mesh.dof(include_bc=True)
		
		oc = self.operator_components
		# Check if empty.  If empty, build the static components
		if not self.operator_components: 
			# build laplacian
			oc.L = build_derivative_matrix(self.mesh, 2, self.spatial_accuracy_order, use_shifted_differences=self.spatial_shifted_differences)
			
			# build sigmaz 
			sz = build_sigma(self.mesh, self.mesh.z)
			oc.sigmaz = make_diag_mtx(sz)
			
			# build Dz
			oc.Dz = build_derivative_matrix(self.mesh, 1, self.spatial_accuracy_order, dimension='z', use_shifted_differences=self.spatial_shifted_differences)
			
			# build other useful things
			oc.I     = spsp.eye(dof,dof)
			oc.empty = spsp.csr_matrix((dof,dof))
			
			# Stiffness matrix K doesn't change
			self.K = spsp.bmat([[           -oc.L,    -oc.Dz],
			                    [ oc.sigmaz*oc.Dz, oc.sigmaz]])
		
		C = self.model_parameters.C # m = self.model_parameters.M[0]
		oc.m = make_diag_mtx((C**-2).reshape(-1,))
		
		self.C = spsp.bmat([[oc.sigmaz*oc.m, None],
			                [None,           oc.I]])
		
		self.M = spsp.bmat([[oc.m,     None],
			                [None, oc.empty]])
				
	class WavefieldVector(WavefieldVectorBase):

		aux_names = ['Phiz']