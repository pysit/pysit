import numpy as np

from pysit.solvers.solver_base import *
from pysit.solvers.model_parameter import *

__all__=['ConstantDensityAcousticBase']

__docformat__ = "restructuredtext en"
		
class ConstantDensityAcousticBase(SolverBase): 
	""" Base class for solvers that use the Constant Density Acoustic Model
	(e.g., in the wave, helmholtz, and laplace domains).
	
	"""
	
	ModelParameters = ConstantDensityAcousticParameters
	
	def _compute_dWaveOp_time(self, solver_data): 		
		ukm1 = solver_data.km1.primary_wavefield
		uk   = solver_data.k.primary_wavefield
		ukp1 = solver_data.kp1.primary_wavefield
		return (ukp1-2*uk+ukm1)/(self.dt**2)
		
	def _compute_dWaveOp_frequency(self, uk_hat, nu):
		omega2 = (2*np.pi*nu)**2
		return -1*omega2*uk_hat
		
	def _compute_dWaveOp_laplace(self, *args):	
		raise NotImplementedError('Derivative Laplace domain operator not yet implemented.')
	
	