import numpy as np

from constant_density_acoustic_frequency_base import *
from pysit.solvers.solver_data import SolverDataFrequencyBase

__all__=['ConstantDensityAcousticFrequencyScalarBase']

__docformat__ = "restructuredtext en"
		
class ConstantDensityAcousticFrequencyScalarBase(ConstantDensityAcousticFrequencyBase):
	
	_SolverData = SolverDataFrequencyBase
	
	def __init__(self, mesh, **kwargs):
		
		
		self.M = None
		self.C = None
		self.K = None
		
		ConstantDensityAcousticFrequencyBase.__init__(self, mesh, **kwargs)
	
	def solve(self, solver_data, rhs, nu, *args, **kwargs):
		
		if type(rhs) is self.WavefieldVector:
			_rhs = rhs.data.reshape(-1)
		else:
			_rhs = rhs.reshape(-1)
		
		u = self.solvers[nu](_rhs)
		u.shape = solver_data.k.data.shape
		
		solver_data.k.data = u
		
	def build_rhs(self, fhat, rhs_wavefieldvector=None):
		
		
		if rhs_wavefieldvector is None:
			rhs_wavefieldvector = self.WavefieldVector(self.mesh, dtype=self.dtype)
		elif type(rhs_wavefieldvector) is not self.WavefieldVector:
			raise TypeError('Input rhs array must be a WavefieldVector.')
		else:
			rhs_wavefieldvector.data *= 0
		rhs_wavefieldvector.u = fhat
		
		return rhs_wavefieldvector
		
	def _build_helmholtz_operator(self, nu):
		omega = 2*np.pi*nu
		return (-(omega**2)*self.M + omega*1j*self.C + self.K).tocsc() # csc is used for the sparse solvers right now
	