import numpy as np

from ..constant_density_acoustic_time_base import *
from pysit.solvers.solver_data import SolverDataTimeBase

__all__=['ConstantDensityAcousticTimeScalarBase']

__docformat__ = "restructuredtext en"

class _ConstantDensityAcousticTimeScalar_SolverData(SolverDataTimeBase):
	
	def __init__(self, solver, time_accuracy_order, **kwargs):
		
		self.solver = solver
		
		self.time_accuracy_order = time_accuracy_order
		
		# self.us[0] is kp1, [1] is k or current, [2] is km1, [3] is km2, etc
		self.us = [solver.WavefieldVector(solver.mesh, dtype=solver.dtype) for x in xrange(3)]	
		
	def advance(self):
		
		self.us[-1] *= 0
		self.us.insert(0, self.us.pop(-1))
		
	@property
	def kp1(self):
		return self.us[0]
	@kp1.setter
	def kp1(self, arg):
		self.us[0] = arg
	@property
	def k(self):
		return self.us[1]
	@k.setter
	def k(self, arg):
		self.us[1] = arg
	@property
	def km1(self):
		return self.us[2]
	@km1.setter
	def km1(self, arg):
		self.us[2] = arg
		
		
class ConstantDensityAcousticTimeScalarBase(ConstantDensityAcousticTimeBase):
	
	def __init__(self, mesh, **kwargs):
		
		self.A_km1 = None
		self.A_k   = None
		self.A_f   = None
		
		ConstantDensityAcousticTimeBase.__init__(self, mesh, **kwargs)

	def time_step(self, solver_data, rhs_k, rhs_kp1):
		
		if self.use_cpp_acceleration and self.cpp_accelerated:
			self._time_step_accelerated(solver_data, rhs_k, rhs_kp1)
		else:
			u_km1 = solver_data.km1
			u_k   = solver_data.k
			u_kp1 = solver_data.kp1
			
			f_bar = self.WavefieldVector(self.mesh, dtype=self.dtype)
			f_bar.u += rhs_k
			
			u_kp1 += self.A_k*u_k.data + self.A_km1*u_km1.data + self.A_f*f_bar.data
	
	def _time_step_accelerated(self, solver_data, rhs_k, rhs_kp1):
		raise NotImplementedError('CPU Acceleration must be implemented at the subclass level.')
		
	_SolverData = _ConstantDensityAcousticTimeScalar_SolverData
	
	def SolverData(self, *args, **kwargs):
		return self._SolverData(self, self.time_accuracy_order, **kwargs)