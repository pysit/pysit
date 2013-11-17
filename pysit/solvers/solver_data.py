import numpy as np

__all__ = ['SolverDataBase', 'SolverDataTimeBase', 'SolverDataFrequencyBase']

class SolverDataBase(object):
	def __init__(self, solver, **kwargs):
		raise NotImplementedError('An specification of SolverData must be made at the solver level.')
	
class SolverDataTimeBase(SolverDataBase):
	def __init__(self, solver, **kwargs):
		
		self.solver = solver
		
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
		
		
	
class SolverDataFrequencyBase(SolverDataBase):
	def __init__(self, solver, **kwargs):
		
		self.solver = solver
		
		self.ubar = solver.WavefieldVector(solver.mesh, dtype=solver.dtype)
		
	@property
	def k(self):
		return self.ubar
	@k.setter
	def k(self, arg):
		self.ubar = arg