import numpy as np

from ..constant_density_acoustic_time_base import *
from pysit.solvers.solver_data import SolverDataTimeBase

__all__=['ConstantDensityAcousticTimeODEBase']

__docformat__ = "restructuredtext en"

multistep_coeffs = { 1: [],
	                 2: [],
	                 3: [],
	                 4: [],
}


class _ConstantDensityAcousticTimeODE_SolverData(SolverDataTimeBase):
	
	def __init__(self, solver, time_accuracy_order, integrator, **kwargs):
		
		self.solver = solver
		
		self.time_accuracy_order = time_accuracy_order
		
		# self.us[0] is kp1, [1] is k or current, [2] is km1, [3] is km2, etc
		self.us = [solver.WavefieldVector(solver.mesh, dtype=solver.dtype) for x in xrange(3)]
		
		if integrator == 'multistep':
			self.u_primes = [solver.WavefieldVector(solver.mesh, dtype=solver.dtype) for x in xrange(time_accuracy_order)]
		else:
			self.u_primes = list()
			
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
		
	def advance(self):
		
		self.us[-1] *= 0
		self.us.insert(0, self.us.pop(-1))
		
		if len(self.u_primes) > 0:
			self.u_primes[-1] *= 0
			self.u_primes.insert(0, self.us.pop(-1))
		
		
class ConstantDensityAcousticTimeODEBase(ConstantDensityAcousticTimeBase):
	
	cpp_accelerated = False
	
	def __init__(self, mesh,  
	                   integrator='rk', # 'multistep'
	                   **kwargs):
		
		self.integrator = integrator
		self.A = None
		
		ConstantDensityAcousticTimeBase.__init__(self, mesh, **kwargs)

					
	def time_step(self, solver_data, rhs_k, rhs_kp1):
		
		if self.integrator == 'rk':
			if self.time_accuracy_order == 4:
				self._rk4(solver_data, rhs_k, rhs_kp1)
			else:
				self._rk2(solver_data, rhs_k, rhs_kp1)
	
	def _ode_rhs(self, u_bar, f):
		data=self.A*u_bar.data
		step = self.WavefieldVector(u_bar.mesh, dtype=u_bar.dtype, data=data)
		step.v += self.operator_components.m_inv*f
		return step
	
	def _rk4(self, solver_data, rhs_k, rhs_kp1):
		u_bar = solver_data.k
		
		k = self.dt * self._ode_rhs(u_bar, rhs_k)
		solver_data.kp1 += u_bar + (1./6)*k
		
		rhs_half = 0.5*(rhs_k + rhs_kp1)
		
		k = self.dt * self._ode_rhs(u_bar + 0.5*k, rhs_half)
		solver_data.kp1 += (1./3)*k
		
		k = self.dt * self._ode_rhs(u_bar + 0.5*k, rhs_half)
		solver_data.kp1 += (1./3)*k
		
		k = self.dt * self._ode_rhs(u_bar + k, rhs_kp1)
		solver_data.kp1 += (1./6)*k
	
	def _rk2(self, solver_data, rhs_k, rhs_kp1):
		u_bar = solver_data.k
		
		k1 = self.dt*self._ode_rhs(u_bar, rhs_k)
		k2 = self.dt*self._ode_rhs(u_bar+k1, rhs_kp1)
		solver_data.kp1 += u_bar + 0.5*(k1+k2)
		
	_SolverData = _ConstantDensityAcousticTimeODE_SolverData
	
	def SolverData(self, *args, **kwargs):
		return self._SolverData(self, self.time_accuracy_order, self.integrator, **kwargs)