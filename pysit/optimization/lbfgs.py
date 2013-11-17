from __future__ import absolute_import

import time
import copy
from collections import deque

import numpy as np

from pysit.optimization.optimization import OptimizationBase

__all__=['LBFGS']

__docformat__ = "restructuredtext en"

class LBFGS(OptimizationBase):

	def __init__(self, objective, memory_length=None, reset_on_new_inner_loop_call=True, *args, **kwargs):
		OptimizationBase.__init__(self, objective, *args, **kwargs)
		self.prev_alpha = None
		
		# collections.deque uses None to indicate no length
		self.memory_length=memory_length		
		self.reset_on_new_inner_loop_call = reset_on_new_inner_loop_call
		
		self._reset_memory()
		
	def _reset_memory(self):
		self.memory = deque([], maxlen=self.memory_length)
		self._reset_line_search = True


	def inner_loop(self, *args, **kwargs):
		
		if self.reset_on_new_inner_loop_call:
			self._reset_memory()
			
		OptimizationBase.inner_loop(self, *args, **kwargs)

	def _select_step(self, shots, current_objective_value, gradient, iteration, objective_arguments, **kwargs):
		"""Compute the LBFGS update for a set of shots.
		
		Gives the step s as a function of the gradient vector.  Implemented as in p178 of Nocedal and Wright.
		
		Parameters
		----------
		shots : list of pysit.Shot
			List of Shots for which to compute.
		grad : ndarray
			Gradient vector.
		i : int
			Current time index.
			
		"""
		
		mem = self.memory
		
		q = copy.deepcopy(gradient)
		
		# fix the 'y' variable, since we saved the previous gradient and just 
		# computed the next gradient.  That is, assume we are at k+1.  the right 
		# side of memory is k, but we didn't compute y_k yet.  in the y_k slot we 
		# stored a copy of the (negative) gradient.  thus y_k is that negative 
		# gradient plus the current gradient.
		if len(mem) > 0:
			mem[-1][2] += gradient # y
			mem[-1][0]  = 1./mem[-1][2].inner_product(mem[-1][1]) # rho
			gamma = mem[-1][1].inner_product(mem[-1][2]) / mem[-1][2].inner_product(mem[-1][2])
		else:
			gamma = -1.0
				
		alphas = []
		
		for rho, s, y in reversed(mem):
			alpha = rho * s.inner_product(q)
			t= alpha * y
			q -= t
			alphas.append(alpha)
		
		alphas.reverse()
		
		r = gamma * q
		
		for alpha, m in zip(alphas, mem):
			rho, s, y = m
			beta = rho*y.inner_product(r)
			r += (alpha-beta)*s
		
		# Search the opposite direction
		direction = r
		
		alpha0_kwargs = {'reset' : False}
		if self._reset_line_search:
			alpha0_kwargs = {'reset' : True}
			self._reset_line_search = False
		
		alpha = self.select_alpha(shots, gradient, direction, objective_arguments, 
		                          current_objective_value=current_objective_value, 
		                          alpha0_kwargs=alpha0_kwargs, **kwargs)
		
		self._print('  alpha {0}'.format(alpha))
		self.store_history('alpha', iteration, alpha)
		
		step = alpha * direction		
		
		# these copy calls might be removable
		self.memory.append([None,copy.deepcopy(-1*step),copy.deepcopy(-1*gradient)])
		
		return step
		
		
	def _compute_alpha0(self, phi0, grad0, reset=False, *args, **kwargs):
		if reset:
			return phi0 / (grad0.norm()*np.prod(self.solver.mesh.deltas))**2
		else:
			return 1.0