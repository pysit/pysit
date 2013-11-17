from __future__ import absolute_import

import time
import copy

import numpy as np

from pysit.optimization.optimization import OptimizationBase

__all__=['ConjugateGradient']

__docformat__ = "restructuredtext en"

class ConjugateGradient(OptimizationBase):

	def __init__(self, objective, reset_length=None, beta_style='fletcher-reeves', *args, **kwargs):
		OptimizationBase.__init__(self, objective, *args, **kwargs)
		self.prev_alpha = None
		
		self.reset_length = reset_length
		
		self.prev_gradient = None
		self.prev_direction = None
		
		if beta_style not in ['fletcher-reeves', 'polak-ribiere']:
			raise ValueError('Invalid beta computation method.')
		self.beta_style = beta_style

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
		
		reset = (self.reset_length is not None) and (not np.mod(i, self.reset_length))
		
		if (self.prev_gradient is None) or reset:
			direction = -1*gradient
			self.prev_direction = direction
			self.prev_gradient = gradient
		else: #compute new search direction
			gkm1 = self.prev_gradient
			gk = gradient
			
			if self.beta_style == 'fletcher-reeves':
				beta = gk.inner_product(gk) / gkm1.inner_product(gkm1)
			elif self.beta_style == 'polak-ribiere':
				beta = (gk-gkm1).inner_product(gk) / gkm1.inner_product(gkm1)
			
			direction = -1*gk + beta*self.prev_direction
			
			self.prev_direction = direction
			self.prev_gradient = gk
		
		alpha0_kwargs = {'reset' : False}
		if iteration == 0:
			alpha0_kwargs = {'reset' : True}
				
		alpha = self.select_alpha(shots, gradient, direction, objective_arguments, 
		                          current_objective_value=current_objective_value, 
		                          alpha0_kwargs=alpha0_kwargs, **kwargs)
		
		self._print('  alpha {0}'.format(alpha))
		self.store_history('alpha', iteration, alpha)
		
		step = alpha * direction
		
		return step
		
	def _compute_alpha0(self, phi0, grad0, reset=False, upscale_factor=None, **kwargs):
		if reset or (self.prev_alpha is None):
			return phi0 / (grad0.norm()*np.prod(self.solver.domain.deltas))**2
		else:
			return self.prev_alpha / upscale_factor