from __future__ import absolute_import

import time
import copy

import numpy as np
from scipy.sparse.linalg import LinearOperator
from pyamg.krylov import cg, gmres

from pysit.optimization.optimization import OptimizationBase

__all__=['GaussNewton']

__docformat__ = "restructuredtext en"

class GaussNewton(OptimizationBase):

	def __init__(self, objective, krylov_maxiter=50, *args, **kwargs):
		OptimizationBase.__init__(self, objective, *args, **kwargs)
		
		self.krylov_maxiter = krylov_maxiter


	def _select_step(self, shots, current_objective_value, gradient, iteration, objective_arguments, **kwargs):
		"""Compute the Gauss-Newton update for a set of shots.
		
		Gives the step s as a function of the gradient vector.  Implemented as in p178 of Nocedal and Wright.
		
		Parameters
		----------
		shots : list of pysit.Shot
			List of Shots for which to compute.
		grad : ndarray
			Gradient vector.
		iteration : int
			Current time index.
			
		"""
		
		m0 = self.base_model
		
		rhs = -1*gradient.asarray()
		
		def matvec(x):
			m1 = m0.perturbation(data=x)
			return self.objective_function.apply_hessian(shots, self.base_model, x, **objective_arguments).data
			
		A_shape = (len(rhs), len(rhs))

		A = LinearOperator(shape=A_shape, matvec=matvec, dtype=gradient.dtype)
		
		resid = []
		
#		d, info = cg(A, rhs, maxiter=self.krylov_maxiter, residuals=resid)
		d, info = gmres(A, rhs, maxiter=self.krylov_maxiter, residuals=resid)
		
		d.shape = rhs.shape
		
		direction = m0.perturbation(data=d)
		
		if info < 0:
			print "CG Failure"
		if info == 0:
			print "CG Converge"
		if info > 0:
			print "CG ran {0} iterations".format(info)
		
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
		
		return 1.0