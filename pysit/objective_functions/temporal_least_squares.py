from __future__ import absolute_import

import numpy as np

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.temporal_modeling import TemporalModeling

__all__ = ['TemporalLeastSquares']

__docformat__ = "restructuredtext en"

class TemporalLeastSquares(ObjectiveFunctionBase):
	""" How to compute the parts of the objective you need to do optimization """
	
	def __init__(self, solver, parallel_wrap_shot=ParallelWrapShotNull()):
		self.solver = solver
		self.modeling_tools = TemporalModeling(solver)
		
		self.parallel_wrap_shot = parallel_wrap_shot
	
	def _residual(self, shot, m0, dWaveOp=None):
		"""Computes residual in the usual sense.
		
		Parameters
		----------
		shot : pysit.Shot
			Shot for which to compute the residual.
		dWaveOp : list of ndarray (optional)
			An empty list for returning the derivative term required for 
			computing the imaging condition.
			
		"""
	
		# If we will use the second derivative info later (and this is usually 
		# the case in inversion), tell the solver to store that information, in 
		# addition to the solution as it would be observed by the receivers in 
		# this shot (aka, the simdata).
		rp = ['simdata']
		if dWaveOp is not None:
			rp.append('dWaveOp')
	
		# Run the forward modeling step
		retval = self.modeling_tools.forward_model(shot, m0, return_parameters=rp)
			
		# Compute the residual vector by interpolating the measured data to the 
		# timesteps used in the previous forward modeling stage.
		# resid = map(lambda x,y: x.interpolate_data(self.solver.ts())-y, shot.gather(), retval['simdata'])
		resid = shot.receivers.interpolate_data(self.solver.ts()) - retval['simdata']
						
		# If the second derivative info is needed, copy it out
		if dWaveOp is not None:
			dWaveOp[:]  = retval['dWaveOp'][:]
				
		return resid
	
	def evaluate(self, shots, m0, **kwargs):
		""" Evaluate the least squares objective function over a list of shots."""
		
		r_norm2 = 0
		for shot in shots:
			r = self._residual(shot, m0)
			r_norm2 += np.linalg.norm(r)**2
		
		# sum-reduce and communicate result
		if self.parallel_wrap_shot.use_parallel:
			# Allreduce wants an array, so we give it a 0-D array
			new_r_norm2 = np.array(0.0) 
			self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
			r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element
			
		return 0.5*r_norm2*self.solver.dt

	def _gradient_helper(self, shot, m0, ignore_minus=False, **kwargs):
		"""Helper function for computing the component of the gradient due to a 
		single shot.
		
		Computes F*_s(d - scriptF_s[u]), in our notation.
		
		Parameters
		----------
		shot : pysit.Shot
			Shot for which to compute the residual.
			
		"""
		
		# Compute the residual vector and its norm
		dWaveOp=[]
		r = self._residual(shot, m0, dWaveOp=dWaveOp, **kwargs)
		
		# Perform the migration or F* operation to get the gradient component
		g = self.modeling_tools.migrate_shot(shot, m0, r, dWaveOp=dWaveOp)
		
		if ignore_minus:
			return g, r
		else:
			return -1*g, r
	
	def compute_gradient(self, shots, m0, aux_info={}, **kwargs):
		"""Compute the gradient for a set of shots.
		
		Computes the gradient as
			-F*(d - scriptF[m0]) = -sum(F*_s(d - scriptF_s[m0])) for s in shots
		
		Parameters
		----------
		shots : list of pysit.Shot
			List of Shots for which to compute the gradient.
		m0 : ModelParameters
			The base point about which to compute the gradient
			
		"""
		
		# compute the portion of the gradient due to each shot
		grad = m0.perturbation()
		r_norm2 = 0.0
		for shot in shots:
			g, r = self._gradient_helper(shot, m0, ignore_minus=True, **kwargs)
			grad -= g # handle the minus 1 in the definition of the gradient of this objective
			r_norm2 += np.linalg.norm(r)**2
					
		# sum-reduce and communicate result
		if self.parallel_wrap_shot.use_parallel:
			# Allreduce wants an array, so we give it a 0-D array
			new_r_norm2 = np.array(0.0) 			
			self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
			r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element
			
			ngrad = np.zeros_like(grad.asarray())
			self.parallel_wrap_shot.comm.Allreduce(grad.asarray(), ngrad)
			grad=m0.perturbation(data=ngrad)
			
		# account for the measure in the integral over time
		r_norm2 *= self.solver.dt
		
		# store any auxiliary info that is requested
		if ('residual_norm' in aux_info) and aux_info['residual_norm'][0]:
			aux_info['residual_norm'] = (True, np.sqrt(r_norm2))
		if ('objective_value' in aux_info) and aux_info['objective_value'][0]:
			aux_info['objective_value'] = (True, 0.5*r_norm2)
		
		return grad
		
	def apply_hessian(self, shots, m0, m1, hessian_mode='approximate', levenberg_mu=0.0, *args, **kwargs):
		
		modes = ['approximate', 'full', 'levenberg']
		if hessian_mode not in modes:
			raise ValueError("Invalid Hessian mode.  Valid options for applying hessian are {0}".format(modes))	
			
		result = m0.perturbation()
		
		if hessian_mode in ['approximate', 'levenberg']:
			for shot in shots:
				# Run the forward modeling step
				retval = self.modeling_tools.forward_model(shot, m0, return_parameters=['dWaveOp'])
				dWaveOp0 = retval['dWaveOp']

				linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, return_parameters=['simdata'], dWaveOp0=dWaveOp0)
				
				d1 = linear_retval['simdata'] # data from F applied to m1
				result += self.modeling_tools.migrate_shot(shot, m0, d1, dWaveOp=dWaveOp0)
				
		elif hessian_mode == 'full':
			for shot in shots:
				# Run the forward modeling step
				dWaveOp0 = list() # wave operator derivative wrt model for u_0
				r0 = self._residual(shot, m0, dWaveOp=dWaveOp0, **kwargs)
				
				linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, return_parameters=['simdata', 'dWaveOp1'], dWaveOp0=dWaveOp0)
				d1 = linear_retval['simdata']
				dWaveOp1 = linear_retval['dWaveOp1']
				
				# <q, u1tt>, first adjointy bit
				dWaveOpAdj1=[]			
				res1 = self.modeling_tools.migrate_shot( shot, m0, r0, dWaveOp=dWaveOp1, dWaveOpAdj=dWaveOpAdj1)	
				result += res1
				
				# <p, u0tt>
				res2 = self.modeling_tools.migrate_shot(shot, m0, d1, operand_dWaveOpAdj=dWaveOpAdj1, operand_model=m1, dWaveOp=dWaveOp0)
				result += res2

		# sum-reduce and communicate result
		if self.parallel_wrap_shot.use_parallel:
			
			nresult = np.zeros_like(result.asarray())
			self.parallel_wrap_shot.comm.Allreduce(result.asarray(), nresult)
			result = m0.perturbation(data=nresult)
			
		# Note, AFTER the application has been done in parallel do this.
		if hessian_mode == 'levenberg':
			result += levenberg_mu*m1
		
		return result
	