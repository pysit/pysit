from __future__ import absolute_import

import itertools

import numpy as np

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.hybrid_modeling import HybridModeling

__all__ = ['HybridLeastSquares']

__docformat__ = "restructuredtext en"

class HybridLeastSquares(ObjectiveFunctionBase):
	
	def __init__(self, solver, parallel_wrap_shot=ParallelWrapShotNull(), modeling_kwargs={}):
		self.solver = solver
		self.modeling_tools = HybridModeling(solver, **modeling_kwargs)
		
		self.parallel_wrap_shot = parallel_wrap_shot

	def _residual(self, shot, m0, frequencies=None, dWaveOp=None, compute_resid_time=False):
		"""Computes residual in the usual sense.
		
		Parameters
		----------
		shot : pysit.Shot
			Shot for which to compute the residual.
		utt : list of ndarray (optional)
			An empty list for returning the derivative term required for 
			computing the imaging condition.
			
		"""
	
		if frequencies is None:
			raise ValueError('A set of frequencies must be specified.')
		
		# If we will use the second derivative info later (and this is usually 
		# the case in inversion), tell the solver to store that information, in 
		# addition to the solution as it would be observed by the receivers in 
		# this shot (aka, the simdata).
		rp = ['simdata']
		if dWaveOp is not None:
			rp.append('dWaveOp')
		if compute_resid_time:
			rp.append('simdata_time')
	
		# Run the forward modeling step
		retval = self.modeling_tools.forward_model(shot, m0, frequencies, return_parameters=rp)
			
		resid = dict()
		for nu in frequencies:
			resid[nu] = shot.receivers.data_dft[nu] - retval['simdata'][nu]
						
		# If the second derivative info is needed, copy it out
		if dWaveOp is not None:
			for nu in frequencies:
				dWaveOp[nu]  = retval['dWaveOp'][nu]
				
		if compute_resid_time:
			resid_time = shot.receivers.interpolate_data(self.solver.ts()) - retval['simdata_time']
			return resid, resid_time
		else:
			return resid
	
	def evaluate(self, shots, m0, frequencies=None, frequency_weights=None, **kwargs):
		""" Evaluate the least squares objective function over a list of shots."""
	
		if frequencies is None:
			raise ValueError('A set of frequencies must be specified.')
		
		if frequency_weights is not None and (len(frequencies) != len(frequency_weights)):
			raise ValueError('Weights and frequencies must be the same length.')
			
		if frequency_weights is None:
			frequency_weights = itertools.repeat(1.0)
		
		r_norm2 = 0
		for shot in shots:
			# ensure that the dft of the data exists
			shot.receivers.compute_data_dft(frequencies)
			r = self._residual(shot, m0, frequencies)
			for nu,weight in zip(frequencies,frequency_weights):
				r_norm2 += weight*np.linalg.norm(r[nu])**2
		
		# sum-reduce and communicate result
		if self.parallel_wrap_shot.use_parallel:
			# Allreduce wants an array, so we give it a 0-D array
			new_r_norm2 = np.array(0.0) 
			self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
			r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element
		
		return 0.5*r_norm2 # *d_omega which does not exist for this problem
		
	def _gradient_helper(self, shot, m0, frequencies, frequency_weights, ignore_minus=False, **kwargs):
		"""Helper function for computing the component of the gradient due to a 
		single shot.
		
		Computes F*_s(d - scriptF_s[u]), in our notation.
		
		Parameters
		----------
		shot : pysit.Shot
			Shot for which to compute the residual.
			
		"""
		
		# Compute the residual vector and its norm
		dWaveOp = dict()
		simdata_time = list()
		r, rt = self._residual(shot, m0, frequencies, dWaveOp=dWaveOp, compute_resid_time=True) #rt is residual in time
		
		# Perform the migration or F* operation to get the gradient component
		g = self.modeling_tools.migrate_shot(shot, m0, rt, frequencies, frequency_weights=frequency_weights, dWaveOp=dWaveOp)
		g.toreal()
		
		if ignore_minus:
			return g, r
		else:
			return -1*g, r
	
	def compute_gradient(self, shots, m0, frequencies=None, frequency_weights=None, aux_info={}, **kwargs):
		"""Compute the gradient for a set of shots.
		
		Computes the gradient as
			-F*(d - scriptF[m0]) = -sum(F*_s(d - scriptF_s[m0])) for s in shots
		
		Parameters
		----------
		shots : list of pysit.Shot
			List of Shots for which to compute the gradient.
		
		"""
	
		if frequencies is None:
			raise ValueError('A set of frequencies must be specified.')
		
		if frequency_weights is not None and (len(frequencies) != len(frequency_weights)):
			raise ValueError('Weights and frequencies must be the same length.')
		
		# compute the portion of the gradient due to each shot
		grad = m0.perturbation()
		r_norm2 = 0.0
		for shot in shots:
			# ensure that the dft of the data exists
			shot.receivers.compute_data_dft(frequencies)
			
			g, r = self._gradient_helper(shot, m0, frequencies, frequency_weights, ignore_minus=True, **kwargs) 
			grad -= g # handle the minus due to the objective function
			
			if frequency_weights is None:
				frequency_weights = itertools.repeat(1.0)
			
			for nu,weight in zip(frequencies,frequency_weights):
				r_norm2 += weight*np.linalg.norm(r[nu])**2
						
		# sum-reduce and communicate result
		if self.parallel_wrap_shot.use_parallel:
			# Allreduce wants an array, so we give it a 0-D array
			new_r_norm2 = np.array(0.0) 			
			self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
			r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element
			
			ngrad = np.zeros_like(grad.asarray())
			self.parallel_wrap_shot.comm.Allreduce(grad.asarray(), ngrad)
			grad=m0.perturbation(data=ngrad)
		
		
		# store any auxiliary info that is requested
		if ('residual_norm' in aux_info) and aux_info['residual_norm'][0]:
			aux_info['residual_norm'] = (True, np.sqrt(r_norm2))
		if ('objective_value' in aux_info) and aux_info['objective_value'][0]:
			aux_info['objective_value'] = (True, 0.5*r_norm2)
		
		return grad
	
	def apply_hessian(self, shots, m0, m1,
	                        frequencies=None, frequency_weights=None,
	                        hessian_mode='approximate', levenberg_mu=0.0, **kwargs):
		
		modes = ['approximate', 'full', 'levenberg']
		if hessian_mode not in modes:
			raise ValueError("Invalid Hessian mode.  Valid options for applying hessian are {0}".format(modes))	
		
		result = m0.perturbation()
		
		if hessian_mode in ['approximate', 'levenberg']:
			for shot in shots:
				# ensure that the dft of the data exists
				shot.receivers.compute_data_dft(frequencies)

				linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, frequencies, return_parameters=['simdata_time', 'dWaveOp0'])
				
				d1 = linear_retval['simdata_time']
				dWaveOp0 = linear_retval['dWaveOp0']
				result += self.modeling_tools.migrate_shot(shot, m0, d1, frequencies, frequency_weights=frequency_weights, dWaveOp=dWaveOp0)
				
				result.toreal()
					
		elif hessian_mode == 'full':
			raise NotImplementedError("Full hessian for the nonlinear hybrid frequency objective function not implemented.")
			for shot in shots:
				# ensure that the dft of the data exists
				shot.receivers.compute_data_dft(frequencies)
				# Run the forward modeling step
				dWaveOp0 = dict()
				r, r0t = self._residual(shot, m0, frequencies, dWaveOp=dWaveOp0, compute_resid_time=True)
				
				linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, frequencies, return_parameters=['simdata_time', 'dWaveOp1'])
				d1t = linear_retval['simdata_time']
				dWaveOp1 = linear_retval['dWaveOp1']
				
				# <q, u1tt>, first adjointy bit
				
				result = self.modeling_tools.migrate_shot_hessian(shot, m0, m1, r0t, d1t, dWaveOp0, dWaveOp1, frequencies, frequency_weights)
				
				result.toreal()
								
#				# This next part is currently a very broken.  For this method, I 
#				# will actually need a migrate shot or adjoint solve method that 
#				# tries to do both operations at once, since we cant store qtt in 
#				# the first one and pass it on to the second one.
#				
#				dWaveOpAdj1=[]			
#				res1 = self.modeling_tools.migrate_shot( shot, m0, r0, dWaveOp=dWaveOp1, dWaveOpAdj=dWaveOpAdj1)	
#				result += res1
#				
#				# <p, u0tt>
#				res2 = self.modeling_tools.migrate_shot(shot, m0, d1, operand_dWaveOpAdj=dWaveOpAdj1, operand_model=m1, dWaveOp=dWaveOp0)
#				result += res2

		# sum-reduce and communicate result
		if self.parallel_wrap_shot.use_parallel:
			
			nresult = np.zeros_like(result.asarray())
			self.parallel_wrap_shot.comm.Allreduce(result.asarray(), nresult)
			result = m0.perturbation(data=nresult)
			
		# Note, AFTER the application has been done in parallel do this.
		if hessian_mode == 'levenberg':
			result += levenberg_mu*m1
		
		return result