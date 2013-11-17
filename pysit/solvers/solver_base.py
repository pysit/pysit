import warnings

import numpy as np

from solver_data import SolverDataBase

__all__=['SolverBase']

__docformat__ = "restructuredtext en"

class NullModelParameters(object):
	def __init__(self, *args, **kwargs):
		raise TypeError("NullModelParameters should never be instantiated.")
		
class SolverBase(object): 
	""" Base class for pysit solvers. (e.g., wave, helmholtz, and laplace-domain)
	
	This class serves as a base class for wave equation solvers in pysit.  It 
	defines some required interface items.
	
	Attributes
	----------
	mesh : pysit.Mesh
		Computational domain on which the source is defined.
	domain : pysit.Domain
		Physical (and numerical) domain on which the solver is operates.
	model_parameters : self.WaveEquationParamters
		Object containing the relevant parameters for a given wave equation.
	
	"""
	
	# Read-only property
	@property #getter
	def solver_type(self): return None
	
	@property #getter
	def cpp_accelerated(self): return False
	
	def __init__(self, mesh, model_parameters={}, 
	                   spatial_accuracy_order=2,
	                   precision = 'double',
	                   spatial_shifted_differences=False,
	                   use_cpp_acceleration=False, 
	                   **kwargs):
		"""Constructor for the WaveSolverBase class.
		
		Parameters
		----------
		mesh : pysit.Mesh
			Computational domain on which the source is defined.
		model_parameters : dict
			Dictionary of initial wave parameters for the solver.	
		"""
	
		self.mesh = mesh
		self.domain = mesh.domain
		
		self.spatial_accuracy_order = spatial_accuracy_order
		self.spatial_shifted_differences = spatial_shifted_differences
		
	
		if precision in ['single', 'double']:
			self.precision = precision
			
			if self.solver_type == 'time':
				self.dtype = np.double if precision == 'double' else np.single
			else:
				self.dtype = np.complex128 if precision == 'double' else np.complex64
		
		self.use_cpp_acceleration = use_cpp_acceleration
		if use_cpp_acceleration and not self.cpp_accelerated:
			self.use_cpp_acceleration = False
			warnings.warn('C++ accelerated solver is not available for solver type {0}'.format(type(self)))
		
		self._mp = None
		self._model_change_count = 0
		self.model_parameters = self.ModelParameters(mesh, inputs=model_parameters)
		
	@property #getter
	def model_parameters(self): return self._mp
	
	@model_parameters.setter
	def model_parameters(self, mp):
		if type(mp) is self.ModelParameters:
			if self._mp is None or np.linalg.norm(self._mp.without_padding().data - mp.data) != 0.0:
				self._mp = mp.with_padding(padding_mode='edge')
				
				self._process_mp_reset()
				
				self._model_change_count += 1
		else:
			raise TypeError('{0} is not of type {1}'.format(type(mp), self.ModelParameters))

	def _process_mp_reset(self, *args, **kwargs):
		raise NotImplementedError('_process_mp_reset() must be implemented by a subclass.')
	
	
	def compute_dWaveOp(self, regime, *args):
		return self.__getattribute__('_compute_dWaveOp_{0}'.format(regime))(*args)
	
	ModelParameters = None
	
	WavefieldVector = None
	
	_SolverData = SolverDataBase
	
	def SolverData(self, **kwargs):
		return self._SolverData(self, **kwargs)
	
	
	