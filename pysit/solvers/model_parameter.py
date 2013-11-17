import itertools

import numpy as np

__all__ = ['ModelParameterBase', 'ModelPerturbationBase',
           'WaveSpeed', 'BulkModulus', 'Density',
           'ConstantDensityAcousticParameters',
           ]

def finite(x):
	""" Checks to see if an array is entirely finite, e.g., no NaN and no inf. """
	
	return np.all(np.isfinite(x))

def positivity(x): 
	""" Checks to see if an array is entirely strictly positive. """
	return np.all(x > 0)

def reasonability(x, val, mode): 
	""" Checks to see if an array is within a set of bounds. """
	if mode == 'high':
		return  np.all(x <= val)
	if mode == 'low':
		return  np.all(x >= val)

class ModelParameterDescription(object):
	""" Base class for describing a wave model.
	
	Attributes
	----------
	
	name : string
		Text name for the model, e.g. 'C' for wavespeed.
	constraints : tuple of tuple
		A tuple (cfunc, cargs, cname) where cfunc is a constraint function, 
		cargs are the arguments to that function, and cname names the constraint 
		for output purposes.
		
	"""
	name = None
	constraints = tuple()
	
	@classmethod
	def linearize(cls, data):
		""" Convert an array from nonlinear model form to linear model form."""
		raise NotImplementedError()
	
	@classmethod
	def unlinearize(cls, data):
		""" Convert an array from linear model form to nonlinear model form."""
		raise NotImplementedError()
		
	@classmethod
	def validate(cls, data):
		""" Verify that a model array satisfies all specified constrains. """
		for cfunc, cargs, cname in cls.constraints:
			if not cfunc(data, *cargs):
				return False, cname
				
		return True, None
		
class WaveSpeed(ModelParameterDescription):
	name = 'C'
	constraints = tuple(((finite, tuple(), 'finite'),
	                     (positivity, tuple(), 'positivity'),
#	                    (reasonability, (300.0, 'low'), 'lower bound'),
#	                    (reasonability, (7500.0, 'high'), 'upper bound')
	                    ))
	    
	@classmethod
	def linearize(cls, data): return data**-2
		
	@classmethod
	def unlinearize(cls, data): return np.sqrt(1./data)

		
class BulkModulus(ModelParameterDescription):
	name = 'kappa'
	constraints = tuple(((finite, tuple(), 'finite'),
	                     (positivity, tuple(), 'positivity'),
	                    ))
	    
	@classmethod
	def linearize(cls, data): return 1./data
		
	@classmethod
	def unlinearize(cls, data): return 1./data

		
class Density(ModelParameterDescription):
	name = 'rho'
	constraints = tuple(((finite, tuple(), 'finite'),
	                     (positivity, tuple(), 'positivity'),
#	                    (reasonability, (300.0, 'low'), 'lower bound'),
#	                    (reasonability, (7500.0, 'low'), 'upper bound')
	                    ))
	    
	@classmethod
	def linearize(cls, data): return 1./data
		
	@classmethod
	def unlinearize(cls, data): return 1./data

class ModelParameterBase(object):
	"""Container class for the model parameters for the wave and Helmholtz 
	equations in question.
	
	This provides array-like functionality so that we can do simple expressions 
	like addition and scalar multiplication of models.  From an outside perspective, models are to be treated as if they are in linear form.  Internally, the data is stored in nonlinear form as this is more convenient from a solver perspective.  In particular, solvers use nonlinear models more often than inversion routines use linear models.  As such, all operations linearize the stored data before operating on it and returning the result to nonlinear form.
	t_density_acoustic.py for an example.)
	
	Attributes
	----------
	domain : pysit.Domain
		Coordinate system for the specified model.
	
	"""	
	class Perturbation(object):
		""" Container for a model perturbation, which is always in linear form and looks similar to a ModelParameter, but is defined in ModelPerturbationBase."""
		
		def __init__(self): raise NotImplementedError("Must be implemented in subclass.")
	
	# Gives access to everything in nonlinear form, but ModelParameterBase is a LINEAR object.  That is,
		
	parameter_list = []
	
	# Automatically add convenience properties for the model names (C, rho, etc)
	def add_property(self, attr, idx, default=None):
		
		setattr(self, "_{0}_idx".format(attr), idx)
		def setter(self, value):
			dof = self.mesh.dof(include_bc=self.padded)
			idx = getattr(self, "_{0}_idx".format(attr))
			self.data[(dof*idx):((idx+1)*dof)] = value
		def getter(self):
			dof = self.mesh.dof(include_bc=self.padded)
			idx = getattr(self, "_{0}_idx".format(attr))
			return self.data[(dof*idx):((idx+1)*dof)].view()
		setattr(type(self), "{0}".format(attr), property(getter, setter))
		setattr(type(self), "_{0}_description".format(attr), self.parameter_list[idx])
	
	def __init__(self, mesh, inputs=None, linear_inputs=None, padded=False):
		
		self.padded = padded
		self.mesh = mesh
		dof = mesh.dof(include_bc=self.padded)
		self.parameter_count = len(self.parameter_list)
		
		self.data = np.ones((self.parameter_count*dof,1))*np.inf
		
		# nonlinear inputs has priority
		# inputs must match the shape correctly, eg padded if padded is set
		if inputs is not None:
			if type(inputs) in (tuple, list):
				for p,inp,cnt in zip(self.parameter_list,inputs,itertools.count()):
					sl = slice(cnt*dof, (cnt+1)*dof)
					self.data[sl] = 0
					self.data[sl] += inp
			elif type(inputs) is dict:
				for p,cnt in zip(self.parameter_list,itertools.count()):
					if p.name in inputs:
						sl = slice(cnt*dof, (cnt+1)*dof)
						self.data[sl] = 0
						self.data[sl] += inputs[p.name] 
			elif type(inputs) is np.ndarray:
				if inputs.size == self.data.size:
					self.data = inputs
					self.data.shape=-1,1
				else:
					raise ValueError("Improper dimensions for input array.")
			else:
				raise ValueError("Invalid format for collection of input parameters.")
		
		# can initialize with a linear model
		if linear_inputs is not None:
			if type(linear_inputs) in (tuple, list):
				for p,inp,cnt in zip(self.parameter_list,linear_inputs,itertools.count()):
					sl = slice(cnt*dof, (cnt+1)*dof)
					self.data[sl] = 0
					self.data[sl] += p.unlinearize(inp)
			elif type(linear_inputs) is np.ndarray:
				if linear_inputs.size == self.data.size:
					for p,cnt in zip(self.parameter_list,itertools.count()):
						sl = slice(cnt*dof, (cnt+1)*dof)
						self.data[sl] = 0
						self.data[sl] += p.unlinearize(linear_inputs[sl])
				else:
					raise ValueError("Improper dimensions for input array.")
			else:
				raise ValueError("Invalid format for collection of input parameters.")
		
		# set up the accessors for the model properties
		for p,idx in zip(self.parameter_list,itertools.count()):
			self.add_property(p.name, idx)
	
	def perturbation(self, data=None, *args, **kwargs):
		if data is None:
			return self.Perturbation(self.mesh, padded=self.padded, *args, **kwargs)
		else:
			return self.Perturbation(self.mesh, padded=self.padded, inputs=data, *args, **kwargs)
				
	def linearize(self, asperturbation=False):
		
		# output is always an array since models store nonlinear things (even though they act like linear things)
		
		dof = self.mesh.dof(include_bc=self.padded)
		out_arr = np.zeros((self.parameter_count*dof,1))
		for p,cnt in zip(self.parameter_list,itertools.count()):
			sl = slice(cnt*dof, (cnt+1)*dof)
			out_arr[sl] += p.linearize(self.data[sl])
		
		if asperturbation:
			return self.perturbation(out_arr)
		else:
			return out_arr
		
	def validate(self, raise_exception=False):
		retval = True
		for p in self.parameter_list:
			result = p.validate(getattr(self, p.name))
			if result[0] == False:
				retval = False
				if raise_exception:
					raise Exception("Validation of model failed: {0}".format(result[1]))
		return retval
		
	def with_padding(self, **kwargs):
		if self.padded:
			return self
			
		olddof = self.mesh.dof(include_bc=False)
		newdof = self.mesh.dof(include_bc=True)
		result = type(self)(self.mesh, padded=True)
		for p,idx in zip(self.parameter_list, itertools.count()):
			oldsl=slice(idx*olddof,(idx+1)*olddof)
			newsl=slice(idx*newdof,(idx+1)*newdof)
			result.data[newsl] = 0
			result.data[newsl] += self.mesh.pad_array(self.data[oldsl], **kwargs)
		
		return result
	
	def without_padding(self):
		if not self.padded:
			return self
			
		olddof = self.mesh.dof(include_bc=True)
		newdof = self.mesh.dof(include_bc=False)
		result = type(self)(self.mesh, padded=False)
		for p,idx in zip(self.parameter_list, itertools.count()):
			oldsl=slice(idx*olddof,(idx+1)*olddof)
			newsl=slice(idx*newdof,(idx+1)*newdof)
			result.data[newsl] = 0
			result.data[newsl] += self.mesh.unpad_array(self.data[oldsl])
		
		return result
		
	def M(self, i):
		dof = self.mesh.dof(include_bc=self.padded)
		if type(i) is not int:
			raise TypeError("__call__ method used for indexing the list of models")
		if (i < self.parameter_count) and (i >= 0):
			return self.data[(i*dof):((i+1)*dof),0].reshape(-1,1)
		else:
			raise IndexError("Parameter index out of bounds. (requires {0} > idx >= 0)".format(self.parameter_count))
	
	def __mul__(self, rhs):
		
		# iterables of scalars, so models can be rescaled differently are OK
		if type(rhs) in (list, tuple, np.ndarray) and (len(rhs) == self.parameter_count):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += p.unlinearize(rhs[idx] * p.linearize(self.data[sl]) )
		# product with an array is OK, but will return an array
		elif type(rhs) is np.ndarray and (rhs.shape == self.data.shape):
			result = self.linearize() * rhs
		# product with a perturbation is OK, but will return an array
		elif type(rhs) is self.Perturbation:
			result = self.linearize() * rhs.data
		# any other sort of legal product (usually a single scalar) will return a new model instance
		else:
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += p.unlinearize(rhs * p.linearize(self.data[sl]) )
		
		return result
		
	def __rmul__(self,lhs):
		return self.__mul__(lhs)
		
	def __add__(self, rhs):
		# iterables of scalars, so models can be rescaled differently are OK
		if type(rhs) in (list, tuple, np.ndarray) and (len(rhs) == self.parameter_count):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += p.unlinearize( p.linearize(self.data[sl]) + rhs[idx] )
		# addition with a model parameter yeilds a model parameter
		elif type(rhs) is type(self) and (rhs.data.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += p.unlinearize(p.linearize(self.data[sl])+p.linearize(rhs.data[sl]))
		# array rhs is treated as LINEAR
		elif type(rhs) is np.ndarray and (rhs.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += p.unlinearize(p.linearize(self.data[sl])+rhs[sl])
		# Perturbation RHS is LINEAR
		elif type(rhs) is self.Perturbation and (rhs.data.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += p.unlinearize(p.linearize(self.data[sl])+rhs.data[sl])
		# any other sort of legal sum (usually a single scalar or an array) will return a new model instance
		else:
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += p.unlinearize(rhs + p.linearize(self.data[sl]) )
		
		return result
	
	def __radd__(self,lhs):
		return self.__add__(lhs)
		
	def __sub__(self, rhs):
		# iterables of scalars, so models can be rescaled differently are OK
		if type(rhs) in (list, tuple, np.ndarray) and (len(rhs) == self.parameter_count):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += p.unlinearize( p.linearize(self.data[sl]) - rhs[idx] )
		# difference with a ModelParamter or self.Perturbation is OK, but will return a perturbation
		elif type(rhs) in [type(self), self.Perturbation] and (rhs.data.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			arr = np.zeros_like(self.data)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				arr[sl] += p.linearize(self.data[sl])-p.linearize(rhs.data[sl])
			result = self.perturbation(data=arr)
		# difference with a ModelParamter is OK, but will return an array
		elif type(rhs) is np.ndarray and (rhs.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += p.unlinearize(p.linearize(self.data[sl])-rhs[sl])
		# any other sort of legal difference (usually a single scalar or an array) will return a perturbation
		else:
			dof = self.mesh.dof(include_bc=self.padded)
			arr = np.zeros_like(self.data)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				arr[sl] = 0
				arra[sl] += p.unlinearize(p.linearize(self.data[sl]) - rhs)
			result = self.perturbation(data=arr)
		
		return result
	
	def asarray(self): 
		return self.data
	
	def norm(self, ord=None):
		return np.linalg.norm(self.data, ord=ord)
		
	@property #so m.T works
	def T(self):
		return self.data.T
	
	def __deepcopy__(self, memo):
		
		new_copy = type(self)(self.mesh, padded=self.padded, inputs=self.data.__deepcopy__(memo))
		
		return new_copy
			
	def __repr__(self):
		return self.data.__repr__()

class ModelPerturbationBase(object):
	
	parameter_list = []
	
	# Automatically add convenience properties for the model names (C, rho, etc), not sure if this should happen as thesea re no longer the nonlinear things
	def add_property(self, attr, idx, default=None):
		
		dof = self.mesh.dof(include_bc=self.padded)
		
		setattr(self, "_{0}_idx".format(attr), idx)
		def setter(self, value):
			idx = getattr(self, "_{0}_idx".format(attr))
			self.data[(dof*idx):((idx+1)*dof)] = value
		def getter(self):
			idx = getattr(self, "_{0}_idx".format(attr))
			return self.data[(dof*idx):((idx+1)*dof),0].reshape(-1,1)
		setattr(type(self), "{0}".format(attr), property(getter, setter))
		setattr(type(self), "_{0}_description".format(attr), self.parameter_list[idx])

	def __init__(self, mesh, padded=False, inputs=None, **kwargs):
		# inputs assumed to be LINEAR
		self.mesh = mesh
		self.padded = padded
		dof = self.mesh.dof(include_bc=self.padded)
		self.parameter_count = len(self.parameter_list)
		
		self.dtype = np.double
		if 'dtype' in kwargs:
			self.dtype = kwargs['dtype']
			
		self.data = np.zeros((self.parameter_count*dof,1), dtype=self.dtype)
		
		# nonlinear inputs has priority
		if inputs is not None:
			if type(inputs) in (tuple, list):
				for p,inp,cnt in zip(self.parameter_list,inputs,itertools.count()):
					sl = slice(cnt*dof, (cnt+1)*dof)
					self.data[sl] = 0
					self.data[sl] += inp
			elif type(inputs) is np.ndarray:
				if inputs.size == self.data.size:
					sh = self.data.shape
					self.data.shape = inputs.shape
					self.data *= 0
					self.data += inputs
					self.data.shape = sh
				else:
					raise ValueError("Improper dimensions for input array.")
			else:
				raise ValueError("Invalid format for collection of input parameters.")
		
		# set up the accessors for the model properties
		for p,idx in zip(self.parameter_list,itertools.count()):
			self.add_property(p.name, idx)
		
	def M(self, i):
		dof = self.mesh.dof(include_bc=self.padded)
		if type(i) is not int:
			raise TypeError("__call__ method used for indexing the list of models")
		if (i < self.parameter_count) and (i >= 0):
			return self.data[(i*dof):((i+1)*dof),0].reshape(-1,1)
		else:
			raise IndexError("Parameter index out of bounds. (requires {0} > idx >= 0)".format(self.parameter_count))
		
	def with_padding(self, **kwargs):
		if self.padded:
			return self
			
		olddof = self.mesh.dof(include_bc=False)
		newdof = self.mesh.dof(include_bc=True)
		result = type(self)(self.mesh, padded=True, dtype=self.dtype)
		for p,idx in zip(self.parameter_list, itertools.count()):
			oldsl=slice(idx*olddof,(idx+1)*olddof)
			newsl=slice(idx*newdof,(idx+1)*newdof)
			result.data[newsl] = 0
			result.data[newsl] += self.mesh.pad_array(self.data[oldsl], **kwargs)
		
		return result
	
	def without_padding(self):
		if not self.padded:
			return self
			
		olddof = self.mesh.dof(include_bc=True)
		newdof = self.mesh.dof(include_bc=False)
		result = type(self)(self.mesh, padded=False, dtype=self.dtype)
		for p,idx in zip(self.parameter_list, itertools.count()):
			oldsl=slice(idx*olddof,(idx+1)*olddof)
			newsl=slice(idx*newdof,(idx+1)*newdof)
			result.data[newsl] = 0
			result.data[newsl] += self.mesh.unpad_array(self.data[oldsl])
			
		return result

	def __add__(self, rhs):
		# iterables of scalars, so models can be rescaled differently are OK
		if type(rhs) in (list, tuple, np.ndarray) and (len(rhs) == self.parameter_count):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded, dtype=self.dtype)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += self.data[sl] + rhs[idx]
		# addition with an array is OK
		elif type(rhs) is np.ndarray and (rhs.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded, dtype=self.dtype)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += self.data[sl]+rhs[sl]
		# Perturbation RHS is LINEAR
		elif type(rhs) is type(self) and (rhs.data.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded, dtype=self.dtype)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += self.data[sl]+rhs.data[sl]
		# if the rhs has its own add routing, try that...(should handle perturbations + model_parameters
		elif hasattr(rhs, '__add__'):
			result = rhs.__add__(self)
		# any other sort of legal sum (usually a single scalar or an array) will return a new model instance
		else:
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded, dtype=self.dtype)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += rhs + self.data[sl]
		
		return result
	
	def __radd__(self,lhs):
		return self.__add__(lhs)
		
	def __iadd__(self, rhs):
		# iterables of scalars, so models can be rescaled differently are OK
		if type(rhs) in (list, tuple, np.ndarray) and (len(rhs) == self.parameter_count):
			dof = self.mesh.dof(include_bc=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				self.data[sl] += rhs[idx]
		# addition with an array is OK
		elif type(rhs) is np.ndarray and (rhs.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				self.data[sl] += rhs[sl]
		# Perturbation RHS is LINEAR
		elif type(rhs) is type(self) and (rhs.data.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				self.data[sl] += rhs.data[sl]
		# any other sort of legal sum (usually a single scalar or an array) will return a new model instance
		else:
			dof = self.mesh.dof(include_bc=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				self.data[sl] += rhs
				
		return self

	def __sub__(self, rhs):
		# iterables of scalars, so models can be rescaled differently are OK
		if type(rhs) in (list, tuple, np.ndarray) and (len(rhs) == self.parameter_count):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded, dtype=self.dtype)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += self.data[sl] - rhs[idx]
		# addition with an array is OK
		elif type(rhs) is np.ndarray and (rhs.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded, dtype=self.dtype)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += self.data[sl]-rhs[sl]
		# Perturbation RHS is LINEAR
		elif type(rhs) is type(self) and (rhs.data.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded, dtype=self.dtype)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += self.data[sl]-rhs.data[sl]
		# any other sort of legal sum (usually a single scalar or an array) will return a new model instance
		else:
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded, dtype=self.dtype)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += self.data[sl] - rhs
		
		return result

	def __isub__(self, rhs):
		# iterables of scalars, so models can be rescaled differently are OK
		if type(rhs) in (list, tuple, np.ndarray) and (len(rhs) == self.parameter_count):
			dof = self.mesh.dof(include_bc=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				self.data[sl] -= rhs[idx]
		# addition with an array is OK
		elif type(rhs) is np.ndarray and (rhs.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				self.data[sl] -= rhs[sl]
		# Perturbation RHS is LINEAR
		elif type(rhs) is type(self) and (rhs.data.shape == self.data.shape):
			dof = self.mesh.dof(include_bc=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				self.data[sl] -= rhs.data[sl]
		# any other sort of legal sum (usually a single scalar or an array) will return a new model instance
		else:
			dof = self.mesh.dof(include_bc=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				self.data[sl] -= rhs
		
		return self

	
	def __mul__(self, rhs):
		
		# iterables of scalars, so models can be rescaled differently are OK
		if type(rhs) in (list, tuple, np.ndarray) and (len(rhs) == self.parameter_count):
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded, dtype=self.dtype)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += rhs[idx] * self.data[sl]
		# product with an array is OK, but will return an array
		elif type(rhs) is np.ndarray and (rhs.shape == self.data.shape):
			result = self.data * rhs
		# product with a perturbation is OK, but will return an array
		elif type(rhs) is type(self):
			result = self.data * rhs.data
		# any other sort of legal product (usually a single scalar) will return a new model instance
		else:
			dof = self.mesh.dof(include_bc=self.padded)
			result = type(self)(self.mesh, padded=self.padded, dtype=self.dtype)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				result.data[sl] = 0
				result.data[sl] += rhs * self.data[sl]
		
		return result
		
	def __rmul__(self,lhs):
		return self.__mul__(lhs)
		
	def __imul__(self, rhs):
		
		# iterables of scalars, so models can be rescaled differently are OK
		if type(rhs) in (list, tuple, np.ndarray) and (len(rhs) == self.parameter_count):
			dof = self.mesh.dof(include_bc=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				self.data[sl] *= rhs[idx]
		# product with an array is OK, but will return an array
		elif type(rhs) is np.ndarray and (rhs.shape == self.data.shape):
			self.data *= rhs
		# product with a perturbation is OK, but will return an array
		elif type(rhs) is type(self):
			self.data *= rhs.data
		# any other sort of legal product (usually a single scalar) will return a new model instance
		else:
			dof = self.mesh.dof(include_bc=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				self.data[sl] *= rhs
		
		return self
	
	def asarray(self): 
		return self.data

	def norm(self, ord=None):
		return np.linalg.norm(self.data, ord=ord)
	
	@property #so p.T works
	def T(self):
		return self.data.T
		
	def toreal(self):
		if np.iscomplexobj(self.data):
			self.data = np.real(self.data)
			self.dtype = self.data.dtype
			
	def inner_product(self, other):
		if type(other) is type(self):
			ip = 0.0
			dof = self.mesh.dof(include_bc=self.padded)
			for p,idx in zip(self.parameter_list, itertools.count()):
				sl=slice(idx*dof,(idx+1)*dof)
				ip += self.mesh.inner_product(self.data[sl], other.data[sl])
		else:
			raise ValueError('Perturbation inner product is only defined for perturbations.')
		return ip
	
	def __deepcopy__(self, memo):
		
		new_copy = type(self)(self.mesh, inputs=self.data.__deepcopy__(memo), padded=self.padded)
		
		return new_copy

class ConstantDensityAcousticParameters(ModelParameterBase):

	parameter_list = [WaveSpeed]

	class Perturbation(ModelPerturbationBase):
		
		parameter_list = [WaveSpeed]
	
#class AcousticParameters(ModelParameterBase):
#	
#	parameter_list = [BulkModulus, Density]
#	
#	class Perturbation(ModelPerturbationBase):
#		
#		parameter_list = [BulkModulus, Density]

