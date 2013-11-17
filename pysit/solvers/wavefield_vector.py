import numpy as np

__all__ = ['WavefieldVectorBase']

class WavefieldVectorBase(object):
	
	aux_names = []
		
	def __init__(self, mesh, dtype=np.double, data=None):
		
		self.mesh = mesh
		self.n_aux = len(self.aux_names)
		self.n_fields = 1 + self.n_aux
		self.dof = mesh.dof(include_bc=True)
		self.data_shape = (self.dof*(1+self.n_aux), 1)
				
		if data is None:
			self.dtype=dtype
			self._data = np.zeros(self.data_shape, dtype=self.dtype)
		elif type(data) is np.ndarray:
			self.dtype=data.dtype
			if data.size != self.n_fields*self.dof:
				raise ValueError('Size mismatch between mesh and proposed data.')
			self._data = data
			self._data.shape = self.data_shape
		else:
			raise ValueError('Input data must be of type ndarray.')
			
		self._uslice = slice(0,self.dof,None)
				
		for i in xrange(self.n_aux):
			self.add_auxiliary_wavefield(self.aux_names[i], i+1)

	@property 
	def data(self):	return self._data
	@data.setter
	def data(self, arg): self._data[:] = arg
		
	@property
	def u(self):
		return self._data[self._uslice]
	@u.setter
	def u(self, value):
		self._data[self._uslice] = value
		
	@property
	def primary_wavefield(self):
		return self._data[self._uslice]
	@primary_wavefield.setter
	def primary_wavefield(self, value):
		self._data[self._uslice] = value
		
	def __deepcopy__(self, memo):
		
		new_copy = type(self)(self.mesh, dtype=self.dtype, data=self._data.__deepcopy__(memo))
		
		return new_copy
		
	def add_auxiliary_wavefield(self, attr, idx):
		# access auxiliary variables by index
		setattr(self, "_{0}_idx".format(attr), idx)
		
		# define setter and getter for the auxiliary
		def setter(self, value):
			idx = getattr(self, "_{0}_idx".format(attr))
			self._data[(self.dof*idx):((idx+1)*self.dof)] = value
		def getter(self):
			idx = getattr(self, "_{0}_idx".format(attr))
			return self._data[(self.dof*idx):((idx+1)*self.dof)]
		
		# Name the auxiliary as a property
		setattr(type(self), "{0}".format(attr), property(getter, setter))
		
	# now for the overloads:
	
	def __add__(self, rhs):
		if type(rhs) is type(self):
			result = type(self)(self.mesh, dtype=self.dtype)
			result.data = self.data + rhs.data
		else: # scalars, ndarrays, etc
			result = type(self)(self.mesh, dtype=self.dtype)
			result.data = self.data + rhs
		return result
		
	def __radd__(self, lhs):
		return self.__add__(lhs)
		
	def __iadd__(self, rhs):
		if type(rhs) is type(self):
			self.data += rhs.data
		else: # scalars, ndarrays, etc
			self.data += rhs
		return self
			
	def __sub__(self, rhs):
		if type(rhs) is type(self):
			result = type(self)(self.mesh, dtype=self.dtype)
			result.data = self.data - rhs.data
		else: # scalars, ndarrays, etc
			result = type(self)(self.mesh, dtype=self.dtype)
			result.data = self.data - rhs
		return result
		
	def __isub__(self, rhs):
		if type(rhs) is type(self):
			self.data -= rhs.data
		else: # scalars, ndarrays, etc
			self.data -= rhs
		return self
		
	def __mul__(self, rhs):
		if type(rhs) is type(self):
			result = type(self)(self.mesh, dtype=self.dtype)
			result.data = self.data * rhs.data
		else: # scalars, ndarrays, etc
			result = type(self)(self.mesh, dtype=self.dtype)
			result.data = self.data * rhs
		return result
		
	def __rmul__(self, lhs):
		return self.__mul__(lhs)
		
	def __imul__(self, rhs):
		if type(rhs) is type(self):
			self.data *= rhs.data
		else: # scalars, ndarrays, etc
			self.data *= rhs
		return self
			
		
		
if __name__ == '__main__':
	
	class Mesh(object):
		def dof(self, **kwargs): return 5
			
	m = Mesh()
	
	
	class MyWFV(WavefieldVectorBase):
		n_aux = 3
		aux_names = ['A', 'B', 'C']
	
	W = WavefieldVectorBase(m)
	
	MW = MyWFV(m)
	
	W._data += np.arange(5).reshape(W._data.shape)
	MW._data += np.arange(MW._data.shape[0]).reshape(MW._data.shape)