import itertools

import numpy as np

__all__ = ['bspline', 'quadratic', 'ConstructableDict', 'Bunch']

def bspline(x):
	x = np.array(x*1.0)  # Note that x must be a numpy array and also must be real.  As implemented, this performs a copy.  If there is slowness, this should be conditional. Also, perhaps should be conditional so no copying large domains.
	if (x.shape == () ): # Non-array argument must go to non-dimensionless array
		x.shape = (1,)		  
							
	retvec = np.zeros_like(x)

	loc = np.where(x < 0.5)
	retvec[loc] = 1.5 * (8./6.) * x[loc]**3

	loc = np.where(x >= 0.5)
	retvec[loc] = 1.5 * ((-4.0*x[loc]**3 + 8.0*x[loc]**2 - 4.0*x[loc] + 2.0/3.0))
				
	return retvec

def quadratic(x):
	return x**2
	
class Bunch(dict):
	""" An implementation of the Bunch pattern.
	
	See also: 
	  http://code.activestate.com/recipes/52308/
	  http://stackoverflow.com/questions/2641484/class-dict-self-init-args
	
	This will likely change into something more parametersific later, so that changes
	in the subparameters will properly handle side effects.  For now, this
	functions as a struct-on-the-fly.
	
	"""
	def __init__(self, **kwargs):
		dict.__init__(self, **kwargs)
		self.__dict__ = self

class ConstructableDict(dict):
	""" A ConstructableDict returns the value mapped to a key.  If that key
	does not exist, a function which creates the desired value at the key is 
	called with the key as the argument.
	
	Examples
	--------
	
	>>> def buildfun(key):
	... 	def myprint(x): print key*x
	... 	return myprint
		
	>>> a = ConstructableDict()
	>>> a
	{}
	>>> a[5]
	<function myprint>
	>>> a[6](7)
	42
	
	"""
	def __init__(self, func):
		self.func = func
		dict.__init__(self)
		
	def __getitem__(self, key):
		if key not in self:
			dict.__setitem__(self, key, self.func(key))
		
		return dict.__getitem__(self, key)
			
	def __setitem__(self, key, arg):
		raise NotImplementedError('__setitem__ should not be called for ConstructableDict instances')