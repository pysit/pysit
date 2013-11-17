import numpy as np 

__all__ = ['RectangularDomain', 'Neumann', 'Dirichlet', 'PML']

_cart_keys = { 1 : [(0,'z')],
               2 : [(0,'x'), (1,'z')],
               3 : [(0,'x'), (1,'y'), (2,'z')]
             }
             	
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
		
class Domain(object):
	pass

class RectangularDomain(Domain):
	
	type = 'rectangular'
	
	def __init__(self, *args):
		
		if (len(args) > 3) or (len(args) < 1):
			raise ValueError('Domain construction must be between one and three dimensions.')
		
		self.dim = len(args)
		
		self.parameters = dict()
		
		for (i,k) in _cart_keys[self.dim]:
			
			config = args[i]
			
			if len(config) == 4:
				L, R, lbc, rbc = config # min, max, left BC, right BC
				unit = None
			elif len(config) == 5:
				L, R, lbc, rbc, unit = config # min, max, left BC, right BC, unit
				
			param = Bunch(lbound=float(L), rbound=float(R), lbc=lbc, rbc=rbc, unit=unit)
			
			param.length = (R - L)
			
			
			# access the dimension data by index, key, or shortcut
			self.parameters[i] = param # d.dim[-1]
			self.parameters[k] = param # d.dim['z']
			self.__setattr__(k, param) # d.z
	
	def collect(self, prop):
		""" Collects a ndim-tuple of property 'prop'."""
		return tuple([self.parameters[i][prop] for i in xrange(self.dim)])
		
	def get_lengths(self):
		"""Returns the physical size of the domain as a ndim-tuple."""		
		return self.collect('length')


class AngularDomain(Domain):
	pass
		
class DomainBC(object):
	pass

class Dirichlet(DomainBC):
	type = 'dirichlet'
	
class Neumann(DomainBC):
	type = 'neumann'

class PML(DomainBC):
	type = 'pml'
	
	def __init__(self, length, amplitude, ftype='quadratic', boundary='dirichlet'):
		"""Constructor for the PML class.

		Parameters
		----------
		length : float
			Size of the PML in physical units.
		amplitude : float
			Scaling factor for the PML coefficient.
		ftype : {'b-spline'}
			PML function profile.

		Examples
		--------
		>>> pmlx = PML(0.1, 100)
			
		Todo
		----
		* Implement other profile functions.

		"""	
		# Length is currently in physical units.
		self.length = length
		
		self.amplitude = amplitude 
		
		# Function is the PML function
		self.ftype = ftype
		
		if (ftype == 'b-spline'):
			# This function is a candidate for numpy.vectorize
			self.pml_func = self._bspline
		elif (ftype == 'quadratic'):
			self.pml_func = self._quadratic
		else:
			raise NotImplementedError('{0} is not a currently defined PML function.'.format(ftype))	
			
		if boundary in ['neumann', 'dirichlet']:
			self.boundary_type = boundary
		else:
			raise ValueError("'{0}' is not 'neumann' or 'dirichlet'.".format(boundary))
			
	def _bspline(self, x):
		x = np.array(x*1.0)  # Note that x must be a numpy array and also must be real.  As implemented, this performs a copy.  If there is slowness, this should be conditional. Also, perhaps should be conditional so no copying large domains.
		if (x.shape == () ): # Non-array argument must go to non-dimensionless array
			x.shape = (1,)		  
								
		retvec = np.zeros_like(x)
	
		loc = np.where(x < 0.5)
		retvec[loc] = 1.5 * (8./6.) * x[loc]**3
	
		loc = np.where(x >= 0.5)
		retvec[loc] = 1.5 * ((-4.0*x[loc]**3 + 8.0*x[loc]**2 - 4.0*x[loc] + 2.0/3.0))
					
		return retvec

	def _quadratic(self, x):
		return x**2	
	
	def evaluate(self, n, orientation='right'):
		"""Evaluates the PML on n points."""
		
		val = self.amplitude * self.pml_func(np.linspace(0., 1., n, endpoint=False))
		if orientation is 'left':
			val = val[::-1]
		
		return val
		
#	def evaluate(self, XX, base_point, orientation):
#		"""Evaluates the PML on the given array.
#
#		Parameters
#		----------
#		XX : ndarray
#			Grid of points on which to evaluate the PML.
#		base_point : float
#			The spatial (physical coordinate) location that the PML begins.
#		orientation : {'left', 'right'}
#			The direction in which the PML is oriented.
#			
#		Notes
#		-----
#		Because a PML object is intended to describe the PML in a single
#		direction, any XX can be provided.  The programmer must be sure that
#		XX corresponds with the correct dimension.
#
#		Examples
#		--------
#		>>> d = Domain(((0.0, 1.0, 100),(0.0, 0.5, 50)), 
#		                 pml=(PML(0.1, 100),PML(0.1, 100)))
#		>>> XX, ZZ = d.generate_grid()
#		>>> pml_mtx = d.x.pml.evaluate(XX)
#	
#		"""		
#		pml = np.zeros_like(XX)
#	
#		if orientation == 'left':
#			loc = np.where((XX < base_point) & (XX >= base_point-self.size))
#			pml[loc] = self.amplitude * self.pml_func( (base_point - XX[loc]) / self.size)
#		if orientation == 'right':
#			loc = np.where((XX >= base_point) & (XX < base_point+self.size))
#			pml[loc] = self.amplitude * self.pml_func( (XX[loc] - base_point) / self.size)
#	
#		# candidate for sparse matrix
#		return pml