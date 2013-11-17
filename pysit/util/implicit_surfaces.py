import itertools
import copy

import numpy as np
import scipy
import scipy.linalg

__all__ = ['ImplicitSurface', 
           'ImplicitCollection', 
           'ImplicitPlane', 
           'ImplicitSphere',
           'ImplicitXAlignedCylinder', 
           'ImplicitEllipse',
           'ImplicitIntersection',
           'ImplicitUnion', 
           'ImplicitDifference',
           'ImplicitComplement', 
           'GridMapBase',
           'GridMap', 
           'GridSlip'
		   ]

class ImplicitSurface(object):

	def __init__(self):
		pass
		
	def __call__(self):
		raise NotImplementedError('Must be implemented by subclass.')
		
	def interior(self, grid, asarray=False):
		val = self.__call__(grid)
		if asarray:
			retval = np.zeros_like(val)
			retval[np.where(val < 0.0)] = 1.0
			return retval
		else:
			return np.where(val < 0.0)
		
	def exterior(self, grid, asarray=False):
		val = self.__call__(grid)
		if asarray:
			retval = np.zeros_like(val)
			retval[np.where(val >= 0.0)] = 1.0
			return retval
		else:
			return np.where(val >= 0.0)
		
class ImplicitCollection(ImplicitSurface):
	
	def __init__(self, *items):
		if np.iterable(items[0]):
			self.items = list(items[0])
		else:
			self.items = list(items)
			
	def __call__(self):
		raise NotImplementedError('Must be implemented by subclass.')

class ImplicitPlane(ImplicitSurface):
	def __init__(self, p, n):
		self.p = np.array(p)
		self.n = np.array(n)
		self.n = self.n/np.linalg.norm(self.n)
		self.d = -np.dot(self.p,self.n)
		
	def __call__(self, grid):	
		return self.d + reduce(lambda x,y:x+y, map(lambda x,y:x*y,self.n,grid))

class ImplicitSphere(ImplicitSurface):
	def __init__(self, c=None, r=1.0):
		if c is None:
			self.c = None
		else:
			self.c = np.array(c)
		self.r = r
		
	def __call__(self, grid):	
		if self.c is None:
			c = np.zeros_like(grid[0].shape)
		else:
			c = self.c
		return reduce(lambda x,y:x+y, map(lambda x,y:(y-x)**2,c,grid)) - self.r**2

class ImplicitXAlignedCylinder(ImplicitSurface):
	def __init__(self, c=None, length=1.0, r=1.0):
		if c is None:
			self.c = None
		else:
			self.c = np.array(c)
		self.len = length
		self.r = r
		
	def __call__(self, grid):	
		if self.c is None:
			c = np.zeros_like(grid[0].shape)
		else:
			c = self.c
		
		g = grid[1:]
		cc = c[1:]
#		longways = (grid[1]-c[1])**2 - self.r**2
		longways =  reduce(lambda x,y:x+y, map(lambda x,y:(y-x)**2,cc,g)) - self.r**2
		cutoff   = np.abs(grid[0] - c[0]) -  self.len/2
		return np.maximum(longways, cutoff)

class ImplicitEllipse(ImplicitSurface):
	def __init__(self, c=None, a=None, r=1.0):
		if c is None:
			self.c = None
		else:
			self.c = np.array(c)
		if a is None:
			self.a = None
		else:
			self.a = np.array(a)
		self.r = r
		
	def __call__(self, grid):	
		if self.c is None:
			c = np.zeros(len(grid))
		else:
			c = self.c
		if self.a is None:
			a = np.ones(len(grid))
		else:
			a = self.a
		return reduce(lambda x,y:x+y, map(lambda x,y,z:(((y-x)**2)/z), c,grid,a) ) - self.r**2

class ImplicitIntersection(ImplicitCollection):
	def __init__(self, *items):
		ImplicitCollection.__init__(self, *items)
		
	def __call__(self, grid):
		return reduce(lambda x,y: np.maximum(x,y), map(lambda x: x(grid), self.items))
		
class ImplicitUnion(ImplicitCollection):
	def __init__(self, *items):
		ImplicitCollection.__init__(self, *items)
		
	def __call__(self, grid):
		return reduce(lambda x,y: np.minimum(x,y), map(lambda x: x(grid), self.items))
		
class ImplicitDifference(ImplicitCollection):
	
	# Maybe sometime, this should just take *items, and pop the first one for
	# base.  This way, we can allow a single list to be passed.  For now, whatever.
	def __init__(self, base, *items):
		ImplicitCollection.__init__(self, *items)
		self.base = base

	def __call__(self, grid):
		items = [self.base] + self.items
		return reduce(lambda x,y: np.maximum(x,-y), map(lambda x: x(grid), items))
	
class ImplicitComplement(ImplicitSurface):
	
	def __init__(self, base):
		self.base = base
	
	def __call__(self, grid):
		return -1.0*self.base(grid)
	
	# These must be defined so that the equality is switched if the complement
	# is the calling surface
	def interior(self, grid, asarray=False):
		val = self.__call__(grid)
		if asarray:
			retval = np.zeros_like(val)
			retval[np.where(val <= 0.0)] = 1.0
			return retval
		else:
			return np.where(val <= 0.0)
		
	def exterior(self, grid, asarray=False):
		val = self.__call__(grid)
		if asarray:
			retval = np.zeros_like(val)
			retval[np.where(val > 0.0)] = 1.0
			return retval
		else:
			return np.where(val > 0.0)
		
class GridMapBase(object):
	def __init__(self):
		pass
		
	def __call__(self):
		raise NotImplementedError('Must be implemented by subclass.')
		
		
class GridMap(GridMapBase):
	def __init__(self, funcs):
		self.funcs = funcs
	
	def __call__(self, grid):
		new_grid = []
		for f,g in zip(self.funcs,grid):
			if f is not None:
				new_grid.append(f(g))
			else:
				new_grid.append(g)
				
		return tuple(new_grid)
		
class GridSlip(GridMapBase):
	# Creates a slip or a fault along the specifid plane
	def __init__(self, p, n):
		self.p = np.array(p)
		self.n = np.array(n)
		self.n = self.n/np.linalg.norm(self.n)
		self.d = -np.dot(self.p,self.n)
		
		self.basis = None
		
	def __call__(self, grid, direction, amount):	
		# amount is a scalar
		# direction is a vector that will be normalized
		# the slip occurs along that vector's projection onto the plane, i.e., it is orthogonal to the normal
		# the exterior of the slip plane (the part the normal points to) is the part that is modified.
		d = np.array(direction)
		d = d/np.linalg.norm(d)

		dim = len(d)
		
		if self.basis is None:
			basis = [self.n]
			
			# Gramm schmidt
			while len(basis) != dim:
				c = np.random.rand(3)
				for b in basis:
					c -= np.dot(b,c)*b
				cn = np.linalg.norm(c)
				if cn > 1e-4:
					basis.append(c/cn)
			
			self.basis = basis[1:]
		
		# Project the slip direction onto the basis
		proj=0
		for b in self.basis:
			proj += np.dot(d,b)*b
				
		# Scale the projection
		proj = amount*proj	
			
		# Evaluate the plane to figure out what components of the old grid are shifted.
		val = self.d + reduce(lambda x,y:x+y, map(lambda x,y:x*y,self.n,grid))
		loc = np.where(val >= 0.0)
		
		# Perform the shift.
		new_grid = [copy.deepcopy(g) for g in grid]
		for ng,p in zip(new_grid,proj):
			ng[loc] -=p
			
		return tuple(new_grid)
		
class Weird(ImplicitSurface):
	def __init__(self, p,n, c=None, r=1.0):
		if c is None:
			self.c = None
		else:
			self.c = np.array(c)
		self.r = r
		
		self.p = np.array(p)
		self.n = np.array(n)
		self.n = self.n/np.linalg.norm(self.n)
		self.d = -np.dot(self.p,self.n)
		
	def __call__(self, grid):	
		if self.c is None:
			c = np.zeros_like(grid[0].shape)
		else:
			c = self.c
			
# Creates flat eye thingies
#		val1 = reduce(lambda x,y:x+y, map(lambda x,y:(y-x)**2,c,grid)) - self.r**2
#			
#		val2 = self.d + reduce(lambda x,y:x+y, map(lambda x,y:x*y,self.n,grid))
#		
#		return 4*val1/np.max(val1)+val2

		val1 = reduce(lambda x,y:x+y, map(lambda x,y:(y-x)**2,c,grid)) - self.r**2
			
		val2 = self.d + reduce(lambda x,y:x+y, map(lambda x,y:x*y,self.n,grid))
		
		return 4*val1/np.max(val1)+val2		
		
class Hyperbola(ImplicitSurface):
	def __init__(self, c=None, r=1.0):
		if c is None:
			self.c = None
		else:
			self.c = np.array(c)
		self.r = r
		
	def __call__(self, grid):	
		if self.c is None:
			c = np.zeros_like(grid[0].shape)
		else:
			c = self.c
			
		return  reduce(lambda x,y:-x+y, map(lambda x,y:(y-x)**2,c,grid)) - self.r**2

class Weird2(ImplicitSurface):
	def __init__(self, c=None, r=1.0, s=None):
		if c is None:
			self.c = None
		else:
			self.c = np.array(c)
		
		if s is None:
			self.s = None
		else:
			self.s = np.array(s)
		
		self.r = r
		
	def __call__(self, grid):	
		c = np.zeros_like(grid[0].shape) if self.c is None else self.c
		s = np.ones_like(grid[0].shape) if self.s is None else self.s
			
		return ((grid[0]-c[0])**2)/s[0] + ((grid[1]-c[1])**1)/s[1] - self.r**2
		#reduce(lambda x,y:-x+y, map(lambda x,y:(y-x)**2,c,grid)) - self.r**2

# FIXME			
#if __name__ == "__main__":
#	
#	import numpy as np
#	import matplotlib.pyplot as mpl
#	
#	from pysit import *
#	
#	x_lbc = PML(0.0, 100.0)
#	x_rbc = PML(0.0, 100.0)
#	z_lbc = PML(0.0, 100.0)
#	z_rbc = PML(0.0, 100.0)
#	
#	xmin, xmax = 0.0, 20
#	zmin, zmax = 0.0, 10
#	nx, nz = 500,250
#		
#	x_config = (xmin, xmax, nx, x_lbc, x_rbc)
#	z_config = (zmin, zmax, nz, z_lbc, z_rbc)
#	
#	d = Domain((x_config, z_config))
#	
#	grid = d.generate_grid()
#	grid = grid[0].reshape(d.shape).T, grid[1].reshape(d.shape).T	
#	
#	p1 = ImplicitPlane((0.0,4.0),(0.0,-1.0))
#	p2 = ImplicitPlane((0.0,6.0),(0.0,1.0))
#	
#	w1 = Weird((0.0,5.0),(0.0,-1.0), (10,5))
#	w2 = Weird((0.0,6.0),(0.0,1.0), (10,5))
#	
#	w3 = Weird2((10,4),1.5,(-100.0,1.))
#	w4 = Weird2((10,4),0.5,( 100.0,1.))
#	
#	band = ImplicitIntersection(w1,w2)
#	band = ImplicitIntersection(p1,p2)
#	band = ImplicitDifference(w3,w4)
#	
#	g = GridSlip((10.0,5.0), (1,0.5))
#	d = (0.0,-1.0)
#	a = 0.5
#	new_grid = g(grid,d,a)
#	
#	
#	plt.figure()
#	plt.imshow(band(grid))
#	plt.colorbar()
#	
#	plt.figure()
#	plt.imshow(band(new_grid))
#	plt.colorbar()
#
#	plt.figure()
#	plt.imshow(band.interior(grid))
#	plt.colorbar()
#
#	plt.figure()
#	plt.imshow(band.interior(new_grid))
#	plt.colorbar()
#
#	plt.show()
	