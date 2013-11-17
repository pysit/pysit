import numpy as np
import scipy.sparse as spsp
from scipy.interpolate import interp1d

__all__ = ['MeshRepresentationBase', 'PointRepresentationBase']

class MeshRepresentationBase(object):
	""" Base class for representing an object on a (Cartesian) grid.
	
	This class serves as a base class for physical objects, such as points
	and planes that model receivers and emitters.
	
	Attributes
	----------
	mesh : pysit.Mesh
		Physical domain on which the source is defined.
	domain : pysit.Domain
		Physical domain on which the source is defined.
	sampling_operator : scipy.sparse matrix or numpy.ndarray
		Linear operator describing how this object is represented on a mesh.
	adjoint_sampling_operator : scipy.sparse matrix or numpy.ndarray
		The adjoint of the sampling operator.
	deltas : list of float
		Mesh spacings in each dimension.
		
	Notes
	-----
	`sampling_operator` is always stored as a single row.  Thus, the domain grid is flat
	in 'C' ordering.  This is so that we can easily stack them into 'operators'
	for cases when there are multiple sources or receivers in a set.
	
	"""
	
	def __init__(self, mesh, **kwargs):
		"""Constructor for the MeshRepresentationBase class.
		
		Parameters
		----------
		mesh : pysit.Mesh
			Physical domain on which the source is defined.
		"""
		
		self.mesh = mesh
		self.domain = mesh.domain
		
		# This is likely to be cut...
		# It won't be cut for now, but things will definitely have to be
		# rethought when a more general mesh is introduced.
		self.deltas = mesh.deltas
		self._prod_deltas = np.prod(self.deltas)
		self._max_deltas = np.max(self.deltas)
		
		# This must be specified by subclass.
		# The sampling operator and its adjoint are defined by 
		# <d, Su>_D = <S*d,u>_W, so d(t) = Su(x,t) = \int_\mathcal{R} u(x,t)\delta(x-x_r)dx = u(x_r,t)
		# and thus S*d(t) = \delta(x-x_r)d(t), because the W inner product is over space
		self.sampling_operator = None
		self.adjoint_sampling_operator = None

		# From empirical testing, for small 1D problems, dense computation
		# is much faster.  For large ones, the sparse LA sampling is faster than
		# dense linear algebra for sampling.  In higher dimensions, sparse
		# always appears to be faster.  Note, for single source case in higher
		# dimensions, dense seems to be faster, but I am not sure what the
		# marginal improvement will be, so that must be investigated later.
		# This may only hold for smaller grids too, so that has to be chekced
		# # out.
		# if domain.dim == 1:	
			# self._interp_method = 'dense'	
			# if domain.nx >= 10000:
				# self._sample_method = 'sparse'
			# else:
				# self._sample_method = 'dense'
		# else:
			# self._interp_method = 'sparse'
			# self._sample_method = 'sparse'
			
		# In deference to not wanting to have two different sampling_operator variables
		# poluting the namespace, we can ignore the above for now and just
		# force dense math for all 1D problems.  It won't matter that much in
		# the long run, I don't think.
		if self.domain.dim == 1:	
			self._sample_interp_method = 'dense'	
		else:
			self._sample_interp_method = 'sparse'
			
		# There, perhaps, should be a _sample and _interpolate routine here so 
		# that they are not perpetually reimplemented.  But maybe not.
	
class PointRepresentationBase(MeshRepresentationBase):
	""" Base class for representing a point (or delta) on a grid.
	
	This class serves as a base class for physical objects, such as receivers
	and source emitters that are usually represented in the domain as volumeless
	points.
	
	Attributes
	----------
	approximation : {'gaussian', 'delta'}
		Method for approximating delta distribution numerically.
	approximation_width : int
		Standard deviation of the Gaussian approximation.	
		
	Notes
	-----

	* np.sum(self._sampling_operator_base) does not equal 1, as the domain will not generally
	  have a unit spaced grid.  However, 
	  np.sum(self._sampling_operator_base)*prod(self.domain.deltas) will equal 1, thus 
	  preserving the integral of the delta distribution on the specified domain.
	
	"""
#	@profile
	def __init__(self, mesh, pos, approximation='gaussian', approximation_width=1, approximation_deviations=3, **kwargs):
		"""Constructor for the SeismicPointBase class.
		
		Parameters
		----------
		domain : pysit.Domain
			Physical (and numerical) domain on which the source is defined.
		position : tuple of float
			Coordinates of the point in the physical coordinates of the domain.
		approximation : {'gaussian', 'delta'}, optional
			Method for approximating delta distribution numerically.
		approximation_width : float, optional
			Standard deviation of the Gaussian approximation.	
		approximation_deviations : float, optional
			Number of standard deviations to use in the Gaussian approximation.		
		"""
		
		MeshRepresentationBase.__init__(self, mesh, **kwargs)
		
		# Sanitize the input some
		if ((self.domain.dim == 1) and (type(pos) is not tuple)):
			pos = pos,
		
		self.position = pos
				
		self.approximation = approximation
		self.approximation_width = approximation_width
		self.approximation_deviations = approximation_deviations
		
		# this will likely have to branch on mesh.type when/if a more general mesh is introduced
		if(approximation == 'gaussian'):
			deltas = self.deltas
			mins = self.domain.collect('lbound')
			
			grid = self.mesh.mesh_coords(sparse=True)
			threshold = 1e-7
			
			# This gives a cleaner way to represent the width of the Gaussian.
			# The std. dev. is the maximum grid spacing.
			sigma = self._max_deltas*approximation_width
			
			# round to three decimals to get around very small floating point errors
			window_width = np.ceil(np.around(approximation_deviations*sigma / np.array(deltas), 3)).astype(np.int)

			# (Position - boundary) / grid spacing, for each direction
			gc = map(lambda x,y,z: int(round((z - x)/y)), mins, deltas, pos)
			
			subranges = [range(max(0,c-w),min(g.size,c+w+1)) for g,w,c in zip(grid,window_width,gc)]
			subwindows = [g.flatten()[r] for g,r in zip(grid,subranges)]
			
			dim = self.domain.dim
			
			if dim==1:
				subranges = (np.array(subranges[0]),)
				subgrid = (np.array(subwindows[0]),)
			else:
				subranges = np.meshgrid(*subranges)
				subgrid = np.meshgrid(*subwindows)
			
			normalization = 1./((np.sqrt(2*np.pi)**dim)*(sigma**dim))
			# reduce(X, map(Y, (Grid,Pos)))	nicely handles both 2D and 3D gaussians
			data = normalization * np.exp( (-0.5/sigma**2) * reduce(lambda x,y: x+y, map(lambda x: (x[0]-x[1])**2, zip(subgrid,pos))))
			indices = np.ravel_multi_index(np.array([g.flatten() for g in subranges]).astype(np.int), self.mesh.shape(as_grid=True))
			if self._sample_interp_method == 'sparse':
				indptr = np.array([0,len(indices)])
				self._sampling_operator_base = spsp.csr_matrix((data.flatten(), indices, indptr), shape=(1,self.mesh.dof()))
			else:
				self._sampling_operator_base = np.zeros((1,self.mesh.dof()))
				self._sampling_operator_base[0,indices] = data.flatten()
				
		elif(approximation == 'delta'):
			deltas = self.deltas
			mins = self.domain.collect('lbound')
			
			# (Position - boundary) / grid spacing, for each direction
			gc = tuple(map(lambda x: int(round((x[2] - x[0])/x[1])), zip(mins, deltas, pos)))
						
			# Get a flat index for that grid coordinate
			grid_coord = np.ravel_multi_index(gc,self.mesh.shape(as_grid=True))
			
			if self._sample_interp_method == 'sparse':
				# Linked list matrix for insertion
				sz = (1, self.mesh.dof())
				self._sampling_operator_base = spsp.lil_matrix(sz)
				# self._sampling_operator_base = np.zeros(self.domain.shape)
				self._sampling_operator_base[0,grid_coord] = 1.0 / self._prod_deltas
				# CSR matrix for manipulation
				self._sampling_operator_base = self._sampling_operator_base.tocsr()
			else:
				self._sampling_operator_base = np.zeros((1,self.mesh.dof()))
				self._sampling_operator_base[0,grid_coord] = 1.0 / self._prod_deltas
			
		else:
			raise ValueError("Only 'delta' and 'gaussian' are valid approximations for a point source or receiver.")	
		
		
		self.sampling_operator = self._sampling_operator_base * self._prod_deltas
		self.adjoint_sampling_operator = self._sampling_operator_base.T
		
class PlaneRepresentationBase(MeshRepresentationBase):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError()
