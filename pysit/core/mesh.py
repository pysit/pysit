# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

__all__ = ['MeshBase', 'CartesianMesh', 'StructuredNeumann', 'StructuredDirichlet', 'StructuredPML']

# Mapping between dimension and the key labels for Cartesian domain
_cart_keys = {1: [(0, 'z')],
              2: [(0, 'x'), (1, 'z')],
              3: [(0, 'x'), (1, 'y'), (2, 'z')]}

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

class MeshBase(object):
	""" Base Class for Pysit mesh objects"""

	@property
	def type(self):
		""" String describing the type of mesh, e.g., structured."""
		return None

	def nodes(self, *args, **kwargs):
		""" Returns a position of mesh nodes as an self.dof() X self.dim array. """
		raise NotImplementedError('Must be implemented by subclass.')

	def edges(self, *args, **kwargs):
		raise NotImplementedError('Must be implemented by subclass.')

	def shape(self, *args, **kwargs):
		""" Returns the shape of the mesh. """
		raise NotImplementedError('Must be implemented by subclass.')

	def dof(self, *args, **kwargs):
		""" Returns number of degrees of freedom."""
		raise NotImplementedError('Must be implemented by subclass.')

	def inner_product(self, arg1, arg2):
		""" Implements inner product on the mesh, accounts for scaling."""
		raise NotImplementedError('Must be implemented by subclass.')

class StructuredMesh(MeshBase):
	""" Base class for structured meshses in PySIT.

	Parameters
	----------

	domain : subclass of `pysit.core.domain.DomainBase`
		The PySIT domain that the mesh discretizes.

	configs
		A variable number of integers specifying the number of points in each
		dimension.  One entry for each dimension.

	Attribute
	---------

	type : str
		class attribute, identifies the type of mesh as 'structured'

	dim : int
		Dimension :math:`d` in :math:`\mathcal{R}^d` of the domain.

	domain : subclass of `pysit.core.domain.DomainBase`
		The PySIT domain that the mesh discretizes.

	"""

	@property
	def type(self):
		""" String describing the type of mesh, e.g., structured."""
		return 'structured'

	def __init__(self, domain, *configs):

		self.domain = domain

		if (len(configs) > 3) or (len(configs) < 1):
			raise ValueError('Mesh dimension must be between 1 and 3.')

		if len(configs) != domain.dim:
			raise ValueError('Mesh dimension must match domain dimension.')

		self.dim = domain.dim

class CartesianMesh(StructuredMesh):
	""" Specification of Cartesian meshses in PySIT.

	Parameters
	----------

	domain : subclass of `pysit.core.domain.DomainBase`
		The PySIT domain that the mesh discretizes.

	configs
		A variable number of integers specifying the number of points in each
		dimension.  One entry for each dimension.

	Attribute
	---------

	type : str
		class attribute, identifies the type of mesh as 'structured-cartesian'

	dim : int
		Dimension :math:`d` in :math:`\mathcal{R}^d` of the domain.

	domain : subclass of `pysit.core.domain.DomainBase`
		The PySIT domain that the mesh discretizes.

	parameters : dict of Bunch
		Dictionary containing descriptions of each dimension.

	Notes
	-----

	1. In any iterable which depends on dimension, the z-dimension is always the
	   *last* element.  Thus, in 1D, the dimension is assumed to be z.

	2. The negative direction is always referred to as the *left* side and the
	   positive direction is always the *right* side.  For example, for the
	   z-dimension, the top is left and the bottom is right.

	3. A dimension parameters Bunch contains four keys:
		1. `n`: an integer number of points for the dimension
		2. `delta`: the distance between points
		3. `lbc`: the mesh description of the left boundary condition
		4. `rbc`: the mesh description of the the right boundary condition

	4. The parameters dictionary can be accessed by number, by letter, or in
	   the style of an attribute of the `~pysit.core.domain.RectangularDomain`.

		1. *Number*

		    >>> # Assume 3D, z is last
		    >>> my_mesh.parameters[2]

		2. *Letter*

			>>> my_mesh.parameters['z']

		3. *Attribute*

			>>> my_mesh.z

	"""

	@property
	def type(self):
		""" String describing the type of mesh, e.g., structured."""
		return 'structured-cartesian'

	def __init__(self, domain, *configs):

		# Initialize the base class
		StructuredMesh.__init__(self, domain, *configs)

		self.parameters = dict()

		# Loop over each specified dimesion
		for (i,k) in _cart_keys[self.dim]:

			# Create the initial parameter Bunch
			n = int(configs[i])
			delta = domain.parameters[i].length / n
			param = Bunch(n=n, delta=delta)

			# Create the left and right boundary specs from the MeshBC factory
			param.lbc = MeshBC(self, domain.parameters[i].lbc, i, 'left', delta)
			param.rbc = MeshBC(self, domain.parameters[i].rbc, i, 'right', delta)

			# access the dimension data by index, key, or shortcut
			self.parameters[i] = param # d.dim[-1]
			self.parameters[k] = param # d.dim['z']
			self.__setattr__(k, param) # d.z


		# Initialize caching of mesh shapes and degrees of freedom
		self._shapes = dict()
		self._dofs = dict()

		# Cache for sparse grids.  Frequently called, so it is useful to cache
		# them, with and without boundary conditions.
		self._spgrid = None
		self._spgrid_bc = None

	def nodes(self, include_bc=False):
		""" Returns a self.dof() X self,.dim `~numpy.ndarray` of node locations.

			Parameters
			----------

			include_bc : bool, optional
		    	Optionally include node locations for the boundary padding,
		    	defaults to False.
		"""
		return np.hstack(self.mesh_coords())

	def mesh_coords(self, sparse=False, include_bc=False):
		""" Returns coordinate arrays for mesh nodes.

		Makes `~numpy.ndarray` arrays for each dimension, similar to `meshgrid`.
		Always in ([X, [Y]], Z) order.  Optionally include nodes due to boundary
		padding and optionally return sparse arrays to save memory.

		Parameters
		----------

		sparse : boolean
			Returns a list of [X, [Y]], Z locations but not for each grid point, rather
			each dimesion.

		include_bc : boolean
			Optionally include physical locations of ghost/boundary points.

		See Also
		--------

		`numpy.meshgrid`

		"""

		# If sparse and we have already generated the sparse grid, return that.
		if sparse:
			if include_bc and self._spgrid_bc is not None:
				return self._spgrid_bc
			elif self._spgrid is not None:
				return self._spgrid


		def _assemble_grid_row_bc(dim):
			p = self.parameters[dim]
			lbound = self.domain.parameters[dim].lbound - p.lbc.n*p.delta
			rbound = self.domain.parameters[dim].rbound + p.rbc.n*p.delta
			n = p.n + p.lbc.n + p.rbc.n
			return np.linspace(lbound, rbound, n, endpoint=False)

		def _assemble_grid_row(dim):
			return np.linspace(self.domain.parameters[dim].lbound,
				               self.domain.parameters[dim].rbound,
				               self.parameters[dim].n,
				               endpoint=False)

		# Build functions for generating the x, y, and z dimension positions
		if include_bc:
			assemble_grid_row = _assemble_grid_row_bc
		else:
			assemble_grid_row = _assemble_grid_row

		# Build the rows depending on the dimension of the mesh and construct
		# the arrays using meshgrid.
		if(self.dim == 1):
			# return value is a 1-tuple, created in python by (foo,)
			tup = tuple([assemble_grid_row('z')])
		elif(self.dim == 2):
			xrow = assemble_grid_row('x')
			zrow = assemble_grid_row('z')
			tup = np.meshgrid(xrow, zrow, sparse=sparse, indexing='ij')
		else:
			xrow = assemble_grid_row('x')
			yrow = assemble_grid_row('y')
			zrow = assemble_grid_row('z')
			tup = np.meshgrid(xrow, yrow, zrow, sparse=sparse, indexing='ij')

		# Cache the grid if needed
		if sparse:
			if not include_bc and self._spgrid is None:
				self._spgrid = tup
			if include_bc and self._spgrid_bc is None:
				self._spgrid_bc = tup

		if not sparse:
			tup = tuple([x.reshape(self.shape(include_bc)) for x in tup])

		return tup


	@property
	def deltas(self):
		"Tuple of grid deltas"
		return tuple([self.parameters[i].delta for i in xrange(self.dim)])

	def _compute_shape(self, include_bc):
		""" Precomputes the shape of a mesh, both as a grid and as a vector.

		The shape as a grid means the result is a tuple containing the number
		of nodes in each dimension.  As a vector means a column vector (an nx1
		`~numpy.ndarray`) where n is the total degrees of freedom.

		Parameters
		----------

		include_bc : bool
			Indicates if the boundary padding is included.

		"""

		sh = []
		for i in xrange(self.dim):
			p = self.parameters[i]

			n = p.n
			if include_bc:
				n += p.lbc.n
				n += p.rbc.n

			sh.append(n)

		# self._shapes dict has a tuple for an index: (include_bc, as_grid)
		self._shapes[(include_bc, True)] = sh
		self._shapes[(include_bc, False)] = (int(np.prod(np.array(sh))), 1)

		# precompute the degrees of freedom dict too
		self._dofs[include_bc] = int(np.prod(np.array(sh)))

	def shape(self, include_bc=False, as_grid=False):
		""" Return the shape, the number of nodes in each dimension.

		The shape as a grid means the result is a tuple containing the number
		of nodes in each dimension.  As a vector means a column vector, so shape
		is nx1, where n is the total degrees of freedom.

		Parameters
		----------

		include_bc : bool, optional
			Include the ghost/boundary condition nodes in the shape; defaults
			to False.

		as_grid : bool, optional
			Return the shape as a self.dim tuple of nodes in each dimension,
			rather than the column vector shape (self.dof(),1); defaults to
			False.

		"""

		# If the shape has not been computed for this combination of parameters,
		# compute it.
		if (include_bc, as_grid) not in self._shapes:
			self._compute_shape(include_bc)

		return self._shapes[(include_bc, as_grid)]

	def dof(self, include_bc=False):
		""" Return the number of degrees of freedom as an integer.

		Parameters
		----------

		include_bc : bool, optional
			Include the ghost/boundary condition nodes in the shape; defaults
			to False.
		"""

		# If the dof has not been computed, compute it.
		if include_bc not in self._dofs:
			self._compute_shape(include_bc)

		return self._dofs[include_bc]

	def unpad_array(self, in_array, copy=False):
		""" Returns a view of in_array, unpadded to the primary or boundary-condition- or ghost-node-free shape.

		Parameters
		----------
		in_array : ndarray
			Input array
		copy : boolean
			Return a copy of the unpadded array
		"""

		sh_unpadded_grid = self.shape(include_bc=False, as_grid=True)
		sh_unpadded_dof = self.shape(include_bc=False, as_grid=False)

		# short circuit the unpad operation if the array is not padded
		if in_array.shape == sh_unpadded_grid or in_array.shape == sh_unpadded_dof:
			out_array = in_array
		else:
			# shape of the input array in grid form
			sh_grid = self.shape(include_bc=True, as_grid=True)

			# Build the slice into the new array (grid like for speed)
			sl = list()
			for i in xrange(self.dim):
				p = self.parameters[i]

				nleft  = p.lbc.n
				nright = p.rbc.n

				sl.append(slice(nleft, sh_grid[i]-nright))

			# Make the input array look like a grid, the slice is for a grid
			out_array = in_array.reshape(sh_grid)[sl]

		if in_array.shape[1] == 1:
			# Return in DOF shape
			out = out_array.reshape(-1,1)
		else:
			out = out_array

		return out.copy() if copy else out

	def pad_array(self, in_array, out_array=None, padding_mode=None):
		""" Returns a version of in_array, padded to add nodes from the boundary conditions or ghost nodes.

		Parameters
		----------
		in_array : ndarray
			Input array
		out_array : ndarray, optional
			If specifed, pad into a pre-allocated array
		padding_mode : string
			Padding mode option for numpy.pad.
		"""

		# Shape of the new destination array
		sh_dof  = self.shape(True, False)
		sh_grid = self.shape(True, True)
		sh_in_grid = self.shape(False,True)
		# Build the slice into the new array (grid like for speed)
		sl = list()
		for i in xrange(self.dim):
			p = self.parameters[i]

			nleft  = p.lbc.n
			nright = p.rbc.n

			sl.append(slice(nleft, sh_grid[i]-nright))

		# Allocate the destination array and copy the source array to it
		if out_array is not None:
			out_array.shape = sh_grid
		else:
			out_array = np.zeros(sh_grid, dtype=in_array.dtype)
		out_array[sl] = in_array.reshape(sh_in_grid)


		# Build the padding tuples and pad the matrix
		# The following if block is equivalent to the above call, but much
		# slower than just allocationg a zero array as above, if the padding
		# method is indeed zeros
		if padding_mode is not None:
			_pad_tuple = tuple([(self.parameters[i].lbc.n, self.parameters[i].rbc.n) for i in xrange(self.dim)])
			out_array = np.pad(in_array.reshape(sh_in_grid), _pad_tuple, mode=padding_mode).copy()


		if in_array.shape[1] == 1: # Does not guarantee dof shaped, but suggests it.
			out_array.shape = sh_dof
		else:
			out_array.shape = sh_grid
		return out_array

	def inner_product(self, arg1, arg2):
		""" Compute the correct scaled inner product on the mesh."""

		return np.dot(arg1.T, arg2).squeeze() * np.prod(self.deltas)


class UnstructuredMesh(MeshBase):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError()

class MeshBC(object):
	""" Factory class for mesh boundary conditions. """

	def __new__(cls, mesh, domain_bc, *args, **kwargs):

		if cls is MeshBC:
			if 'structured' in mesh.type:
				if domain_bc.type is 'dirichlet':
					mesh_bc = StructuredDirichlet
				if domain_bc.type is 'neumann':
					mesh_bc = StructuredNeumann
				if domain_bc.type is 'pml':
					mesh_bc = StructuredPML
				if domain_bc.type is 'ghost':
					mesh_bc = StructuredGhost

			return mesh_bc(mesh, domain_bc, *args, **kwargs)
		else:
			return super(MeshBC, cls).__new__(cls, mesh, domain_bc, *args, **kwargs)



class MeshBCBase(object):
	""" Base class for mesh boundary conditions. """
	def __init__(self, mesh, domain_bc, *args, **kwargs):
		self.mesh = mesh
		self.domain_bc = domain_bc

class StructuredBCBase(MeshBCBase):
	""" Base class for mesh boundary conditions on structured meshes."""
	def __init__(self, mesh, domain_bc, dim, side, *args, **kwargs):
		MeshBCBase.__init__(self, mesh, domain_bc, *args, **kwargs)
		self._n = 0
		self.solver_padding = 1
		self.dim = dim
		self.side = side

	n = property(lambda self: self._n, None, None, None)

class StructuredDirichlet(StructuredBCBase):
	"""Specification of the Dirichlet boundary condition on structured meshes."""

	@property
	def type(self): return 'dirichlet'
	@property
	def boundary_type(self): return 'dirichlet'

class StructuredNeumann(StructuredBCBase):
	"""Specification of the Neumann boundary condition on structured meshes."""

	@property
	def type(self): return 'neumann'
	@property
	def boundary_type(self): return 'neumann'

class StructuredPML(StructuredBCBase):
	"""Specification of the PML-absorbing boundary condition on structured meshes."""

	@property
	def type(self): return 'pml'

	def __init__(self, mesh, domain_bc, dim, side, delta, *args, **kwargs):
		""" Constructor for PML-absorbing boundary conditions on structured meshes.

		Parameters
		----------
		mesh : pysit mesh
			Mesh for the boundary condition
		domain_bc : PySIT domain boundary condition object
			Physical specification of the boundary condition
		dim : int
			Dimension number for the BC.
		side : str, {'left', 'right'}
			Side at which PML is applied.
		delta : float
			Mesh spacing

		"""


		StructuredBCBase.__init__(self, mesh, domain_bc, dim, side, delta, *args, **kwargs)

		pml_width = domain_bc.length

		self._n = int(np.ceil(pml_width / delta))
		self.sigma = domain_bc.evaluate(self._n, side)
		self.boundary_type = domain_bc.boundary_type

	def eval_on_mesh(self):
		""" Evaluates the PML profile function sigma on the mesh."""

		sh_dof  = self.mesh.shape(True, False)
		sh_grid = self.mesh.shape(True, True)
		out_array = np.zeros(sh_grid)

		if self.side == 'left':
			nleft = self._n
			sl_block = tuple([slice(None) if self.dim!=k else slice(0,nleft) for k in xrange(self.mesh.dim)])
		else: # right
			nright = self._n
			sl_block = tuple([slice(None) if self.dim!=k else slice(out_array.shape[k]-nright,None) for k in xrange(self.mesh.dim)])

		sh = tuple([-1 if self.sigma.shape[0]==out_array[sl_block].shape[k] else 1 for k in xrange(self.mesh.dim)])
		out_array[sl_block] = self.sigma.reshape(sh)

		return out_array.reshape(sh_dof)

class StructuredGhost(StructuredBCBase):
	@property
	def type(self): return 'ghost'
	@property
	def boundary_type(self): return 'ghost'

	def __init__(self, mesh, domain_bc, ghost_padding, *args, **kwargs):

		StructuredBCBase.__init__(self, mesh, domain_bc, ghost_padding, *args, **kwargs)
		self.solver_padding = ghost_padding

	n = property(lambda self: self.ghost_padding, None, None, None)

class UnstructuredBCBase(MeshBCBase):
	pass