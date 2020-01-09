# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

__all__ = ['MeshBase', 'CartesianMesh',
           'StructuredNeumann', 'StructuredDirichlet', 'StructuredPML']

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
    """ Base class for structured meshes in PySIT.

    Parameters
    ----------

    domain : subclass of `pysit.core.domain.DomainBase`
        The PySIT domain that the mesh discretizes.

    configs
        A variable number of integers specifying the number of points in each
        dimension.  One entry for each dimension.

    Attributes
    ----------

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
    """ Specification of Cartesian meshes in PySIT.

    Parameters
    ----------

    domain : subclass of `pysit.core.domain.DomainBase`
        The PySIT domain that the mesh discretizes.

    configs
        A variable number of integers specifying the number of points in each
        dimension.  One entry for each dimension.

    Attributes
    ----------

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
        for (i, k) in _cart_keys[self.dim]:

            # Create the initial parameter Bunch
            # n-1 because the number of points includes both boundaries.
            n = int(configs[i])
            delta = domain.parameters[i].length / (n-1)
            param = Bunch(n=n, delta=delta)

            # Create the left and right boundary specs from the MeshBC factory
            # Note, delta is necessary for some boundary constructions, but the
            # mesh has not been instantiated yet, so it must be passed to
            # the boundary constructor.
            param.lbc = MeshBC(self, domain.parameters[i].lbc, i, 'left', delta)
            param.rbc = MeshBC(self, domain.parameters[i].rbc, i, 'right', delta)

            # access the dimension data by index, key, or shortcut
            self.parameters[i] = param  # d.dim[-1]
            self.parameters[k] = param  # d.dim['z']
            self.__setattr__(k, param)  # d.z

        # Initialize caching of mesh shapes and degrees of freedom
        self._shapes = dict()
        self._dofs = dict()

        # Cache for sparse grids.  Frequently called, so it is useful to cache
        # them, with and without boundary conditions.
        self._spgrid = None
        self._spgrid_bc = None

    def nodes(self, include_bc=False):
        """ Returns a self.dof() X self.dim `~numpy.ndarray` of node locations.

            Parameters
            ----------

            include_bc : bool, optional
                Optionally include node locations for the boundary padding,
                defaults to False.
        """
        return np.hstack(self.mesh_coords())

    def mesh_coords(self, sparse=False, include_bc=False):
        """ Returns coordinate arrays for mesh nodes.

        Makes `~numpy.ndarray` arrays for each dimension, similar to
        `~numpy.meshgrid`. Always in ([X, [Y]], Z) order.  Optionally include
        nodes due to boundary padding and optionally return sparse arrays to
        save memory.

        Parameters
        ----------

        sparse : boolean
            Returns a list of [X, [Y]], Z locations but not for each grid point, rather
            each dimesion.

        include_bc : boolean
            Optionally include physical locations of ghost/boundary points.

        """

        # If sparse and we have already generated the sparse grid, return that.
        if sparse:
            if include_bc and self._spgrid_bc is not None:
                return self._spgrid_bc
            elif self._spgrid is not None:
                return self._spgrid

        def _assemble_grid_row_bc(dim):
            p = self.parameters[dim]

            # Note, we don't use p.lbc.domain_bc.length because the PML width
            # can be longer than specified.  This occurs if the specified width
            # is not evenly divisible by the mesh delta.  It is more important
            # that the mesh delta remain constant.
            actual_pml_length_l = p.lbc.n*p.delta
            actual_pml_length_r = p.rbc.n*p.delta

            lbound = self.domain.parameters[dim].lbound - actual_pml_length_l
            rbound = self.domain.parameters[dim].rbound + actual_pml_length_r
            n = p.n + p.lbc.n + p.rbc.n
            return np.linspace(lbound, rbound, n)

        def _assemble_grid_row(dim):
            return np.linspace(self.domain.parameters[dim].lbound,
                               self.domain.parameters[dim].rbound,
                               self.parameters[dim].n)

        # Build functions for generating the x, y, and z dimension positions
        if include_bc:
            assemble_grid_row = _assemble_grid_row_bc
        else:
            assemble_grid_row = _assemble_grid_row

        # Build the rows depending on the dimension of the mesh and construct
        # the arrays using meshgrid.
        # NUMPY: When numpy 1.9, this can be reduced to one line or two, as
        # meshgrid will work with single inputs.
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
        return tuple([self.parameters[i].delta for i in range(self.dim)])

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
        for i in range(self.dim):
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
        """ Returns a view of input array, `unpadded` to the primary or
            boundary-condition- or ghost-node-free shape.

            Truncates the array, information in the padded area is discarded.

        Parameters
        ----------

        in_array : numpy.ndarray
            Input array

        copy : boolean, optional
            Optionally return a copy of the unpadded array rather than a view.

        Notes
        -----

        1. This function preserves array shape.  If the input is grid shaped,
           the return is grid shaped.  Similarly, if the input is vector shaped,
           the output will be vector shaped.

        """

        # Get the various shapes of the array without boundary conditions
        sh_unpadded_grid = self.shape(include_bc=False, as_grid=True)
        sh_unpadded_vector = self.shape(include_bc=False, as_grid=False)

        # If the array is already not padded, do nothing.
        if (in_array.shape == sh_unpadded_grid or
            in_array.shape == sh_unpadded_vector):
            out_array = in_array
        else:
            # Shape of the input array in grid form
            sh_grid = self.shape(include_bc=True, as_grid=True)

            # Build the slice into the new array (grid like for speed)
            # For each dimension, build a slice object which gathers the
            # unpadded section of the array.  Slice excludes the left and right
            # boundary nodes.
            sl = list()
            for i in range(self.dim):
                p = self.parameters[i]

                nleft = p.lbc.n
                nright = p.rbc.n

                sl.append(slice(nleft, sh_grid[i]-nright))

            # Make the input array look like a grid
            # and extract the slice is for a grid
            out_array = in_array.reshape(sh_grid)[tuple(sl)]

        # If the input shape is a vector, the return array has vector shape
        if in_array.shape[1] == 1:
            out = out_array.reshape(-1, 1)
        else:
            out = out_array

        return out.copy() if copy else out

    def pad_array(self, in_array, out_array=None, padding_mode=None):
        """ Returns a version of in_array, padded to add nodes from the
            boundary conditions or ghost nodes.

        Parameters
        ----------

        in_array : ndarray
            Input array

        out_array : ndarray, optional
            If specifed, pad into a pre-allocated array

        padding_mode : string
            Padding mode option for numpy.pad.  ``None`` indicates to pad with
            zeros (see Note 2).

        Notes
        -----

        1. ``padding_mode`` options are found in the `numpy` :func:`~numpy.pad`
           function.

        2. If ``padding_mode`` is left at its default (``None``), the array will
           be padded with zeros *without* calling :func:`~numpy.pad` and will
           instead use a faster method.

        3. Recommended value for ``padding_mode`` is 'edge' for repeating the
           edge value.

        4. This function preserves array shape.  If the input is grid shaped,
           the return is grid shaped.  Similarly, if the input is vector shaped,
           the output will be vector shaped.

        """

        # Shape of input array
        sh_in_vector = self.shape(include_bc=False, as_grid=False)
        sh_in_grid = self.shape(include_bc=False, as_grid=True)

        # Shape of the new destination array
        sh_out_vector = self.shape(include_bc=True, as_grid=False)
        sh_out_grid = self.shape(include_bc=True, as_grid=True)

        # If the output array is provided, we will need it in grid shape.
        # Plus, this excepts early if the size of the output array is wrong.
        if out_array is not None:
            out_array.shape = sh_out_grid

        # If padding_mode is not None, use numpy.pad() for padding.
        # Otherwise, pads with zeros in a faster way.  This is a necessary
        # optimization.
        if padding_mode is not None:
            _pad_tuple = tuple([(self.parameters[i].lbc.n, self.parameters[i].rbc.n) for i in range(self.dim)])
            _out_array = np.pad(in_array.reshape(sh_in_grid), _pad_tuple, mode=padding_mode).copy()

            # If the output memory is allocated, copy padded array into it.
            if out_array is not None:
                out_array[:] = _out_array[:]
            else:
                out_array = _out_array
        else:
            # Allocate the destination array
            if out_array is None:
                out_array = np.zeros(sh_out_grid, dtype=in_array.dtype)

            # Build the slice of the unpadded array into the new array.
            # For each dimension, build a slice object which gathers the
            # unpadded section of the output array.  Slice excludes the left
            # and right boundary nodes.
            sl = list()
            for i in range(self.dim):
                p = self.parameters[i]

                nleft = p.lbc.n
                nright = p.rbc.n

                sl.append(slice(nleft, sh_out_grid[i]-nright))

            # Copy the source array
            out_array[tuple(sl)] = in_array.reshape(sh_in_grid)

        # If the input shape is a vector, the return array has vector shape
        if in_array.shape == sh_in_vector:
            out_array.shape = sh_out_vector
        else:
            out_array.shape = sh_out_grid
        return out_array

    def inner_product(self, arg1, arg2):
        """ Compute the correct scaled inner product on the mesh."""

        return np.dot(arg1.T, arg2).squeeze() * np.prod(self.deltas)


class UnstructuredMesh(MeshBase):
    """ [NotImplemented] Base class for specifying unstructured meshes in
    PySIT.
    """

    pass


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
            return super(cls).__new__(cls)


class MeshBCBase(object):
    """ Base class for mesh boundary conditions.

    Parameters
    ----------

    mesh : subclass of pysit.core.mesh.MeshBase

    domain_bc : subclass of pysit.core.domain.DomainBC

    """
    def __init__(self, mesh, domain_bc, *args, **kwargs):
        self.mesh = mesh
        self.domain_bc = domain_bc


class StructuredBCBase(MeshBCBase):
    """ Base class for mesh boundary conditions on structured meshes.

    Parameters
    ----------

    mesh : subclass of pysit.core.mesh.MeshBase

    domain_bc : subclass of pysit.core.domain.DomainBC

    dim : int
        Dimension number of the boundary condition.  See Note 1.

    side : {'left', 'right'}
        Side of the domain that the boundary condition applies to.

    Notes
    -----

    1. ``dim`` is the dimension number corresponding to x, y, or z, **not** the
       spatial dimension that the problem lives in.

    """

    # The distinction between ``type`` and ``boundary_type`` is most clear for
    # absorbing boundaries.  Currently, the domain boundaries are somewhat
    # poorly named as PML, when they are really meant to be general absorbing
    # boundaries which can be implemented in an arbitrary way.
    # ``boundary_type`` specifies the physical interpretation of the boundary
    # condition and ``type`` specifies the discrete implementation.

    @property
    def type(self):
        """The type of discrete implementation of the boundary condition. """
        return None

    @property
    def boundary_type(self):
        """The physical type of boundary condition. """
        return None

    def __init__(self, mesh, domain_bc, dim, side, *args, **kwargs):

        MeshBCBase.__init__(self, mesh, domain_bc, *args, **kwargs)

        self._n = 0
        self.solver_padding = 1
        self.dim = dim
        self.side = side

    n = property(lambda self: self._n,
                 None,
                 None,
                 "Number of padding nodes on the boundary.")


class StructuredDirichlet(StructuredBCBase):
    """Specification of the Dirichlet boundary condition on structured meshes.

    Parameters
    ----------

    mesh : subclass of pysit.core.mesh.MeshBase

    domain_bc : subclass of pysit.core.domain.DomainBC

    dim : int
        Dimension number of the boundary condition.  See Note 1.

    side : {'left', 'right'}
        Side of the domain that the boundary condition applies to.

    Notes
    -----

    1. ``dim`` is the dimension number corresponding to x, y, or z, **not** the
       spatial dimension that the problem lives in.

    """

    @property
    def type(self):
        """The type of discrete implementation of the boundary condition. """
        return 'dirichlet'

    @property
    def boundary_type(self):
        """The physical type of boundary condition. """
        return 'dirichlet'


class StructuredNeumann(StructuredBCBase):
    """Specification of the Neumann boundary condition on structured meshes.

    Parameters
    ----------

    mesh : subclass of pysit.core.mesh.MeshBase

    domain_bc : subclass of pysit.core.domain.DomainBC

    dim : int
        Dimension number of the boundary condition.  See Note 1.

    side : {'left', 'right'}
        Side of the domain that the boundary condition applies to.

    Notes
    -----

    1. ``dim`` is the dimension number corresponding to x, y, or z, **not** the
       spatial dimension that the problem lives in.

    """

    @property
    def type(self):
        """The type of discrete implementation of the boundary condition. """
        return 'neumann'

    @property
    def boundary_type(self):
        """The physical type of boundary condition. """
        return 'neumann'


class StructuredPML(StructuredBCBase):
    """Specification of the PML-absorbing boundary condition on structured meshes.

    Parameters
    ----------

    mesh : subclass of pysit.core.mesh.MeshBase

    domain_bc : subclass of pysit.core.domain.DomainBC

    dim : int
        Dimension number of the boundary condition.  See Note 1.

    side : {'left', 'right'}
        Side of the domain that the boundary condition applies to.

    delta : float
        Mesh spacing

    Notes
    -----

    1. ``dim`` is the dimension number corresponding to x, y, or z, **not** the
       spatial dimension that the problem lives in.

    """

    @property
    def type(self):
        """The type of discrete implementation of the boundary condition. """
        return 'pml'

    @property
    def boundary_type(self):
        """The physical type of boundary condition. """
        return self._boundary_type

    def __init__(self, mesh, domain_bc, dim, side, delta, *args, **kwargs):

        StructuredBCBase.__init__(self, mesh, domain_bc, dim, side, delta, *args, **kwargs)

        pml_width = domain_bc.length

        self._n = int(np.ceil(pml_width / delta))

        # Sigma is the evaluation of the profile function on n points over the
        # range [0, 1].  The extra +1 is because _n *excludes* the original
        # boundary node.  However, the length of the PML is counted directly
        # from this boundary node.
        # For example, consider the discretized domain below. Let * be a node
        # and @ be a boundary node.  Assume the domain length is 5, so the
        # spacing is 1.
        # @----*----*----*----*----@
        # When the domain is extended for the PML, we get something like the
        # following, where o is a node added in the PML.
        # o----o----o----@----*----*----*----*----@----o----o----o
        # The PML profile function must be evaluated from [0, 1] and the PML
        # technically begins on @.  That is, from the mapping of [0, 1] onto
        # the PML region, @ = 0 and the right-most o = 1.  self._n would take
        # the value 3 in this example, not 4.  So we give the evaluation
        # function an extra node to work with, to ensure that the delta is
        # correct, then we take that extra node off of the sigma function.
        s = domain_bc.evaluate(self._n+1, side)

        if side == 'right':
            self.sigma = s[1:]
        elif side == 'left':
            self.sigma= s[:-1]

        # Get the physical boundary type
        self._boundary_type = domain_bc.boundary_type

    def eval_on_mesh(self):
        """ Evaluates the PML profile function sigma on the mesh.  Returns an
        array the size of the mesh with the PML function evalauted at each node.
        """

        sh_dof = self.mesh.shape(include_bc=True, as_grid=False)
        sh_grid = self.mesh.shape(include_bc=True, as_grid=True)

        out_array = np.zeros(sh_grid)

        sl_block = list()

        if self.side == 'left':
            nleft = self._n

            for k in range(self.mesh.dim):
                if self.dim != k:
                    s = slice(None)
                else:
                    s = slice(0, nleft)
                sl_block.append(s)

        else:  # self.side == 'right'
            nright = self._n

            for k in range(self.mesh.dim):
                if self.dim != k:
                    s = slice(None)
                else:
                    s = slice(out_array.shape[k]-nright, None)
                sl_block.append(s)

        sl_block = tuple(sl_block)

        # Get the shape of sigma in the appropriate dimension
        sh = list()
        for k in range(self.mesh.dim):
            if self.sigma.shape[0] == out_array[sl_block].shape[k]:
                sh.append(-1)
            else:
                sh.append(1)
        sh = tuple(sh)

        # Use numpy broadcasts to copy sigma throught the correct block.
        out_array[sl_block] = self.sigma.reshape(sh)

        return out_array.reshape(sh_dof)


class StructuredGhost(StructuredBCBase):
    """Specification of the ghost-node padding as a boundary condition on
    structured meshes.

    Parameters
    ----------

    mesh : subclass of pysit.core.mesh.MeshBase

    domain_bc : subclass of pysit.core.domain.DomainBC

    dim : int
        Dimension number of the boundary condition.  See Note 1.

    side : {'left', 'right'}
        Side of the domain that the boundary condition applies to.

    ghost_padding : int
        Number of ghost nodes.

    Notes
    -----

    1. ``dim`` is the dimension number corresponding to x, y, or z, **not** the
       spatial dimension that the problem lives in.

    """

    @property
    def type(self):
        """The type of discrete implementation of the boundary condition. """
        return 'ghost'

    @property
    def boundary_type(self):
        """The physical type of boundary condition. """
        return 'ghost'

    def __init__(self, mesh, domain_bc, dim, side, ghost_padding, *args, **kwargs):

        StructuredBCBase.__init__(self, mesh, domain_bc, dim, side, ghost_padding, *args, **kwargs)

        self.solver_padding = ghost_padding

    n = property(lambda self: self.ghost_padding,
                 None,
                 None,
                 "Number of padding nodes on the boundary.")


class UnstructuredBCBase(MeshBCBase):
    """ [NotImplemented] Base class for specifying boundary conditions on
    unstructured meshes in PySIT.
    """
