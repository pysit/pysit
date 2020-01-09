# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

__all__ = ['DomainBase', 'RectangularDomain', 'DomainBC', 'Neumann', 'Dirichlet', 'PML']

# Mapping between dimension and the key labels for Cartesian domain
_cart_keys = {1: [(0, 'z')],
              2: [(0, 'x'), (1, 'z')],
              3: [(0, 'x'), (1, 'y'), (2, 'z')]
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


class DomainBase(object):
    """ Base class for `pysit` physical domains. """

    type = None


class RectangularDomain(DomainBase):
    """ Class for describing rectangular domains in `pysit`.

    Parameters
    ----------

    configs
        A variable number of config tuples, see Notes for details. One tuple for
        each desired problem dimension.

    Attributes
    ----------

    type : str
        class attribute, identifies the type of domain as 'rectangular'

    dim : int
        Dimension :math:`d` in :math:`\mathcal{R}^d` of the domain.

    parameters : dict of Bunch
        Dictionary containing descriptions of each dimension.

    Notes
    -----

    1. A `config` tuple is a 4-tuple (or 5-tuple) with the following elements:
        1. left boundary position
        2. right boundary position
        3. left boundary condition
        4. right boundary condition
        5. unit, (optional)

       The left and right boundary positions are in physical coordinates.  The
       left and right boundary conditions are instances of the subclasses of
       `pysit.core.domain.DomainBC`.

    2. The coordinate system is stored in a left-handed ordering (the positive
       z-direction points downward).

    3. In any iterable which depends on dimension, the z-dimension is always the
       *last* element.  Thus, in 1D, the dimension is assumed to be z.

    4. The negative direction is always referred to as the *left* side and the
       positive direction is always the *right* side.  For example, for the
       z-dimension, the top is left and the bottom is right.

    5. A dimension parameters Bunch contains six keys:
        1. `lbound`: a float with the closed left boundary of the domain
        2. `rbound`: a float with the open right boundary of the domain
        3. `lbc`: the left boundary condition
        4. `rbc`: the right boundary condition
        5. `unit`: a string with the physical units of the dimension, e.g., 'm'
        6. `length`: a float with the length of the domain, `rbound`-`lbound`

    6. The parameters dictionary can be accessed by number, by letter, or in
       the style of an attribute of the `~pysit.core.domain.RectangularDomain`.

        1. *Number*

            >>> # Assume 3D, z is last
            >>> my_domain.parameters[2]

        2. *Letter*

            >>> my_domain.parameters['z']

        3. *Attribute*

            >>> my_domain.z

    """

    type = 'rectangular'

    def __init__(self, *configs):

        if (len(configs) > 3) or (len(configs) < 1):
            raise ValueError('Domain construction must be between one and three dimensions.')

        self.dim = len(configs)

        self.parameters = dict()

        # Setup access in the parameters dict by both letter and dimension number
        for (i,k) in _cart_keys[self.dim]:

            config = configs[i]

            if len(config) == 4:
                L, R, lbc, rbc = config # min, max, left BC, right BC
                unit = None
            elif len(config) == 5:
                L, R, lbc, rbc, unit = config # min, max, left BC, right BC, unit

            param = Bunch(lbound=float(L), rbound=float(R), lbc=lbc, rbc=rbc, unit=unit)

            param.length = float(R) - float(L)


            # access the dimension data by index, key, or shortcut
            self.parameters[i] = param # d.dim[-1]
            self.parameters[k] = param # d.dim['z']
            self.__setattr__(k, param) # d.z

    def collect(self, prop):
        """ Collects a ndim-tuple of property `prop`."""
        return tuple([self.parameters[i][prop] for i in range(self.dim)])

    def get_lengths(self):
        """Returns the physical size of the domain as a ndim-tuple."""
        return self.collect('length')


class PolarDomain(DomainBase):
    """ [NotImplemented] Class for describing polar domains in `pysit`."""

    pass

class DomainBC(object):
    """ Base class for domain boundary conditions."""
    type = None

class Dirichlet(DomainBC):
    """ Homogeneous Dirichlet domain boundary condition. """
    type = 'dirichlet'

class Neumann(DomainBC):
    """ Neumann domain boundary condition. [Temporarily not supported.]"""
    type = 'neumann'

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Neumann boundary support has been temporarily dropped.')

class PML(DomainBC):
    """ Perfectly Matched Layer (PML) domain boundary condition.

    Parameters
    ----------

    length : float
        Size of the PML in physical units.

    amplitude : float
        Scaling factor for the PML coefficient.

    ftype : {'quadratic', 'b-spline'}, optional
        PML function profile.  Defaults to `'quadratic'`

    Examples
    --------

    >>> pmlx = PML(0.1, 100)

    """

    type = 'pml'

    def __init__(self, length, amplitude, ftype='quadratic', boundary='dirichlet',compact=False):
        # Length is currently in physical units.
        self.length = length

        self.amplitude = amplitude

        #flag which constructs the compact operator
        self.compact = compact

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
        """Evaluates the PML profile function on `n` points over the range [0,1].

        Parameters
        ----------

        n : int

        orientation : {'left', 'right'}
            Orients the direction of increase.  Defaults to `'right`'.


        """

        val = self.amplitude * self.pml_func(np.linspace(0., 1., n))
        if orientation is 'left':
            val = val[::-1]

        return val

#   def evaluate(self, XX, base_point, orientation):
#       """Evaluates the PML on the given array.
#
#       Parameters
#       ----------
#       XX : ndarray
#           Grid of points on which to evaluate the PML.
#       base_point : float
#           The spatial (physical coordinate) location that the PML begins.
#       orientation : {'left', 'right'}
#           The direction in which the PML is oriented.
#
#       Notes
#       -----
#       Because a PML object is intended to describe the PML in a single
#       direction, any XX can be provided.  The programmer must be sure that
#       XX corresponds with the correct dimension.
#
#       Examples
#       --------
#       >>> d = Domain(((0.0, 1.0, 100),(0.0, 0.5, 50)),
#                        pml=(PML(0.1, 100),PML(0.1, 100)))
#       >>> XX, ZZ = d.generate_grid()
#       >>> pml_mtx = d.x.pml.evaluate(XX)
#
#       """
#       pml = np.zeros_like(XX)
#
#       if orientation == 'left':
#           loc = np.where((XX < base_point) & (XX >= base_point-self.size))
#           pml[loc] = self.amplitude * self.pml_func( (base_point - XX[loc]) / self.size)
#       if orientation == 'right':
#           loc = np.where((XX >= base_point) & (XX < base_point+self.size))
#           pml[loc] = self.amplitude * self.pml_func( (XX[loc] - base_point) / self.size)
#
#       # candidate for sparse matrix
#       return pml