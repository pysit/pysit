.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_core_domain:

****************************
Domain (`pysit.core.domain`)
****************************


Introduction
============

A  PySIT Domain object specifies the physical properties of the domain being
imaged, which may include properties like width, depth, physical units,
origin, etc.

Like everything in PySIT, domains are specified in physical coordinates.
Currently, PySIT supports `Cartesian Domains` through the
`pysit.core.domain.RectangularDomain` class.

Boundary conditions are also specified for each domain boundary.  Currently,
PySIT supports homogeneous Dirichlet boundaries
(`~pysit.core.domain.Dirichlet`) and perfectly matched layers
(`~pysit.core.domain.PML`) on `~pysit.core.domain.RectangularDomain` domains.


Getting Started
===============

Import the relevant classes from PySIT:

.. code:: python

    from pysit import RectangularDomain, PML, Dirichlet

To construct a `~pysit.core.domain.RectangularDomain`, properties for each
dimension must be specified:

.. code:: python

    z_config = (0.0, 1.0, Dirichlet(), PML(0.1, 100))

Here we are specifying a 1D problem on the domain :math:`[0,1]` with a
Dirichlet boundary on the left (top, if you are thinking depth) and an
absorbing layer of width 0.1 and intensity 100 on the right (bottom).

Then, construct the domain:

.. code:: python

    domain = RectangularDomain(z_config)

The `~pysit.core.domain.RectangularDomain` class automatically infers problem
dimension from the number of configurations it receives.  Thus, to setup a 2D
problem,

.. code:: python

    # Define a configuration for the x-dimension
    pml = PML(0.1, 100)
    x_config = (0.0, 1.0, pml, pml)

    # Construct 2D domain
    domain_2D = RectangularDomain(x_config, z_config)


Using `~pysit.core.domain`
==========================

.. THIS SECTION SHOULD BE EITHER

Domain Configuration Tuples
---------------------------

The fundamental properties of the domain are specified using a `tuple`
containing the basic properties of that dimension, or axis, e.g.,

    1. left boundary position,
    2. right boundary position,
    3. left boundary condition,
    4. right boundary condition,
    5. physical unit, (optional).

The formal specification of the configuration will depend on the type of
domain.  The left and right boundary positions are in physical coordinates.
The left and right boundary conditions are instances of the subclasses of
`pysit.core.domain.DomainBC`.

The `~pysit.core.domain.RectangularDomain` class infers the dimension of the
problem from the number of configuration tuples passed to its constructor.

The boundary condition is the physical specification of that boundary, not the
numerical specification.  For example, setting the boundary as
`PML(0.1, 100)` does not specify the width in grid points,
rather the physical width in, e.g., meters.

.. note::

    In future versions, the names of the boundary condition classes might change
    to better match their physical or mathematical meaning.  E.g.,
    `~pysit.core.domain.PML` might become `Absorbing` or `SommerfeldRadiation` as
    appropriate.

Storage of Dimensional Information
----------------------------------

Cartesian Domains
^^^^^^^^^^^^^^^^^

For the `~pysit.core.domain.RectangularDomain` class:

Information about a dimension is stored as an entry in the `parameters`
dictionary attribute.  Each dimension has the following keys:

1. `lbound`: a float with the closed left boundary of the domain
2. `rbound`: a float with the open right boundary of the domain
3. `lbc`: the left boundary condition
4. `rbc`: the right boundary condition
5. `unit`: a string with the physical units of the dimension, e.g., 'm'
6. `length`: a float with the length of the domain, `rbound`-`lbound`

The coordinate system is stored in a left-handed ordering (the positive
z-direction points downward).  In any iterable which depends on dimension, the
z-dimension is always the *last* element.  Thus, in 1D, the dimension is
assumed to be z.

The negative direction is always referred to as the *left* side and the
positive direction is always the *right* side.  For example, for the
z-dimension, our intuitive top dimension is referred to as the left and the
bottom as the right.

The parameters dictionary can be accessed by number, by letter, or in the
style of an attribute of the `~pysit.core.domain.RectangularDomain`.  E.g.,

.. code:: python

    # Number
    domain.parameters[2] # Assume 3D, z is last entry

    # Letter
    domain.parameters['z']

    # Attribute-style
    domain.z

The boundary condition information is given in the `lbc` and `rbc` properties
of each dimension.  For example, to get the length of the right PML,

.. code:: python

    print domain.z.rbc.length

A big key of PySIT is that most code is dimension independent.  To iterate
over the dimensions of a domain, without knowing their number ahead of time,

.. code:: python

    # Print the length of each dimension
    for i in xrange(domain.dim):
        print domain[i].length


.. Extending `domain`
.. ==================
..
.. Define the required interfaces.
..

Reference/API
=============

.. automodapi:: pysit.core.domain
    :no-inheritance-diagram: