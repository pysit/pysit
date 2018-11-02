.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_core_mesh:

************************
Mesh (`pysit.core.mesh`)
************************


Introduction
============

A  PySIT Mesh object specifies the computational properties of the domain
being imaged.  For example, this may include grid spacing, node count, units,
etc.

Like everything in PySIT, meshes are specified in physical coordinates.
Currently, PySIT supports :ref:`pysit_core_mesh_structured_cartesian` through
the `pysit.core.mesh.CartesianMesh` class.

Boundary conditions are also specified for each mesh boundary.  Currently,
PySIT supports homogeneous Dirichlet boundaries
(`~pysit.core.mesh.StructuredDirichlet`) and perfectly matched layers
(`~pysit.core.mesh.StructuredPML`) on `~pysit.core.mesh.CartesianMesh` meshes.


Getting Started
===============

Constructing a mesh requires a domain.

.. code:: python

    from pysit import RectangularDomain, PML
    pml = PML(0.1, 100)
    x_config = (0.0, 1.0, pml, pml)
    z_config = (0.0, 1.0, pml, pml)
    domain = RectangularDomain(x_config, z_config)

Then, import the relevant class:

.. code:: python

    from pysit import CartesianMesh

Next, construct the actual mesh.  The mesh infers the problem dimension from
the domain and the number of arguments after it.  An integer number of nodes
is required for *each* dimension.  Both endpoints are included in the count.

.. code:: python

    nx = 101
    nz = 101
    mesh = CartesianMesh(domain, nx, nz)


Using `~pysit.core.mesh`
========================

.. _pysit_core_mesh_structured_cartesian:

Structured Cartesian Meshes
---------------------------

Storage of Dimensional Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the `~pysit.core.mesh.CartesianMesh` class, information about a dimension
is stored as an entry in the ``parameters`` dictionary attribute.  Each
dimension has the following keys:

1. ``n``: an integer number of points for the dimension
2. ``delta``: the distance between points
3. ``lbc``: the mesh description of the left boundary condition
4. ``rbc``: the mesh description of the the right boundary condition

The coordinate system is stored in a left-handed ordering (the positive
z-direction points downward).  In any iterable which depends on dimension, the
z-dimension is always the *last* element.  Thus, in 1D, the dimension is
assumed to be z.

The negative direction is always referred to as the *left* side and the
positive direction is always the *right* side.  For example, for the
z-dimension, our intuitive top dimension is referred to as the left and the
bottom as the right.

The parameters dictionary can be accessed by number, by letter, or in the
style of an attribute of the `~pysit.core.mesh.CartesianMesh`.  E.g.,

.. code:: python

    # Number
    mesh.parameters[2] # Assume 3D, z is last entry

    # Letter
    mesh.parameters['z']

    # Attribute-style
    mesh.z

The boundary condition information is given in the `lbc` and `rbc` properties
of each dimension.  For example, to get the number of padding nodes the right
PML,

.. code:: python

    print mesh.z.rbc.length

A big key of PySIT is that most code is dimension independent.  To iterate
over the dimensions of a mesh, without knowing their number ahead of time,

.. code:: python

    # Print the number of nodes in each dimension
    for i in xrange(mesh.dim):
        print mesh[i].n

Key Methods
^^^^^^^^^^^

.. automethod:: pysit.core.mesh.CartesianMesh.shape
    :noindex:

.. automethod:: pysit.core.mesh.CartesianMesh.mesh_coords
    :noindex:

.. automethod:: pysit.core.mesh.CartesianMesh.pad_array
    :noindex:

.. automethod:: pysit.core.mesh.CartesianMesh.unpad_array
    :noindex:

.. Extending `mesh`
.. ==================

.. Define the required interfaces for mesh objects


Reference/API
=============

.. automodapi:: pysit.core.mesh
    :no-inheritance-diagram: