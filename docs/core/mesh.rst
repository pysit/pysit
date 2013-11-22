.. Based on _pkgtemplate.rst from Astropy, Licensed under a 3-clause BSD style
.. license - see ASTROPY_SPHINXEXT_LICENSES.rst

.. Licensed under a 3-clause BSD style license - see LICENSE.rst

.. _pysit_core_mesh:

************************
Mesh (`pysit.core.mesh`)
************************


Introduction
============

A  PySIT `~pysit.core.mesh.Mesh` object specifies the computational properties
of the domain being imaged.  For example, this may include grid spacing, node
count, units, etc.


Getting Started
===============

* Importing a mesh object
* Constructing a mesh from a domain


Using `mesh`
============

.. THIS SECTION SHOULD BE EITHER

More details

Storage of Dimensional Information
----------------------------------

Cartesian Meshs
^^^^^^^^^^^^^^^

* What information is stored in a dimension (delta, node count, boundary conditions, etc)
* Accessing single dimension by name and by index
* Accessing boundary condition information
* Iterating over dimensions

Key Methods
-----------

* shape
* mesh_coords
* dof

Extending `mesh`
==================

Define the required interfaces for mesh objects


Reference/API
=============

.. automodapi:: pysit.core.mesh
    :no-inheritance-diagram: